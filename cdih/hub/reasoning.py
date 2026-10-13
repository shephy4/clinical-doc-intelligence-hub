"""Deterministic clinical reasoning over extracted records.

Illustrative decision-support rules for a PoC (NOT clinical guidance). Every flag
carries severity, rationale and the evidence documents, so a clinician can audit it.
All output is framed 'for clinician review'.
"""
from __future__ import annotations
from datetime import date

SEV_ORDER = {"CRITICAL": 3, "HIGH": 2, "MODERATE": 1, "INFO": 0}


def v(x, default=None):
    """Unwrap {'value':..} field."""
    return x["value"] if isinstance(x, dict) and "value" in x else (default if x is None else x)


def _d(s):
    try:
        y, m, d = map(int, s.split("-")); return date(y, m, d)
    except Exception:
        return None


def doc_date(rec):
    for k in ("document_date", "discharge", "collected", "admission"):
        if k in rec["dates"]: return _d(v(rec["dates"][k]))
    return None


def merge_patient(records):
    """Check identity consistency and build a single patient header."""
    names = {v(r["patient"].get("name")).strip().lower() for r in records if r["patient"].get("name")}
    dobs = {v(r["patient"].get("dob")) for r in records if r["patient"].get("dob")}
    nhs = {v(r["patient"].get("nhs_number")) for r in records if r["patient"].get("nhs_number")}
    warnings = []
    if len(names) > 1 or len(dobs) > 1 or len(nhs) > 1:
        warnings.append("Identity mismatch across documents (name/DOB/NHS number differ) - "
                        "verify these belong to the same patient before using the combined view.")
    hdr = {}
    for r in records:
        for k, f in r["patient"].items():
            hdr.setdefault(k, v(f))
    return hdr, warnings


def age_on(dob, on):
    b = _d(dob) if dob else None
    if not b or not on: return None
    return on.year - b.year - ((on.month, on.day) < (b.month, b.day))


def latest(records, getter):
    """Latest non-null value across docs (by document date)."""
    best = None
    for r in sorted(records, key=lambda r: doc_date(r) or date.min):
        val = getter(r)
        if val is not None: best = (val, r)
    return best


def _num(med_dose):
    try: return float(str(med_dose).split()[0])
    except Exception: return None


def find_med(records, name):
    """Latest listing of a drug -> (med, record)"""
    out = None
    for r in sorted(records, key=lambda r: doc_date(r) or date.min):
        for m in r["medications"]:
            if m["name"].lower() == name.lower(): out = (m, r)
    return out


def flags_for(records, hdr):
    flags = []

    def add(sev, title, why, sources, action=None):
        flags.append({"severity": sev, "title": title, "rationale": why, "sources": sources, "action": action})

    today = max((doc_date(r) for r in records if doc_date(r)), default=None)
    age = age_on(hdr.get("dob"), today)
    labs = {}
    for r in sorted(records, key=lambda r: doc_date(r) or date.min):
        for l in r["labs"]:
            labs[l["test"].lower()] = (l, r)
    lab = lambda n: labs.get(n.lower(), (None, None))[0]

    # --- lab flags
    k = lab("potassium")
    if k:
        if k["value"] >= 6.0:
            add("CRITICAL", f"Severe hyperkalaemia (K+ {k['value']:g} {k['unit']})", "K+ >= 6.0 needs same-day review/ECG.", ["Lab report"])
        elif k["value"] > (k["ref_high"] or 99):
            add("HIGH", f"Raised potassium (K+ {k['value']:g} {k['unit']}, ref <= {k['ref_high']})", "Hyperkalaemia on the supplied lab report; repeat/recheck needed.", ["Lab report"], "Repeat U&E today/tomorrow; review ramipril and any potassium-raising drugs.")
    egfr = lab("egfr")
    if egfr and egfr["value"] < 45:
        add("HIGH" if egfr["value"] < 45 else "MODERATE", f"Reduced kidney function (eGFR {egfr['value']:g})",
            "eGFR < 45 affects dosing of renally cleared drugs.", ["Lab report"], "Pharmacist renal-dose review of all medications.")
    hb = lab("haemoglobin")
    if hb and hb["value"] < 110:
        add("MODERATE", f"Anaemia (Hb {hb['value']:g} g/L)", "Below reference range; relevant with anticoagulation and heart failure.", ["Lab report"])
    bnp = lab("nt-probnp")
    if bnp and bnp["value"] > 2000:
        add("MODERATE", f"Markedly elevated NT-proBNP ({bnp['value']:g} ng/L)", "Consistent with significant cardiac strain.", ["Lab report"])
    a1c = lab("hba1c")
    if a1c and a1c["value"] > 58:
        add("MODERATE", f"Suboptimal diabetic control (HbA1c {a1c['value']:g} mmol/mol)", "Above typical target; review diabetes plan.", ["Lab report"])

    # --- medication safety
    ram = find_med(records, "Ramipril")
    if ram and k and k["value"] > (k["ref_high"] or 99):
        add("HIGH", "ACE inhibitor with raised potassium", "Ramipril can raise K+; physician note already asks for U&E recheck - confirm it happened.", ["Lab report", "Physician note"], "Prescriber to decide whether to hold/reduce ramipril while K+ is raised.")
    met = find_med(records, "Metformin")
    if met and egfr:
        if egfr["value"] < 30:
            add("CRITICAL", "Metformin with eGFR < 30", "Generally contraindicated; stop/review urgently.", ["Lab report", met[1]["doc_type"]["value"]])
        elif egfr["value"] < 45:
            add("HIGH", f"Metformin with reduced eGFR ({egfr['value']:g})", "Dose review recommended at eGFR 30-44; physician note flags this as unresolved.", ["Lab report", "Physician note"], "Prescriber/pharmacist to review metformin dose against eGFR.")
    apx = find_med(records, "Apixaban")
    if apx:
        weight = latest(records, lambda r: v(r["vitals"].get("weight_kg")))
        creat = lab("creatinine")
        crit = []
        if age is not None and age >= 80: crit.append("age >= 80")
        if weight and weight[0] <= 60: crit.append("weight <= 60 kg")
        if creat and creat["value"] >= 133: crit.append(f"creatinine {creat['value']:g} >= 133")
        want = 2.5 if len(crit) >= 2 else 5.0
        for r in records:
            for m in r["medications"]:
                dose = _num(m["dose"]) if m["name"].lower() == "apixaban" else None
                if dose and dose != want:
                    kind = "possible UNDER-dose" if dose < want else "possible OVER-dose"
                    add("HIGH", f"Apixaban {m['dose']} BD in {r['doc_type']['value'].lower()}: {kind}",
                        f"AF dose rule: 2.5 mg BD only if >= 2 of (age >= 80, weight <= 60 kg, creatinine >= 133). "
                        f"Criteria met: {len(crit)} ({', '.join(crit) or 'none'}) -> expected {want:g} mg BD. "
                        f"Physician note also left the dose 'to be confirmed'. Prescriber/pharmacist to verify.",
                        [r["doc_type"]["value"]], "Prescriber to confirm correct apixaban dose and reconcile records.")
    # discrepancy across documents
    doses = {}
    for r in records:
        for m in r["medications"]:
            doses.setdefault(m["name"], {})[r["doc_type"]["value"]] = m["dose"]
    already = {f["title"].split()[0].lower() for f in flags if f["title"].startswith("Apixaban")}
    for name, by in doses.items():
        if len(set(by.values())) > 1 and name.lower() not in already:
            add("HIGH", f"{name}: dose differs between documents", "; ".join(f"{d}: {x}" for d, x in by.items()) + " - reconcile with prescriber.", list(by), f"Reconcile {name} dose across records with prescriber/pharmacist.")
    # NSAID
    for r in records:
        for m in r["medications"]:
            if m["name"].lower() in {"ibuprofen", "naproxen", "diclofenac"}:
                why = "NSAIDs can worsen heart failure and kidney function, and raise bleeding risk with an anticoagulant."
                add("HIGH", f"NSAID on medication list ({m['name']})", why, [r["doc_type"]["value"]], "Ask prescriber to review/stop NSAID; advise patient to avoid OTC NSAIDs.")
                break

    # --- clinical trend / status (needs 2+ docs with vitals)
    ws = [(doc_date(r), v(r["vitals"].get("weight_kg"))) for r in records if r["vitals"].get("weight_kg")]
    ws.sort(key=lambda t: t[0] or date.min)
    if len(ws) >= 2 and ws[-1][1] - ws[0][1] >= 2:
        add("HIGH", f"Weight up {ws[-1][1]-ws[0][1]:.1f} kg since discharge ({ws[0][1]} -> {ws[-1][1]} kg)",
            "Discharge advice: return if weight rises > 2 kg in 3 days. Possible fluid re-accumulation (time window not verifiable from documents).", ["Discharge summary", "Intake form"], "Same-week HF nurse/clinician assessment of fluid status and diuretic dose.")
    bps = [(doc_date(r), v(r["vitals"].get("bp"))) for r in records if r["vitals"].get("bp")]
    bps.sort(key=lambda t: t[0] or date.min)
    if bps:
        sys_ = int(bps[-1][1].split("/")[0])
        dizzy = any("dizzy" in " ".join(r["free_text_flags"]) for r in records)
        if sys_ < 110 and dizzy:
            add("HIGH", f"Low BP ({bps[-1][1]}) with dizziness on standing",
                "Possible over-diuresis/hypotension on HF medications; falls risk on an anticoagulant.", ["Intake form"], "Check lying/standing BP; review diuretic and antihypertensive doses.")
    # falls + anticoagulant
    falls = latest(records, lambda r: v(r["falls_last_3m"]))
    if falls and falls[0] >= 1 and apx:
        add("HIGH", "Recent fall while anticoagulated", f"{falls[0]} fall(s) in last 3 months on apixaban - head-injury/bleed risk and falls assessment.", ["Intake form"], "Falls assessment; counsel on head-injury warning signs while on apixaban.")
    if any(r["social"].get("lives_alone") for r in records):
        add("MODERATE", "Lives alone", "Higher readmission risk; social care assessment was requested in physician note - check outcome.", ["Physician note", "Intake form"], "Confirm social care assessment outcome and home support.")
    # follow-up gaps
    dis = next((r for r in records if r["doc_type"]["value"] == "Discharge summary"), None)
    if dis:
        later_labs = [r for r in records if r["labs"] and (doc_date(r) or date.min) > (doc_date(dis) or date.min)]
        if any("u&e" in f.lower() for f in dis["follow_up"]) and not later_labs:
            add("MODERATE", "Repeat U&E requested at discharge - no result in supplied documents",
                "Result not evidenced; with raised K+ and low eGFR this matters.", ["Discharge summary"], "Obtain/arrange the repeat U&E result.")
    return sorted(flags, key=lambda f: -SEV_ORDER[f["severity"]])


def triage(flags):
    crit = sum(f["severity"] == "CRITICAL" for f in flags)
    high = sum(f["severity"] == "HIGH" for f in flags)
    if crit:
        return {"level": "URGENT", "timeframe": "Same-day clinician review",
                "why": f"{crit} critical flag(s)."}
    if high >= 3:
        return {"level": "PRIORITY", "timeframe": "Clinician review within 24-48 hours",
                "why": f"{high} high-severity flags with interacting risks (see flags)."}
    if high:
        return {"level": "SOON", "timeframe": "Review within 1 week", "why": f"{high} high-severity flag(s)."}
    return {"level": "ROUTINE", "timeframe": "Routine follow-up", "why": "No high-severity flags found."}


def reconcile_meds(records):
    """Medication reconciliation matrix across documents."""
    docs = [r for r in sorted(records, key=lambda r: doc_date(r) or date.min) if r["medications"]]
    names = []
    for r in docs:
        for m in r["medications"]:
            if m["name"] not in names: names.append(m["name"])
    rows = []
    for n in names:
        row = {"Medication": n}
        vals = []
        for r in docs:
            m = next((m for m in r["medications"] if m["name"] == n), None)
            label = r["doc_type"]["value"]
            row[label] = f"{m['dose']} {m['frequency'] or ''}".strip() if m else "-"
            vals.append(row[label])
        present = [x for x in vals if x != "-"]
        row["Status"] = ("NEW" if vals[0] == "-" else "STOPPED" if vals[-1] == "-" else
                         "CHANGED" if len(set(present)) > 1 else "unchanged")
        rows.append(row)
    return rows


def _allergies(records):
    best = {}
    for r in records:
        for a in r["allergies"]:
            txt = v(a); base = txt.split("(")[0].strip().lower()
            if base not in best or len(txt) > len(best[base]): best[base] = txt
    return sorted(best.values())


def build_case(records):
    hdr, warnings = merge_patient(records)
    today = max((doc_date(r) for r in records if doc_date(r)), default=None)
    flags = flags_for(records, hdr)
    return {"patient": hdr, "age": age_on(hdr.get("dob"), today), "warnings": warnings,
            "flags": flags, "triage": triage(flags), "med_reconciliation": reconcile_meds(records),
            "timeline": sorted(({"date": str(doc_date(r)), "document": r["doc_type"]["value"]} for r in records),
                               key=lambda t: t["date"]),
            "diagnoses": next(([v(r["primary_diagnosis"])] + [v(x) for x in r["secondary_diagnoses"]]
                              for r in records if r["primary_diagnosis"]), []),
            "allergies": _allergies(records)}


def next_steps(case, limit=9):
    """De-duplicated, severity-ordered recommended actions (each linked to its trigger)."""
    steps, seen = [], set()
    for f in case["flags"]:
        if f["severity"] in ("CRITICAL", "HIGH") and f.get("action") and f["action"] not in seen:
            seen.add(f["action"]); steps.append(f"**{f['action']}** *(because: {f['title']})*")
    return steps[:limit]
