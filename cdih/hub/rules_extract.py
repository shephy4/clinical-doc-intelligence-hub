"""Offline, deterministic extractor. Used when no LLM key is set and as a
cross-check / fallback. Every value carries a confidence and evidence snippet."""
from __future__ import annotations
import re

FREQ = {"once daily": "OD", "twice daily": "BD", "three times daily": "TDS",
        "daily": "OD", "as needed": "PRN", "at night": "ON"}


def _f(value, conf, ev):
    return {"value": value, "confidence": round(conf, 2), "evidence": ev}


def classify(text: str) -> tuple[str, float]:
    head = text[:300].upper()
    for key, label in [("DISCHARGE SUMMARY", "Discharge summary"),
                       ("LABORATORY REPORT", "Lab report"),
                       ("PROGRESS NOTE", "Physician note"),
                       ("INTAKE FORM", "Intake form")]:
        if key in head:
            return label, 0.97
    low = text.lower()
    scores = {"Lab report": low.count("reference range") + low.count("umol/l") + low.count("mmol/l"),
              "Discharge summary": low.count("discharge") + low.count("admission"),
              "Physician note": low.count("assessment") + low.count("plan:"),
              "Intake form": low.count("intake") + low.count("current medications")}
    best = max(scores, key=scores.get)
    return (best, 0.6) if scores[best] else ("Unknown", 0.3)


def _line(text, label):
    m = re.search(rf"^\s*{label}\s*:\s*(.+)$", text, re.I | re.M)
    return (m.group(1).strip(), m.group(0).strip()) if m else (None, None)


def _section(text, label):
    m = re.search(rf"^\s*{label}\s*:\s*\n?(.*?)(?=\n\s*\n|\n[A-Z][A-Za-z ]+:\s|\Z)", text, re.I | re.M | re.S)
    return m.group(1).strip() if m else ""


def _split(s):
    return [x.strip() for x in re.split(r";|,(?![^()]*\))", s) if x.strip()]


def _parse_range(s):
    s = s.replace("–", "-")
    m = re.match(r"\s*([\d.]+)\s*-\s*([\d.]+)", s)
    if m: return float(m.group(1)), float(m.group(2))
    m = re.match(r"\s*<\s*([\d.]+)", s)
    if m: return None, float(m.group(1))
    m = re.match(r"\s*>\s*([\d.]+)", s)
    if m: return float(m.group(1)), None
    return None, None


def extract_meds(text):
    meds, seen = [], set()
    pat = re.compile(r"([A-Z][a-z]+)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b\s*(oral\s+)?"
                     r"(once daily|twice daily|three times daily|daily|as needed|at night)?", re.I)
    zone = ""
    for label in ("Discharge Medications", "Current Medications", "Medications"):
        zone = _section(text, label)
        if zone: break
    for m in pat.finditer(zone):
        name = m.group(1).capitalize()
        if name.lower() in seen and False: continue
        freq = FREQ.get((m.group(5) or "").lower(), None)
        conf = 0.95 if freq else 0.8
        meds.append({"name": name, "dose": f"{m.group(2)} {m.group(3).lower()}",
                     "route": "oral" if m.group(4) else None, "frequency": freq,
                     "confidence": conf, "evidence": m.group(0).strip()})
    return meds


def extract_labs(text):
    labs = []
    for ln in text.splitlines():
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) == 4 and re.match(r"^-?[\d.]+$", parts[1]):
            lo, hi = _parse_range(parts[3])
            v = float(parts[1])
            flag = "HIGH" if hi is not None and v > hi else "LOW" if lo is not None and v < lo else "NORMAL"
            labs.append({"test": parts[0], "value": v, "unit": parts[2], "ref_low": lo,
                         "ref_high": hi, "flag": flag, "confidence": 0.96, "evidence": ln.strip()})
    return labs


def extract_vitals(text):
    v = {}
    for key, pat, conv in [
        ("bp", r"BP(?: today)?[:\s]+(\d{2,3})/(\d{2,3})", lambda m: f"{m.group(1)}/{m.group(2)}"),
        ("hr", r"HR[:\s]+(\d{2,3})", lambda m: int(m.group(1))),
        ("spo2", r"SpO2[:\s]+(\d{2,3})", lambda m: int(m.group(1))),
        ("temp_c", r"Temp[:\s]+([\d.]+)", lambda m: float(m.group(1))),
        ("weight_kg", r"Weight(?: today)?[:\s]+([\d.]+)\s*kg", lambda m: float(m.group(1)))]:
        m = re.search(pat, text, re.I)
        if m: v[key] = _f(conv(m), 0.93, m.group(0))
    return v


def _date(s):
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})", s or "")
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None


def extract(text: str) -> dict:
    doc_type, tconf = classify(text)
    out = {"doc_type": _f(doc_type, tconf, text[:60].strip()), "patient": {}, "dates": {}}
    for key, label in [("name", "Patient Name"), ("dob", "DOB"), ("nhs_number", "NHS Number"), ("sex", "Sex")]:
        val, ev = _line(text, label)
        if val:
            if key == "dob": val = _date(val)
            if key == "nhs_number": val = re.sub(r"\s*\(synthetic\)", "", val)
            out["patient"][key] = _f(val, 0.97, ev)
    for key, label in [("admission", "Date of Admission"), ("discharge", "Date of Discharge"),
                       ("collected", "Collected"), ("document_date", "Date")]:
        val, ev = _line(text, label)
        if val and _date(val): out["dates"][key] = _f(_date(val), 0.95, ev)
    out["author"] = None
    for label in ("Consultant", "Author"):
        val, ev = _line(text, label)
        if val: out["author"] = _f(val, 0.9, ev); break
    prim, ev = _line(text, "Primary Diagnosis")
    out["primary_diagnosis"] = _f(prim, 0.93, ev) if prim else None
    sec, ev2 = _line(text, "Secondary Diagnoses")
    out["secondary_diagnoses"] = [_f(x, 0.9, ev2) for x in _split(sec)] if sec else []
    al, ev3 = _line(text, "Allergies")
    out["allergies"] = [_f(x, 0.95, ev3) for x in _split(al)] if al else []
    rv, ev4 = _line(text, "Reason for Visit")
    out["reason_for_visit"] = _f(rv, 0.9, ev4) if rv else None
    out["medications"] = extract_meds(text)
    out["labs"] = extract_labs(text)
    out["vitals"] = extract_vitals(text)
    out["follow_up"] = [x.strip() for x in re.split(r"(?<=\.)\s+", _section(text, "Follow-up")) if x.strip()]
    plan = _section(text, "Plan")
    out["plan_items"] = [re.sub(r"^\d+\.\s*", "", l).strip() for l in plan.splitlines() if l.strip()]
    ls, ev5 = _line(text, "Living situation")
    out["social"] = {"lives_alone": _f(True, 0.9, ev5)} if (ls and "alone" in ls.lower()) or \
        re.search(r"lives alone", text, re.I) else {}
    fm = re.search(r"Falls in last 3 months:\s*(\d+)", text, re.I)
    out["falls_last_3m"] = _f(int(fm.group(1)), 0.9, fm.group(0)) if fm else None
    out["free_text_flags"] = [k for k in ("walking stick", "falls risk", "poor appetite", "dizzy")
                              if k in text.lower()]
    out["extractor"] = "rules"
    return out
