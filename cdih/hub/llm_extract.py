"""LLM extraction + narrative summary, provider-agnostic.

Providers (first key found wins, free tiers first; force with CDIH_PROVIDER):
  gemini      GEMINI_API_KEY / GOOGLE_API_KEY   (Google AI Studio free tier)
  groq        GROQ_API_KEY                      (free tier)
  openrouter  OPENROUTER_API_KEY                (":free" models)
  anthropic   ANTHROPIC_API_KEY
Override the model with CDIH_MODEL. Any failure falls back to the rules engine.
Anthropic uses forced tool-use; the others use OpenAI-compatible JSON mode.
"""
from __future__ import annotations
import json, os, re
from . import rules_extract

PROVIDERS = {
    "gemini": {"keys": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "model": "gemini-2.5-flash"},
    "groq": {"keys": ("GROQ_API_KEY",), "base_url": "https://api.groq.com/openai/v1",
             "model": "llama-3.3-70b-versatile"},
    "openrouter": {"keys": ("OPENROUTER_API_KEY",), "base_url": "https://openrouter.ai/api/v1",
                   "model": "meta-llama/llama-3.3-70b-instruct:free"},
    "anthropic": {"keys": ("ANTHROPIC_API_KEY",), "base_url": None, "model": "claude-sonnet-5-5"},
}


FIELD = {"type": "object", "properties": {
    "value": {}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "evidence": {"type": "string", "description": "verbatim snippet the value came from"}},
    "required": ["value", "confidence", "evidence"]}
MED = {"type": "object", "properties": {
    "name": {"type": "string"}, "dose": {"type": "string", "description": "e.g. '40 mg'"},
    "route": {"type": ["string", "null"]},
    "frequency": {"type": ["string", "null"], "description": "OD, BD, TDS, PRN, ON"},
    "confidence": {"type": "number"}, "evidence": {"type": "string"}},
    "required": ["name", "dose", "confidence", "evidence"]}
LAB = {"type": "object", "properties": {
    "test": {"type": "string"}, "value": {"type": "number"}, "unit": {"type": "string"},
    "ref_low": {"type": ["number", "null"]}, "ref_high": {"type": ["number", "null"]},
    "flag": {"enum": ["HIGH", "LOW", "NORMAL"]}, "confidence": {"type": "number"}, "evidence": {"type": "string"}},
    "required": ["test", "value", "unit", "flag", "confidence", "evidence"]}
SCHEMA = {"type": "object", "properties": {
    "doc_type": FIELD,
    "patient": {"type": "object", "properties": {k: FIELD for k in ("name", "dob", "nhs_number", "sex")},
                "description": "dob as YYYY-MM-DD"},
    "dates": {"type": "object", "properties": {k: FIELD for k in ("admission", "discharge", "collected", "document_date")},
              "description": "ISO YYYY-MM-DD. Source documents use UK DD/MM/YYYY."},
    "author": {"anyOf": [FIELD, {"type": "null"}]},
    "primary_diagnosis": {"anyOf": [FIELD, {"type": "null"}]},
    "secondary_diagnoses": {"type": "array", "items": FIELD},
    "allergies": {"type": "array", "items": FIELD},
    "reason_for_visit": {"anyOf": [FIELD, {"type": "null"}]},
    "medications": {"type": "array", "items": MED},
    "labs": {"type": "array", "items": LAB},
    "vitals": {"type": "object", "properties": {k: FIELD for k in ("bp", "hr", "spo2", "temp_c", "weight_kg")}},
    "follow_up": {"type": "array", "items": {"type": "string"}},
    "plan_items": {"type": "array", "items": {"type": "string"}},
    "social": {"type": "object", "properties": {"lives_alone": FIELD}},
    "falls_last_3m": {"anyOf": [FIELD, {"type": "null"}]},
    "free_text_flags": {"type": "array", "items": {"type": "string"},
                        "description": "from: walking stick, falls risk, poor appetite, dizzy"}},
    "required": ["doc_type", "patient", "dates", "medications", "labs", "vitals"]}

SYSTEM = ("You extract structured data from clinical documents. Extract ONLY what is stated in the "
          "document; never infer or invent values. Use null/omit when absent. Confidence reflects how "
          "explicit and unambiguous the source text is (1.0 = verbatim labelled field, <0.6 = guessed/"
          "unclear/OCR noise). Evidence must be a verbatim snippet. Dates are UK DD/MM/YYYY in the source; "
          "return ISO. The documents are synthetic.")



EXAMPLE = (
    'Return ONLY one JSON object (no prose, no markdown) shaped like: '
    '{"doc_type":F,"patient":{"name":F,"dob":F,"nhs_number":F,"sex":F},'
    '"dates":{"admission":F,"discharge":F,"collected":F,"document_date":F},'
    '"author":F|null,"primary_diagnosis":F|null,"secondary_diagnoses":[F],"allergies":[F],'
    '"reason_for_visit":F|null,"medications":[{"name","dose":"40 mg","route","frequency":"OD|BD|TDS|PRN|ON",'
    '"confidence","evidence"}],"labs":[{"test","value":number,"unit","ref_low","ref_high",'
    '"flag":"HIGH|LOW|NORMAL","confidence","evidence"}],'
    '"vitals":{"bp":F,"hr":F,"spo2":F,"temp_c":F,"weight_kg":F},"follow_up":[str],"plan_items":[str],'
    '"social":{"lives_alone":F},"falls_last_3m":F|null,"free_text_flags":[str]} '
    'where F = {"value":...,"confidence":0-1,"evidence":"verbatim snippet"}. '
    'Omit keys that are absent. doc_type is one of: Discharge summary, Lab report, Physician note, Intake form.')


def active():
    """(provider, model, api_key) for the provider that will be used, or None."""
    order = [os.environ["CDIH_PROVIDER"]] if os.environ.get("CDIH_PROVIDER") in PROVIDERS else list(PROVIDERS)
    for name in order:
        for k in PROVIDERS[name]["keys"]:
            if os.environ.get(k):
                return name, os.environ.get("CDIH_MODEL", PROVIDERS[name]["model"]), os.environ[k]
    return None


def available() -> bool:
    return active() is not None


def label() -> str:
    a = active()
    return f"{a[0]}:{a[1]}" if a else "rules"


def _client():
    name, _, key = active()
    if name == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=key)
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=PROVIDERS[name]["base_url"], timeout=60)


def _model():
    a = active()
    return a[1] if a else os.environ.get("CDIH_MODEL", "claude-sonnet-5-5")


def _is_openai_style(client) -> bool:
    return hasattr(client, "chat")


def _parse_json(s: str) -> dict:
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    i, j = s.find("{"), s.rfind("}")
    return json.loads(s[i:j + 1])


def _chat(client, system, user, max_tokens, temperature, as_json):
    """OpenAI-compatible call; retries without response_format if the provider rejects it."""
    kw = dict(model=_model(), temperature=temperature, max_tokens=max_tokens,
              messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if as_json:
        try:
            return client.chat.completions.create(response_format={"type": "json_object"}, **kw)
        except Exception:
            pass
    return client.chat.completions.create(**kw)


def _llm_json(client, text: str) -> dict:
    if _is_openai_style(client):
        r = _chat(client, SYSTEM + " " + EXAMPLE, f"<document>\n{text}\n</document>", 4096, 0, True)
        return _parse_json(r.choices[0].message.content)
    resp = client.messages.create(
        model=_model(), max_tokens=4096, temperature=0, system=SYSTEM,
        tools=[{"name": "record_extraction", "description": "Record structured extraction",
                "input_schema": SCHEMA}],
        tool_choice={"type": "tool", "name": "record_extraction"},
        messages=[{"role": "user", "content": f"<document>\n{text}\n</document>"}])
    return next(b.input for b in resp.content if b.type == "tool_use")


def _field(f):
    """Coerce to {'value','confidence','evidence'} or None."""
    if isinstance(f, dict) and f.get("value") not in (None, ""):
        try: c = float(f.get("confidence", 0.6))
        except Exception: c = 0.6
        return {"value": f["value"], "confidence": round(min(max(c, 0), 1), 2), "evidence": str(f.get("evidence", ""))}
    return None


def _harden(d: dict, text: str) -> dict:
    """Validate/coerce LLM output; cross-check against the deterministic extractor.
    Items the rules found but the model missed (labs/meds) are added at reduced confidence."""
    d = dict(d)
    for k in ("patient", "dates", "vitals", "social"):
        d[k] = {a: fb for a, b in (d.get(k) or {}).items() if (fb := _field(b))}
    for k in ("author", "primary_diagnosis", "reason_for_visit", "falls_last_3m"):
        d[k] = _field(d.get(k))
    d["doc_type"] = _field(d.get("doc_type")) or {"value": "Unknown", "confidence": 0.3, "evidence": ""}
    for k in ("secondary_diagnoses", "allergies"):
        d[k] = [fb for x in (d.get(k) or []) if (fb := _field(x))]
    meds, labs = [], []
    for m in d.get("medications") or []:
        if isinstance(m, dict) and m.get("name") and m.get("dose"):
            m["name"] = str(m["name"]).strip().capitalize(); m["dose"] = str(m["dose"])
            m["confidence"] = float(m.get("confidence", 0.6)); m.setdefault("evidence", "")
            meds.append(m)
    for l in d.get("labs") or []:
        try:
            l["value"] = float(l["value"]); l["confidence"] = float(l.get("confidence", 0.6))
            l.setdefault("evidence", ""); l.setdefault("flag", "NORMAL")
            labs.append(l)
        except Exception:
            continue
    ref = rules_extract.extract(text)
    have_m = {m["name"].lower() for m in meds}; have_l = {l["test"].lower() for l in labs}
    for m in ref["medications"]:
        if m["name"].lower() not in have_m:
            meds.append({**m, "confidence": round(m["confidence"] * 0.85, 2)})
    for l in ref["labs"]:
        if l["test"].lower() not in have_l:
            labs.append({**l, "confidence": round(l["confidence"] * 0.85, 2)})
    d["medications"], d["labs"] = meds, labs
    if not d["patient"].get("name"):
        raise ValueError("no patient name extracted")
    return d


def extract(text: str, client=None) -> dict:
    """LLM extraction; on failure returns rules result tagged with the error."""
    try:
        client = client or _client()
        out = _normalise(_harden(_llm_json(client, text), text))
        out["extractor"] = f"llm ({label()})"
        return out
    except Exception as e:  # network, auth, quota, bad JSON...
        out = rules_extract.extract(text)
        out["extractor"] = f"rules (LLM failed: {type(e).__name__})"
        return out

def _normalise(d: dict) -> dict:
    base = {"doc_type": d["doc_type"], "patient": d.get("patient") or {}, "dates": d.get("dates") or {},
            "author": d.get("author"), "primary_diagnosis": d.get("primary_diagnosis"),
            "secondary_diagnoses": d.get("secondary_diagnoses") or [], "allergies": d.get("allergies") or [],
            "reason_for_visit": d.get("reason_for_visit"), "medications": d.get("medications") or [],
            "labs": d.get("labs") or [], "vitals": d.get("vitals") or {}, "follow_up": d.get("follow_up") or [],
            "plan_items": d.get("plan_items") or [], "social": d.get("social") or {},
            "falls_last_3m": d.get("falls_last_3m"), "free_text_flags": d.get("free_text_flags") or [],
            "extractor": "llm"}
    for k in ("patient", "dates", "vitals", "social"):
        base[k] = {a: b for a, b in base[k].items() if b}
    for l in base["labs"]:
        l.setdefault("ref_low", None); l.setdefault("ref_high", None)
    for m in base["medications"]:
        m.setdefault("route", None); m.setdefault("frequency", None)
    return base




def narrative(case: dict, client=None) -> str | None:
    """Short clinician-facing narrative grounded ONLY in the structured case."""
    if not available() and client is None:
        return None
    system = ("You write concise handover summaries for clinicians from structured data. "
              "Use ONLY the supplied JSON. 120 words max, plain prose, no new facts, "
              "no diagnoses beyond those listed. End with: 'For clinician review.'")
    payload = json.dumps({k: case[k] for k in ("patient", "age", "diagnoses", "allergies", "triage")} |
                         {"top_flags": [f["title"] for f in case["flags"][:8]]}, default=str)
    try:
        client = client or _client()
        if _is_openai_style(client):
            r = _chat(client, system, payload, 600, 0.2, False)
            return r.choices[0].message.content.strip()
        r = client.messages.create(model=_model(), max_tokens=600, temperature=0.2, system=system,
                                   messages=[{"role": "user", "content": payload}])
        return r.content[0].text.strip()
    except Exception:
        return None

def template_narrative(case: dict) -> str:
    p = case["patient"]; t = case["triage"]
    high = [f["title"] for f in case["flags"] if f["severity"] in ("CRITICAL", "HIGH")]
    dx = case["diagnoses"]
    return (f"{p.get('name','Patient')}, {case['age']}-year-old {str(p.get('sex','')).lower()}, "
            f"admitted with {dx[0].lower() if dx else 'an unspecified condition'}"
            f"{' (also: ' + ', '.join(d.lower() for d in dx[1:4]) + ')' if len(dx) > 1 else ''}. "
            f"Allergies: {', '.join(case['allergies']) or 'none recorded'}. "
            f"Cross-document review raised {len(high)} high-priority issue(s), notably: "
            f"{'; '.join(high[:4])}. Suggested triage: {t['level']} - {t['timeframe'].lower()}. "
            f"For clinician review.")
