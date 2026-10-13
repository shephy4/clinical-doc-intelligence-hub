from __future__ import annotations
from . import ingest, rules_extract, llm_extract, reasoning


def process(files: list[tuple[str, bytes]], use_llm: bool | None = None):
    """files: [(filename, bytes)] -> (records, case)"""
    use_llm = llm_extract.available() if use_llm is None else use_llm
    records = []
    for name, data in files:
        text, method = ingest.read_document(name, data)
        rec = llm_extract.extract(text) if use_llm else rules_extract.extract(text)
        rec["_file"], rec["_ingest"], rec["_text"] = name, method, text
        records.append(rec)
    case = reasoning.build_case(records)
    case["narrative"] = (llm_extract.narrative(case) if use_llm else None) or llm_extract.template_narrative(case)
    case["next_steps"] = reasoning.next_steps(case)
    case["extractor"] = f"LLM ({llm_extract.label()})" if use_llm else "Rule-based (offline)"
    return records, case


def confidence_summary(rec):
    """Mean confidence per document + list of low-confidence fields."""
    vals, low = [], []
    def visit(label, f):
        if isinstance(f, dict) and "confidence" in f:
            vals.append(f["confidence"])
            if f["confidence"] < 0.8: low.append((label, f["value"], f["confidence"]))
    for k in ("doc_type", "author", "primary_diagnosis", "reason_for_visit", "falls_last_3m"):
        visit(k, rec.get(k))
    for g in ("patient", "dates", "vitals", "social"):
        for k, f in rec[g].items(): visit(f"{g}.{k}", f)
    for g in ("secondary_diagnoses", "allergies"):
        for f in rec[g]: visit(g, f)
    for g, nm in (("medications", "name"), ("labs", "test")):
        for f in rec[g]: visit(f"{g}:{f[nm]}", {"value": f.get("dose", f.get("value")), "confidence": f["confidence"]})
    return (round(sum(vals) / len(vals), 3) if vals else 0.0), low
