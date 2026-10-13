import copy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hub import pipeline, rules_extract as R, reasoning as X

S = Path(__file__).resolve().parents[1] / "samples"
FILES = [(p.name, p.read_bytes()) for p in sorted(S.iterdir()) if p.suffix in (".pdf", ".png")]


def titles(case): return [f["title"] for f in case["flags"]]


def test_all_input_types_extract():
    recs, _ = pipeline.process(FILES, use_llm=False)
    kinds = {r["doc_type"]["value"] for r in recs}
    assert kinds == {"Discharge summary", "Lab report", "Physician note", "Intake form"}
    assert {r["_ingest"] for r in recs} >= {"PDF text layer", "image (OCR)"}


def test_lab_flags_and_values():
    labs = {l["test"]: l for l in R.extract((S / "lab_report_2026-09-13.txt").read_text())["labs"]}
    assert labs["Potassium"]["flag"] == "HIGH" and labs["Potassium"]["value"] == 5.6
    assert labs["Sodium"]["flag"] == "NORMAL"          # boundary: 133 within 133-146
    assert labs["eGFR"]["flag"] == "LOW"               # '> 60' lower-bound range


def test_expected_safety_flags():
    _, case = pipeline.process(FILES, use_llm=False)
    t = " | ".join(titles(case))
    for needle in ["Raised potassium", "ACE inhibitor", "Metformin", "Apixaban 2.5 mg", "NSAID",
                   "Weight up", "Low BP", "Recent fall"]:
        assert needle in t, needle
    assert case["triage"]["level"] == "PRIORITY"


def test_apixaban_rule_boundaries():
    recs, _ = pipeline.process(FILES, use_llm=False)
    # make patient meet 2 criteria (age>=80) -> 2.5 mg becomes correct, 5 mg becomes over-dose
    r2 = copy.deepcopy(recs)
    for r in r2:
        if "dob" in r["patient"]: r["patient"]["dob"]["value"] = "1940-01-01"
    case = X.build_case(r2)
    t = " | ".join(titles(case))
    assert "5 mg BD in intake form: possible OVER-dose" in t
    assert "2.5 mg BD in discharge summary" not in t


def test_identity_mismatch_warning():
    recs, _ = pipeline.process(FILES, use_llm=False)
    r2 = copy.deepcopy(recs); r2[0]["patient"]["name"]["value"] = "Someone Else"
    assert X.build_case(r2)["warnings"]


def test_med_reconciliation_status():
    _, case = pipeline.process(FILES, use_llm=False)
    st = {r["Medication"]: r["Status"] for r in case["med_reconciliation"]}
    assert st["Apixaban"] == "CHANGED" and st["Ibuprofen"] == "NEW" and st["Furosemide"] == "unchanged"


def test_single_document_does_not_crash():
    recs, case = pipeline.process(FILES[:1], use_llm=False)
    assert case["triage"]["level"] in {"URGENT", "PRIORITY", "SOON", "ROUTINE"}


# ---- provider-agnostic LLM path (mocked OpenAI-compatible client; no network) ----
import json as _json
from types import SimpleNamespace as NS
from hub import llm_extract as L

LAB_TXT = (S / "lab_report_2026-09-13.txt").read_text()


class FakeOpenAI:
    def __init__(self, content, reject_json_mode=False):
        self.content, self.reject = content, reject_json_mode
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.reject and "response_format" in kw:
            raise RuntimeError("response_format unsupported")
        return NS(choices=[NS(message=NS(content=self.content))])


def _gold_json(drop_lab=None):
    g = R.extract(LAB_TXT); g.pop("extractor")
    if drop_lab: g["labs"] = [l for l in g["labs"] if l["test"] != drop_lab]
    return _json.dumps(g)


def test_openai_style_clean_json():
    out = L.extract(LAB_TXT, client=FakeOpenAI(_gold_json()))
    assert out["extractor"].startswith("llm") and len(out["labs"]) == 11


def test_fenced_json_and_json_mode_rejected_retry():
    c = FakeOpenAI("Here you go:\n```json\n" + _gold_json() + "\n```", reject_json_mode=True)
    out = L.extract(LAB_TXT, client=c)
    assert out["extractor"].startswith("llm") and len(c.calls) == 2


def test_missed_lab_is_filled_from_rules_at_lower_confidence():
    out = L.extract(LAB_TXT, client=FakeOpenAI(_gold_json(drop_lab="Potassium")))
    k = next(l for l in out["labs"] if l["test"] == "Potassium")
    assert k["value"] == 5.6 and k["confidence"] < 0.96


def test_garbage_llm_output_falls_back_to_rules():
    out = L.extract(LAB_TXT, client=FakeOpenAI("I cannot help with that."))
    assert out["extractor"].startswith("rules (LLM failed") and len(out["labs"]) == 11


def test_provider_selection(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
              "ANTHROPIC_API_KEY", "CDIH_PROVIDER", "CDIH_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert L.active() is None and not L.available()
    monkeypatch.setenv("GROQ_API_KEY", "x"); assert L.active()[0] == "groq"
    monkeypatch.setenv("GEMINI_API_KEY", "y"); assert L.active()[0] == "gemini"   # free-first order
    monkeypatch.setenv("CDIH_PROVIDER", "groq"); assert L.active()[0] == "groq"   # explicit override
    monkeypatch.setenv("CDIH_MODEL", "m1"); assert L.active()[1] == "m1"
