# Clinical Document Intelligence Hub (PoC)

Turns unstructured clinical documents (PDF, scanned image, text) into a structured, auditable
**patient card**: extracted fields with confidence, cross-document medication reconciliation,
risk flags with rationale, a recommended next step list and a triage suggestion.
*Synthetic data only. Decision support for clinician review, not a diagnosis.*

## Run it
```bash
pip install -r requirements.txt && sudo apt install tesseract-ocr   # OCR for images/scans
streamlit run app.py                    # offline rule-based engine, no key needed
# optional: enable LLM extraction + narrative with ANY ONE of these (free tiers work):
export GEMINI_API_KEY=...               # Google AI Studio free tier
export GROQ_API_KEY=...                 # Groq free tier
export OPENROUTER_API_KEY=...           # OpenRouter ":free" models
export ANTHROPIC_API_KEY=...            # Claude also works
# optional: CDIH_PROVIDER=gemini|groq|openrouter|anthropic   CDIH_MODEL=<model id>
python make_samples.py                  # regenerate the synthetic documents
python -m pytest -q tests               # 12 tests
```
In the app, click **Load sample patient** (4 documents: discharge summary, lab report, physician
note as PDFs, and an intake form as a scanned image) or upload your own.

## Approach
`ingest` (PDF text layer, OCR fallback) → `extract` (Claude or rules, same schema) →
`reasoning` (deterministic rules) → `app` (patient card).
- **Extraction**: provider-agnostic. Gemini / Groq / OpenRouter use OpenAI-compatible JSON mode; Claude uses
  forced tool-use. Every field has a `confidence` and a verbatim `evidence` snippet; temperature 0.
  Output is validated and cross-checked against the rule extractor (labs/meds the model missed are added at
  reduced confidence). On no key, quota/rate-limit errors or unparseable output it falls back to the
  offline rules, so the demo never breaks. Free-tier models are weaker, which is why the cross-check exists.
- **Reasoning is code, not prompt**: lab flags vs reference ranges; ACE inhibitor + raised K+; metformin vs eGFR;
  apixaban dose-reduction criteria; NSAID with HF/CKD/anticoagulant; weight/BP trend; falls on anticoagulant;
  follow-ups requested but not evidenced. Each flag shows severity, rationale, action and source documents.
- **Optional enhancements built**: per-field confidence (fields < 0.80 are called out), multi-document
  comparison (medication reconciliation + identity-mismatch warning), triage routing
  (URGENT / PRIORITY / SOON / ROUTINE).

## Models / tools
Python, Streamlit, pdfplumber, Tesseract (pytesseract), `openai` SDK (OpenAI-compatible endpoints) and `anthropic` SDK. Default models: gemini-2.5-flash, llama-3.3-70b-versatile (Groq), a ":free" Llama model (OpenRouter); override with `CDIH_MODEL` if a name has changed.

## Assumptions and limits
- All documents are fictional; UK date format (DD/MM/YYYY). Thresholds are illustrative decision-support
  rules and would need clinical sign-off. Not a medical device.
- Extraction accuracy has only been checked on the synthetic set. The LLM paths are unit-tested with mocked
  clients (clean JSON, fenced JSON, missed fields, garbage output, provider selection); run once with your
  real key before recording, since free-tier model names and limits change.
- "Not evidenced in supplied documents" means exactly that, not that something did not happen.

## Example
**Input** (lab report excerpt):
`Potassium | 5.6 | mmol/L | 3.5 - 5.3` · intake form lists `Ramipril 2.5mg daily`, `Ibuprofen 400 mg as needed`

**Output** (excerpt, full file in `outputs/sample_case.md`):
```
Triage: PRIORITY - Clinician review within 24-48 hours
HIGH  Raised potassium (K+ 5.6 mmol/L, ref <= 5.3)   [Lab report]
HIGH  ACE inhibitor with raised potassium            [Lab report, Physician note]
HIGH  Apixaban 2.5 mg BD in discharge summary: possible UNDER-dose   (criteria met 1/3 -> expected 5 mg)
HIGH  NSAID on medication list (Ibuprofen)           [Intake form]
Reconciliation: Apixaban 2.5 mg BD -> 5 mg BD (CHANGED);  Ibuprofen NEW
```
