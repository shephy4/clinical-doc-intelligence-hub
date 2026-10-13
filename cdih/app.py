"""Clinical Document Intelligence Hub - Streamlit prototype."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from hub import llm_extract, pipeline

st.set_page_config(page_title="Clinical Document Intelligence Hub", page_icon="🩺", layout="wide")

SEV_STYLE = {"CRITICAL": ("#b00020", "#fde7ea"), "HIGH": ("#b45309", "#fef3c7"),
             "MODERATE": ("#1d4ed8", "#e0ecff"), "INFO": ("#475569", "#f1f5f9")}
TRIAGE_STYLE = {"URGENT": "#b00020", "PRIORITY": "#c2410c", "SOON": "#a16207", "ROUTINE": "#15803d"}
SAMPLES = Path(__file__).parent / "samples"

st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1250px}
.badge{display:inline-block;padding:2px 9px;border-radius:10px;font-size:.72rem;font-weight:700;letter-spacing:.04em}
.flag{border-left:5px solid;border-radius:6px;padding:.55rem .8rem;margin:.35rem 0}
.flag b{font-size:.95rem}.flag small{display:block;margin-top:2px;opacity:.85}
.triage{border-radius:10px;padding:.9rem 1.1rem;color:white;margin-bottom:.8rem}
.triage h3{margin:0;color:white}.triage span{opacity:.92}
.card{border:1px solid #d9dee5;border-radius:10px;padding:.9rem 1.1rem;background:var(--secondary-background-color)}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Clinical Document Intelligence Hub")
    st.caption("Synthetic data only. Decision support for clinician review - not a diagnosis.")
    llm_ok = llm_extract.available()
    LLM_LABEL = f"LLM ({llm_extract.label()})"
    mode = st.radio("Extraction engine",
                    [LLM_LABEL, "Rule-based (offline)"] if llm_ok else ["Rule-based (offline)"],
                    help="LLM mode needs a free API key (GEMINI_API_KEY, GROQ_API_KEY or OPENROUTER_API_KEY; "
                         "ANTHROPIC_API_KEY also works). Rule-based mode runs without any key.")
    if not llm_ok:
        st.info("No LLM API key found - running the offline rule-based engine. Set GEMINI_API_KEY, "
                "GROQ_API_KEY or OPENROUTER_API_KEY to enable LLM extraction.")
    st.divider()
    uploads = st.file_uploader("Upload documents (PDF / image / text)", accept_multiple_files=True,
                               type=["pdf", "png", "jpg", "jpeg", "txt"])
    if st.button("Load sample patient (4 documents)", use_container_width=True):
        st.session_state["files"] = [(p.name, p.read_bytes()) for p in sorted(SAMPLES.iterdir())
                                     if p.suffix in (".pdf", ".png")]
    if uploads:
        st.session_state["files"] = [(u.name, u.getvalue()) for u in uploads]
    st.caption("Sample set: discharge summary (PDF), lab report (PDF), physician note (PDF), "
               "intake form (scanned image -> OCR).")

files = st.session_state.get("files")
st.title("🩺 Clinical Document Intelligence Hub")
st.caption("Unstructured clinical documents → structured, auditable patient intelligence")

if not files:
    st.markdown("#### Start here")
    st.write("Upload one or more clinical documents, or click **Load sample patient** in the sidebar. "
             "The hub will read each document, extract structured fields with a confidence score, "
             "cross-check them across documents, and produce a patient card with risk flags and a triage suggestion.")
    st.stop()


@st.cache_data(show_spinner=False)
def run(files, use_llm):
    return pipeline.process(list(files), use_llm=use_llm)


with st.spinner("Reading and analysing documents..."):
    records, case = run(tuple(files), mode.startswith("LLM"))

p = case["patient"]
t = case["triage"]
# ---------------------------------------------------------------- tabs
tab_card, tab_flags, tab_meds, tab_docs, tab_json = st.tabs(
    ["Patient card", "Risk flags", "Medication reconciliation", "Extracted data & confidence", "Structured JSON"])

with tab_card:
    st.markdown(f"<div class='triage' style='background:{TRIAGE_STYLE[t['level']]}'>"
                f"<h3>Suggested triage: {t['level']}</h3><span>{t['timeframe']} - {t['why']}</span></div>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient", p.get("name", "?"))
    c2.metric("Age / Sex", f"{case['age']} / {p.get('sex', '?')}")
    c3.metric("DOB", p.get("dob", "?"))
    c4.metric("NHS no.", p.get("nhs_number", "?"))
    for w in case["warnings"]:
        st.error(w)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Summary")
        st.markdown(f"<div class='card'>{case['narrative']}</div>", unsafe_allow_html=True)
        st.caption(f"Engine: {case['extractor']}")
        st.subheader("Recommended next steps")
        for s in case["next_steps"]:
            st.markdown(f"- {s}")
    with right:
        st.subheader("Diagnoses")
        for i, d in enumerate(case["diagnoses"]):
            st.markdown(f"{'**' + d + '**' if i == 0 else d}")
        st.subheader("Allergies")
        st.markdown(", ".join(f"`{a}`" for a in case["allergies"]) or "None recorded")
        st.subheader("Document timeline")
        st.dataframe(pd.DataFrame(case["timeline"]), hide_index=True, use_container_width=True)

with tab_flags:
    counts = pd.Series([f["severity"] for f in case["flags"]]).value_counts()
    st.write("  ".join(f"**{k}**: {v}" for k, v in counts.items()))
    only = st.multiselect("Filter severity", list(SEV_STYLE), default=["CRITICAL", "HIGH", "MODERATE"])
    for f in case["flags"]:
        if f["severity"] not in only:
            continue
        fg, bg = SEV_STYLE[f["severity"]]
        st.markdown(f"<div class='flag' style='border-color:{fg};background:{bg};color:#111'>"
                    f"<span class='badge' style='background:{fg};color:white'>{f['severity']}</span> "
                    f"<b>{f['title']}</b><small>{f['rationale']}</small>"
                    f"<small>Source: {', '.join(f['sources'])}</small></div>", unsafe_allow_html=True)

with tab_meds:
    st.caption("Same drug compared across documents (oldest → newest). Changes are highlighted.")
    df = pd.DataFrame(case["med_reconciliation"])
    colour = {"CHANGED": "background-color:#fde7ea", "NEW": "background-color:#fef3c7",
              "STOPPED": "background-color:#fde7ea"}
    st.dataframe(df.style.map(lambda x: colour.get(x, ""), subset=["Status"]), hide_index=True,
                 use_container_width=True)

with tab_docs:
    for r in records:
        mean, low = pipeline.confidence_summary(r)
        with st.expander(f"{r['doc_type']['value']} - {r['_file']}  |  mean confidence {mean:.0%}  |  "
                         f"read via {r['_ingest']}", expanded=False):
            a, b = st.columns(2)
            rows = []
            for grp in ("patient", "dates", "vitals"):
                for k, f in r[grp].items():
                    rows.append({"field": f"{grp}.{k}", "value": str(f["value"]),
                                 "confidence": f["confidence"], "evidence": f["evidence"]})
            for k in ("primary_diagnosis", "reason_for_visit", "author"):
                if r.get(k):
                    rows.append({"field": k, "value": str(r[k]["value"]),
                                 "confidence": r[k]["confidence"], "evidence": r[k]["evidence"]})
            for k in ("secondary_diagnoses", "allergies"):
                for f in r[k]:
                    rows.append({"field": k, "value": str(f["value"]), "confidence": f["confidence"], "evidence": f["evidence"]})
            if rows:
                a.markdown("**Fields**")
                a.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                            column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
            if r["medications"]:
                b.markdown("**Medications**")
                b.dataframe(pd.DataFrame(r["medications"])[["name", "dose", "route", "frequency", "confidence"]],
                            hide_index=True, use_container_width=True)
            if r["labs"]:
                b.markdown("**Lab results**")
                ldf = pd.DataFrame(r["labs"])[["test", "value", "unit", "ref_low", "ref_high", "flag"]]
                b.dataframe(ldf.style.map(lambda x: "background-color:#fde7ea" if x in ("HIGH", "LOW") else "", subset=["flag"]),
                            hide_index=True, use_container_width=True)
            if r["plan_items"]:
                b.markdown("**Plan**"); [b.markdown(f"- {x}") for x in r["plan_items"]]
            if r["follow_up"]:
                b.markdown("**Follow-up**"); [b.markdown(f"- {x}") for x in r["follow_up"]]
            if low:
                st.warning("Low-confidence fields (<0.80) - verify manually: " +
                           "; ".join(f"{l[0]}={l[1]} ({l[2]:.2f})" for l in low))
            st.text_area("Source text", r["_text"], height=160, key=f"t{r['_file']}")

with tab_json:
    export = {"case": case, "documents": [{k: v for k, v in r.items() if k != "_text"} for r in records]}
    s = json.dumps(export, indent=2, default=str)
    st.download_button("Download structured JSON", s, "patient_case.json", "application/json")
    st.code(s[:6000] + ("\n..." if len(s) > 6000 else ""), language="json")
