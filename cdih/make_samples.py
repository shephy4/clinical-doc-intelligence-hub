"""Generate fully synthetic (fictional) clinical documents for the demo."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont

OUT = Path("samples")
OUT.mkdir(exist_ok=True)

DOCS = {
"discharge_summary_2026-09-14": """DISCHARGE SUMMARY
Hospital: Northbridge General Hospital
Patient Name: Margaret Ellis
DOB: 12/03/1951 (Age 75)
NHS Number: 943 476 5919 (synthetic)
Sex: Female
Date of Admission: 09/09/2026
Date of Discharge: 14/09/2026
Consultant: Dr A. Rahman

Primary Diagnosis: Acute decompensated heart failure
Secondary Diagnoses: Atrial fibrillation; Type 2 diabetes mellitus; Chronic kidney disease stage 3b; Hypertension

Allergies: Penicillin (rash); Sulfonamides

Presenting Complaint:
72-hour history of worsening breathlessness, orthopnoea and bilateral leg swelling. Weight up 4 kg in one week.

Hospital Course:
Treated with IV furosemide with good diuresis (net -5.2 L). Echocardiogram showed LVEF 35%. Rate controlled with bisoprolol. Renal function monitored; creatinine peaked at 168 umol/L and improved to 142 umol/L. Mobilised with physiotherapy.

Vital Signs at Discharge:
BP 118/72 mmHg, HR 78 bpm, SpO2 95% on air, Temp 36.6 C, Weight 71.8 kg

Discharge Medications:
- Furosemide 40 mg oral once daily
- Bisoprolol 2.5 mg oral once daily
- Apixaban 2.5 mg oral twice daily
- Metformin 500 mg oral twice daily
- Ramipril 2.5 mg oral once daily
- Atorvastatin 40 mg oral once daily

Follow-up:
Heart failure nurse review within 7 days. Repeat U&E in 1 week. Cardiology clinic in 6 weeks.
Advice: daily weights, 1.5 L fluid restriction, low-salt diet. Return if weight rises >2 kg in 3 days.
""",
"lab_report_2026-09-13": """PATHOLOGY LABORATORY REPORT
Laboratory: Northbridge General Hospital Pathology
Patient Name: Margaret Ellis
DOB: 12/03/1951
NHS Number: 943 476 5919 (synthetic)
Collected: 13/09/2026 07:40
Reported: 13/09/2026 11:15

Test | Result | Unit | Reference Range
Haemoglobin | 104 | g/L | 115 - 165
White Cell Count | 8.2 | x10^9/L | 4.0 - 11.0
Platelets | 231 | x10^9/L | 150 - 400
Sodium | 133 | mmol/L | 133 - 146
Potassium | 5.6 | mmol/L | 3.5 - 5.3
Urea | 14.8 | mmol/L | 2.5 - 7.8
Creatinine | 152 | umol/L | 45 - 84
eGFR | 31 | mL/min/1.73m2 | > 60
HbA1c | 61 | mmol/mol | 20 - 42
NT-proBNP | 4850 | ng/L | < 125
CRP | 9 | mg/L | 0 - 5

Comment: Hyperkalaemia and reduced eGFR. Clinical correlation advised.
""",
"physician_note_2026-09-11": """PHYSICIAN PROGRESS NOTE
Patient Name: Margaret Ellis
DOB: 12/03/1951
Date: 11/09/2026
Author: Dr A. Rahman, Consultant Cardiologist

Subjective: Breathlessness improving. Sleeping flat for the first time in a week. Reports poor appetite. Lives alone; daughter visits twice weekly.
Objective: BP 122/70, HR 84 irregular, SpO2 94% on air. Bibasal crackles reduced. Peripheral oedema 1+.
Assessment: Heart failure with reduced EF responding to IV diuresis. AF rate controlled. AKI on CKD 3b, creatinine trending down.
Plan:
1. Convert IV furosemide to oral furosemide 40 mg daily tomorrow.
2. Continue ramipril but recheck U&E in 48 hours given raised potassium.
3. Review metformin dose given eGFR - discuss with pharmacist.
4. Social work assessment for home support. Falls risk: uses walking stick.
5. Anticoagulation: apixaban continued, dose to be confirmed against age, weight and creatinine.
""",
"intake_form_2026-10-02": """OUTPATIENT INTAKE FORM
Clinic: Northbridge Heart Failure Clinic
Patient Name: Margaret Ellis
DOB: 12/03/1951
Date: 02/10/2026
Reason for Visit: Follow-up after hospital admission. Ankle swelling returning, tired, dizzy on standing.

Allergies: Penicillin
Current Medications: Furosemide 40mg daily, Bisoprolol 2.5mg daily, Apixaban 5 mg twice daily, Metformin 500 mg twice daily, Ramipril 2.5mg daily, Atorvastatin 40mg daily, Ibuprofen 400 mg as needed for knee pain
Smoker: No
Alcohol: 2 units per week
Falls in last 3 months: 1 (tripped at home)
Weight today: 74.3 kg
BP today: 104/62 mmHg, HR 82 bpm
Living situation: Lives alone
""",
}

def write_pdf(name, text):
    c = canvas.Canvas(str(OUT / f"{name}.pdf"), pagesize=A4)
    w, h = A4
    y = h - 50
    c.setFont("Helvetica", 9.5)
    for line in text.splitlines():
        if y < 50:
            c.showPage(); c.setFont("Helvetica", 9.5); y = h - 50
        if line and line.isupper() and len(line) < 60:
            c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, line[:120])
        c.setFont("Helvetica", 9.5)
        y -= 13
    c.save()

for name, text in DOCS.items():
    (OUT / f"{name}.txt").write_text(text)
    if not name.startswith("intake"):
        write_pdf(name, text)

# Image version of the intake form (tests OCR path)
text = DOCS["intake_form_2026-10-02"]
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
except Exception:
    font = ImageFont.load_default()
import textwrap
lines = [w for l in text.splitlines() for w in (textwrap.wrap(l, 85, subsequent_indent="    ") or [""])]
img = Image.new("RGB", (1500, 60 + 34 * len(lines)), "white")
d = ImageDraw.Draw(img)
for i, l in enumerate(lines):
    d.text((40, 30 + 34 * i), l, fill="black", font=font)
img.save(OUT / "intake_form_2026-10-02.png")
print("Generated:", sorted(p.name for p in OUT.iterdir()))
