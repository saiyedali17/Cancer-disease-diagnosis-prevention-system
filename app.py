from flask import Flask, render_template, request, send_file, session
import numpy as np
import io
from datetime import datetime
from model_pipeline import get_trained_pipeline

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

app = Flask(__name__)
app.secret_key = 'cancer_prediction_secret_key'

pipeline = get_trained_pipeline()

FIELD_LABELS = {
    "GENDER": "Gender",
    "AGE": "Age",
    "SMOKING": "Smoking",
    "ALCOHOL CONSUMING": "Alcohol Consuming",
    "PEER_PRESSURE": "Peer Pressure",
    "YELLOW_FINGERS": "Yellow Fingers",
    "CHRONIC DISEASE": "Chronic Disease",
    "ANXIETY": "Anxiety",
    "FATIGUE ": "Fatigue",
    "ALLERGY ": "Allergy",
    "WHEEZING": "Wheezing",
    "COUGHING": "Coughing",
    "SHORTNESS OF BREATH": "Shortness of Breath",
    "SWALLOWING DIFFICULTY": "Swallowing Difficulty",
    "CHEST PAIN": "Chest Pain",
}


def get_display_value(key, value):
    if key == "GENDER":
        return "Male" if value == "1" else "Female"
    if key == "AGE":
        return value
    return "Yes" if value == "1" else "No"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/hospitals')
def hospitals():
    return render_template('hospitals.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

@app.route('/awareness')
def awareness():
    return render_template('awareness.html')

@app.route('/stages')
def stages():
    return render_template('stages.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = dict(request.form)
        features = [float(v) for v in request.form.values()]
        final_features = np.array([features])

        prediction = pipeline.predict(final_features)[0]

        if prediction == 1:
            result = "High Risk of Lung Cancer"
            high_risk = True
        else:
            result = "Low Risk of Lung Cancer"
            high_risk = False

        session['prediction_result'] = result
        session['high_risk'] = bool(high_risk)
        session['form_data'] = {k: (v[0] if isinstance(v, list) else v) for k, v in form_data.items()}
        session['prediction_time'] = datetime.now().strftime("%d %B %Y, %I:%M %p")

        return render_template('index.html', prediction_text=result, high_risk=high_risk)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


@app.route('/download_report')
def download_report():
    result          = session.get('prediction_result', 'N/A')
    high_risk       = session.get('high_risk', False)
    form_data       = session.get('form_data', {})
    prediction_time = session.get('prediction_time', datetime.now().strftime("%d %B %Y, %I:%M %p"))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=20*mm,  bottomMargin=20*mm)

    PRIMARY    = colors.HexColor('#0f2027')
    ACCENT     = colors.HexColor('#1c92d2')
    DANGER     = colors.HexColor('#ff4b2b')
    SUCCESS    = colors.HexColor('#00c853')
    LIGHT_GREY = colors.HexColor('#f0f4f8')
    WHITE      = colors.white
    risk_color = DANGER if high_risk else SUCCESS
    risk_label = "HIGH RISK" if high_risk else "LOW RISK"

    styles = getSampleStyleSheet()

    def ps(name, **kw):
        return ParagraphStyle(name, parent=styles['Normal'], **kw)

    story = []

    # ── HEADER ──
    hdr = Table([[Paragraph("AI Lung Cancer Prediction System",
        ps('T', fontSize=22, textColor=WHITE, fontName='Helvetica-Bold', alignment=TA_CENTER))]],
        colWidths=[170*mm])
    hdr.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), PRIMARY),
        ('TOPPADDING',(0,0),(-1,-1),14),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
    ]))
    story.append(hdr)

    sub = Table([[Paragraph("Clinical Risk Analysis Report",
        ps('S', fontSize=10, textColor=colors.HexColor('#cce8ff'), fontName='Helvetica', alignment=TA_CENTER))]],
        colWidths=[170*mm])
    sub.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),ACCENT),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
    ]))
    story.append(sub)
    story.append(Spacer(1,10))

    # ── META ──
    meta = Table([
        ["Report Generated:", prediction_time],
        ["Report Type:", "Lung Cancer Risk Prediction"],
        ["Prediction Model:", "Random Forest Classifier"],
    ], colWidths=[50*mm,120*mm])
    meta.setStyle(TableStyle([
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),('FONTNAME',(1,0),(1,-1),'Helvetica'),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('TEXTCOLOR',(0,0),(0,-1),PRIMARY),('TEXTCOLOR',(1,0),(1,-1),colors.HexColor('#444444')),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),
    ]))
    story.append(meta)
    story.append(Spacer(1,8))
    story.append(HRFlowable(width="100%",thickness=1,color=ACCENT,spaceAfter=10))

    # ── RESULT BANNER ──
    res_tbl = Table([
        [Paragraph(f"PREDICTION RESULT: {risk_label}",
            ps('RL', fontSize=16, textColor=WHITE, fontName='Helvetica-Bold', alignment=TA_CENTER))],
        [Paragraph(result,
            ps('RS', fontSize=11, textColor=WHITE, fontName='Helvetica', alignment=TA_CENTER))],
    ], colWidths=[170*mm])
    res_tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),risk_color),
        ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),
    ]))
    story.append(res_tbl)
    story.append(Spacer(1,14))

    # ── HELPER: build a 2-col table ──
    def build_table(rows, hdr_color=PRIMARY):
        t = Table(rows, colWidths=[85*mm,85*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),hdr_color),
            ('TEXTCOLOR',(0,0),(-1,0),WHITE),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
            ('FONTSIZE',(0,0),(-1,-1),9),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LIGHT_GREY]),
            ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#dddddd')),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('LEFTPADDING',(0,0),(-1,-1),8),
        ]))
        return t

    sec  = ps('SEC', fontSize=12, textColor=ACCENT, fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=6)
    sh   = ps('SH',  fontSize=10, textColor=PRIMARY, fontName='Helvetica-Bold', spaceAfter=4)

    story.append(Paragraph("Patient Input Details", sec))
    story.append(HRFlowable(width="100%",thickness=0.5,color=colors.HexColor('#cccccc'),spaceAfter=6))

    # Personal
    story.append(Paragraph("Personal Information", sh))
    rows = [["Parameter","Value"]]
    for k in ["GENDER","AGE"]:
        if k in form_data:
            rows.append([FIELD_LABELS.get(k,k), get_display_value(k, form_data[k])])
    story.append(build_table(rows))
    story.append(Spacer(1,10))

    # Lifestyle
    story.append(Paragraph("Lifestyle Factors", sh))
    rows = [["Parameter","Value"]]
    for k in ["SMOKING","ALCOHOL CONSUMING","PEER_PRESSURE","YELLOW_FINGERS","CHRONIC DISEASE"]:
        if k in form_data:
            rows.append([FIELD_LABELS.get(k,k), get_display_value(k, form_data[k])])
    story.append(build_table(rows))
    story.append(Spacer(1,10))

    # Symptoms
    story.append(Paragraph("Reported Symptoms", sh))
    rows = [["Symptom","Present"]]
    for k in ["ANXIETY","FATIGUE ","ALLERGY ","WHEEZING","COUGHING",
              "SHORTNESS OF BREATH","SWALLOWING DIFFICULTY","CHEST PAIN"]:
        if k in form_data:
            rows.append([FIELD_LABELS.get(k,k), get_display_value(k, form_data[k])])
    story.append(build_table(rows))
    story.append(Spacer(1,14))

    # ── RECOMMENDATIONS ──
    story.append(HRFlowable(width="100%",thickness=0.5,color=colors.HexColor('#cccccc'),spaceAfter=8))
    story.append(Paragraph("Recommendations", sec))

    body = ps('B', fontSize=9, textColor=PRIMARY, fontName='Helvetica', leading=14)

    if high_risk:
        recs = [
            "Consult an oncologist or pulmonologist immediately for a thorough clinical evaluation.",
            "Undergo diagnostic imaging such as a CT scan or chest X-ray as advised by your doctor.",
            "Avoid smoking and exposure to second-hand smoke or industrial pollutants.",
            "Maintain a healthy lifestyle including a balanced diet and regular moderate exercise.",
            "Visit one of the recommended hospitals listed below for specialist care.",
        ]
        hospitals_list = [
            ("SRMS IMS, Bareilly UP",               "www.srmsims.ac.in"),
            ("Tata Memorial Hospital, Mumbai",       "tmc.gov.in"),
            ("AIIMS, New Delhi",                     "www.aiims.edu"),
            ("Rajiv Gandhi Cancer Institute, Delhi", "www.rgcirc.org"),
            ("Apollo Cancer Centres",                "www.apollohospitals.com"),
        ]
    else:
        recs = [
            "Your current indicators suggest low risk. Continue maintaining a healthy lifestyle.",
            "Avoid smoking and limit alcohol consumption to reduce long-term cancer risk.",
            "Schedule regular health check-ups (at least once a year) with your physician.",
            "Stay physically active and maintain a balanced, nutrient-rich diet.",
            "If any new symptoms develop, consult a doctor promptly without delay.",
        ]
        hospitals_list = None

    for i, rec in enumerate(recs, 1):
        story.append(Paragraph(f"{i}. {rec}", body))
        story.append(Spacer(1,4))

    if hospitals_list:
        story.append(Spacer(1,8))
        story.append(Paragraph("Recommended Hospitals for Consultation", sh))
        rows = [["Hospital","Website"]] + list(hospitals_list)
        t = Table(rows, colWidths=[100*mm,70*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),DANGER),
            ('TEXTCOLOR',(0,0),(-1,0),WHITE),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
            ('FONTSIZE',(0,0),(-1,-1),9),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LIGHT_GREY]),
            ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#dddddd')),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('LEFTPADDING',(0,0),(-1,-1),8),
        ]))
        story.append(t)

    # ── DISCLAIMER ──
    story.append(Spacer(1,16))
    story.append(HRFlowable(width="100%",thickness=0.5,color=colors.HexColor('#cccccc'),spaceAfter=8))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI-based prediction system for informational purposes only. "
        "It is not a substitute for professional medical diagnosis or advice. "
        "Please consult a qualified healthcare professional for any medical concerns.",
        ps('D', fontSize=8, textColor=colors.HexColor('#888888'),
           alignment=TA_CENTER, fontName='Helvetica-Oblique', leading=12)
    ))

    doc.build(story)
    buffer.seek(0)
    filename = f"Cancer_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')


if __name__ == "__main__":
    app.run(debug=True)
