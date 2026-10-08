import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_retention_pdf(customer_id, customer_data, churn_prob, risk_level, retention_strategy):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=12
    )
    subtitle_style = ParagraphStyle(
        'SubTitleStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#2563EB"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )

    # Document Header
    story.append(Paragraph("⚡ ChurnAI - Customer Retention Executive Report", title_style))
    story.append(Spacer(1, 10))

    # Summary Table
    table_data = [
        ["Customer ID:", str(customer_id)],
        ["Churn Risk Score:", f"{churn_prob * 100:.1f}%"],
        ["Risk Category:", risk_level]
    ]
    t = Table(table_data, colWidths=[150, 350])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F3F4F6")),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor("#1F2937")),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB"))
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    # Profile Section
    story.append(Paragraph("Customer Details Summary", subtitle_style))
    profile_text = "<br/>".join([f"<b>{k}:</b> {v}" for k, v in list(customer_data.items())[:6]])
    story.append(Paragraph(profile_text, body_style))
    story.append(Spacer(1, 15))

    # GenAI Strategy Section
    story.append(Paragraph("AI-Generated Retention Action Plan (Gemini)", subtitle_style))
    clean_strategy = retention_strategy.replace("\n", "<br/>")
    story.append(Paragraph(clean_strategy, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer