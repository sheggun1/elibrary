import os
import io
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import google.generativeai as genai
from weasyprint import HTML
from docx import Document
from bs4 import BeautifulSoup
import uvicorn

# ---------- Config ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI(title="Lesson Note AI Backend")

# ---------- Request model ----------
class LessonNoteRequest(BaseModel):
    class_level: str
    subject: str
    term: str
    week_number: int
    topic: str
    performance_objectives: List[str]
    content_knowledge: List[str]
    key_competencies: List[str]
    learning_activities: List[str]
    teaching_resources: List[str]
    evaluation_guide: List[str]

# ---------- Prompt ----------
PROMPT_TEMPLATE = """
You are an expert Nigerian JSS teacher. Generate a complete NERDC-compliant LESSON NOTE.

=== CURRICULUM DATA (USE ONLY THIS) ===
Class: {class_level}
Subject: {subject}
Term: {term}
Week: {week_number}
Topic: {topic}
Performance Objectives: {objectives}
Content Knowledge: {content}
Key Competencies: {competencies}
Learning Activities: {activities}
Teaching Resources: {resources}
Evaluation Guide: {evaluation}
=========================================

Output a complete lesson note in HTML with these sections:
1. GENERAL INFORMATION (Class, Subject, Topic, Duration 40min, Week)
2. PERFORMANCE OBJECTIVES ("By the end of the lesson, students should be able to...")
3. INSTRUCTIONAL MATERIALS
4. PREVIOUS KNOWLEDGE (1-2 sentences, Nigerian context)
5. INTRODUCTION (5 min, 3-4 sentence hook)
6. LESSON DEVELOPMENT (25 min, HTML table: Step | Teacher Activity | Student Activity | Time, at least 3 steps)
7. ENRICHMENT SECTION (3 Nigerian examples, 1 group activity, 1 differentiation tip)
8. EVALUATION (5 min, 3-5 verbal questions from Evaluation Guide)
9. SUMMARY & CONCLUSION (2-3 sentences)
10. ASSIGNMENT (1 practical homework)

Use simple English for JSS 1. Do not invent topics outside the data.
Return ONLY the HTML body content (no <html> or <body> tags, just inner HTML).
"""

# ---------- Generate HTML from Gemini ----------
def generate_lesson_html(req: LessonNoteRequest) -> str:
    prompt = PROMPT_TEMPLATE.format(
        class_level=req.class_level,
        subject=req.subject,
        term=req.term,
        week_number=req.week_number,
        topic=req.topic,
        objectives=", ".join(req.performance_objectives),
        content=", ".join(req.content_knowledge),
        competencies=", ".join(req.key_competencies),
        activities=", ".join(req.learning_activities),
        resources=", ".join(req.teaching_resources),
        evaluation=", ".join(req.evaluation_guide)
    )
    try:
        response = model.generate_content(prompt)
        html_content = response.text.strip()
        if html_content.startswith("```html"):
            html_content = html_content[7:]
        if html_content.endswith("```"):
            html_content = html_content[:-3]
        return html_content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

# ---------- HTML -> DOCX ----------
def html_to_docx(html_str: str, title: str) -> bytes:
    doc = Document()
    doc.add_heading(title, level=0)

    soup = BeautifulSoup(html_str, "html.parser")

    for element in soup.find_all(["h1", "h2", "h3", "p", "li", "table"]):
        if element.name == "h1":
            doc.add_heading(element.get_text(strip=True), level=1)
        elif element.name == "h2":
            doc.add_heading(element.get_text(strip=True), level=2)
        elif element.name == "h3":
            doc.add_heading(element.get_text(strip=True), level=3)
        elif element.name == "p":
            doc.add_paragraph(element.get_text(strip=True))
        elif element.name == "li":
            doc.add_paragraph(element.get_text(strip=True), style="List Bullet")
        elif element.name == "table":
            rows = element.find_all("tr")
            if rows:
                cols = len(rows[0].find_all(["th", "td"]))
                table = doc.add_table(rows=0, cols=cols)
                table.style = "Table Grid"
                for row in rows:
                    cells = row.find_all(["th", "td"])
                    row_cells = table.add_row().cells
                    for i, cell in enumerate(cells):
                        row_cells[i].text = cell.get_text(strip=True)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()

# ---------- HTML -> PDF (WeasyPrint) ----------
def html_to_pdf(html_str: str, title: str) -> bytes:
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"><title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20mm; line-height: 1.6; }}
        h1, h2, h3 {{ color: #1a5276; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #333; padding: 8px; text-align: left; }}
        th {{ background: #eaf2f8; }}
    </style>
    </head>
    <body>{html_str}</body>
    </html>
    """
    try:
        pdf_bytes = HTML(string=full_html).write_pdf()
        return pdf_bytes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"status": "alive", "service": "Lesson Note AI"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "gemini-1.5-flash"}

@app.post("/generate-lesson-note")
def generate_lesson_note(req: LessonNoteRequest):
    html = generate_lesson_html(req)
    return {"html": html, "topic": req.topic, "week": req.week_number}

@app.post("/generate-lesson-note/pdf")
def generate_lesson_pdf(req: LessonNoteRequest):
    html = generate_lesson_html(req)
    pdf_bytes = html_to_pdf(html, f"Lesson Note - {req.topic}")
    filename = f"Lesson_Note_{req.subject}_Week{req.week_number}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@app.post("/generate-lesson-note/docx")
def generate_lesson_docx(req: LessonNoteRequest):
    html = generate_lesson_html(req)
    docx_bytes = html_to_docx(html, f"Lesson Note - {req.topic}")
    filename = f"Lesson_Note_{req.subject}_Week{req.week_number}.docx"
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)