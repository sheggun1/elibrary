import os
import io
import time
import random
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from google import genai
from weasyprint import HTML
from docx import Document
from bs4 import BeautifulSoup
import uvicorn

# ---------- Config ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.8-flash"

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

1. GENERAL INFORMATION
   Class: {class_level}
   Subject: {subject}
   Term: {term}
   Week: {week_number}
   Topic: {topic}
   Duration: 40 minutes

2. PERFORMANCE OBJECTIVES
   "By the end of the lesson, students should be able to:" followed by a numbered list.

3. INSTRUCTIONAL MATERIALS
   A bulleted list.

4. PREVIOUS KNOWLEDGE
   1-2 sentences in Nigerian context.

5. INTRODUCTION (5 minutes)
   3-4 sentence hook in Nigerian context.

6. LESSON DEVELOPMENT (25 minutes)
   Use this format for EACH step (at least 3 steps):
   <h4>Step N: [Title] (X min)</h4>
   <p><strong>Teacher Activity:</strong> [description]</p>
   <p><strong>Student Activity:</strong> [description]</p>
   DO NOT use HTML tables for this section.

7. ENRICHMENT SECTION
   - 3 Nigerian context examples (as <h4>Nigerian Context Examples:</h4> then bulleted list)
   - 1 group activity (as <h4>Group Activity:</h4> then paragraph)
   - 1 differentiation tip (as <h4>Differentiation Tip:</h4> then paragraph)

8. EVALUATION (5 minutes)
   Numbered list of 3-5 verbal questions from the Evaluation Guide.

9. SUMMARY & CONCLUSION
   2-3 sentences.

10. ASSIGNMENT
    1 practical homework task.

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

    max_retries = 4
    base_delay = 1.0

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            html_content = response.text.strip()
            if html_content.startswith("```html"):
                html_content = html_content[7:]
            if html_content.startswith("```"):
                html_content = html_content[3:]
            if html_content.endswith("```"):
                html_content = html_content[:-3]
            return html_content.strip()

        except Exception as e:
            error_str = str(e)
            is_retryable = (
                "503" in error_str or
                "429" in error_str or
                "UNAVAILABLE" in error_str or
                "high demand" in error_str
            )
            if is_retryable and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Attempt {attempt + 1} failed (retryable). Retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"AI generation failed after {attempt + 1} attempts: {error_str}"
                )

    raise HTTPException(status_code=500, detail="AI generation failed: max retries exceeded")

# ---------- HTML -> DOCX ----------
def html_to_docx(html_str: str, title: str) -> bytes:
    doc = Document()
    doc.add_heading(title, level=0)

    soup = BeautifulSoup(html_str, "html.parser")

    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        # Skip anything nested inside a table (avoid double-walk)
        if element.find_parent("table") is not None:
            continue

        if element.name == "h1":
            doc.add_heading(element.get_text(strip=True), level=1)
        elif element.name == "h2":
            doc.add_heading(element.get_text(strip=True), level=2)
        elif element.name == "h3":
            doc.add_heading(element.get_text(strip=True), level=3)
        elif element.name == "h4":
            doc.add_heading(element.get_text(strip=True), level=4)
        elif element.name == "p":
            text = element.get_text(strip=True)
            if text:
                doc.add_paragraph(text)
        elif element.name == "li":
            doc.add_paragraph(element.get_text(strip=True), style="List Bullet")
        elif element.name == "table":
            rows = element.find_all("tr")
            if rows:
                cols = max(len(r.find_all(["th", "td"])) for r in rows)
                table = doc.add_table(rows=0, cols=cols)
                table.style = "Table Grid"
                for row in rows:
                    cells = row.find_all(["th", "td"])
                    row_cells = table.add_row().cells
                    for i, cell in enumerate(cells):
                        if i < cols:
                            row_cells[i].text = cell.get_text(strip=True)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()

# ---------- HTML -> PDF ----------
def html_to_pdf(html_str: str, title: str) -> bytes:
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"><title>{title}</title>
    <style>
        @page {{ size: A4; margin: 18mm; }}
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #222; }}
        h1, h2, h3, h4 {{ color: #1a5276; }}
        h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 18px; }}
        h4 {{ margin-top: 12px; margin-bottom: 4px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; page-break-inside: auto; }}
        tr {{ page-break-inside: avoid; }}
        th, td {{ border: 1px solid #333; padding: 8px; text-align: left; vertical-align: top; word-wrap: break-word; }}
        th {{ background: #eaf2f8; }}
        ul, ol {{ margin: 6px 0 6px 18px; }}
        p {{ margin: 6px 0; }}
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
    return {"status": "alive", "service": "Lesson Note AI", "model": MODEL_NAME}

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME}

@app.post("/generate-lesson-note")
def generate_lesson_note(req: LessonNoteRequest):
    html = generate_lesson_html(req)
    return {"html": html, "topic": req.topic, "week": req.week_number}

@app.post("/generate-lesson-note/pdf")
def generate_lesson_pdf(req: LessonNoteRequest):
    html = generate_lesson_html(req)
    pdf_bytes = html_to_pdf(html, f"Lesson Note - {req.topic}")
    filename = f"Lesson_Note_{req.subject}_Week{req.week_number}.pdf".replace(" ", "_")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@app.post("/generate-lesson-note/docx")
def generate_lesson_docx(req: LessonNoteRequest):
    html = generate_lesson_html(req)
    docx_bytes = html_to_docx(html, f"Lesson Note - {req.topic}")
    filename = f"Lesson_Note_{req.subject}_Week{req.week_number}.docx".replace(" ", "_")
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
