import os
import io
import json
import time
import random
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Depends
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime,
    ForeignKey, select
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from google import genai
from google.genai import types as genai_types
from weasyprint import HTML
from docx import Document
from bs4 import BeautifulSoup
from pypdf import PdfReader
import uvicorn

# ---------- Config ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY", "change-me-later")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.8-flash"

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./curriculum.db")

# ---------- Database ----------
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Curriculum(Base):
    __tablename__ = "curricula"
    id = Column(Integer, primary_key=True, index=True)
    class_level = Column(String(50), nullable=False)
    subject = Column(String(255), nullable=False)
    term = Column(String(50), nullable=False, default="First Term")
    created_at = Column(DateTime, default=datetime.utcnow)
    weeks = relationship("Week", back_populates="curriculum", cascade="all, delete-orphan")


class Week(Base):
    __tablename__ = "weeks"
    id = Column(Integer, primary_key=True, index=True)
    curriculum_id = Column(Integer, ForeignKey("curricula.id"), nullable=False)
    week_number = Column(Integer, nullable=False)
    topic = Column(String(500), nullable=False)
    performance_objectives = Column(Text, default="[]")
    content_knowledge = Column(Text, default="[]")
    key_competencies = Column(Text, default="[]")
    learning_activities = Column(Text, default="[]")
    teaching_resources = Column(Text, default="[]")
    evaluation_guide = Column(Text, default="[]")
    curriculum = relationship("Curriculum", back_populates="weeks")


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def week_to_dict(w: Week):
    return {
        "id": w.id,
        "curriculum_id": w.curriculum_id,
        "week_number": w.week_number,
        "topic": w.topic,
        "performance_objectives": json.loads(w.performance_objectives or "[]"),
        "content_knowledge": json.loads(w.content_knowledge or "[]"),
        "key_competencies": json.loads(w.key_competencies or "[]"),
        "learning_activities": json.loads(w.learning_activities or "[]"),
        "teaching_resources": json.loads(w.teaching_resources or "[]"),
        "evaluation_guide": json.loads(w.evaluation_guide or "[]"),
    }


# ---------- Pydantic models ----------
class WeekInput(BaseModel):
    week_number: int
    topic: str
    performance_objectives: List[str] = []
    content_knowledge: List[str] = []
    key_competencies: List[str] = []
    learning_activities: List[str] = []
    teaching_resources: List[str] = []
    evaluation_guide: List[str] = []


class CurriculumInput(BaseModel):
    class_level: str
    subject: str
    term: str = "First Term"
    weeks: List[WeekInput]


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


# ---------- Admin auth ----------
def require_admin(x_admin_key: Optional[str] = Header(None)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Key header")
    return True


# ---------- Lesson Note Prompt ----------
PROMPT_TEMPLATE = """
You are an expert Nigerian teacher. Generate a complete NERDC-compliant LESSON NOTE.

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

Use simple English appropriate to {class_level}. Do not invent topics outside the data.
Return ONLY the HTML body content (no <html> or <body> tags, just inner HTML).
"""


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
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=16384,
                    temperature=0.7,
                ),
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
            is_retryable = any(k in error_str for k in ["503", "429", "UNAVAILABLE", "high demand"])
            if is_retryable and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Attempt {attempt + 1} failed. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            raise HTTPException(status_code=500, detail=f"AI generation failed: {error_str}")
    raise HTTPException(status_code=500, detail="AI generation failed: max retries exceeded")


# ---------- HTML -> DOCX ----------
def html_to_docx(html_str: str, title: str) -> bytes:
    doc = Document()
    doc.add_heading(title, level=0)
    soup = BeautifulSoup(html_str, "html.parser")
    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
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
        return HTML(string=full_html).write_pdf()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# ---------- FastAPI app ----------
app = FastAPI(title="Lesson Note AI Backend")


@app.on_event("startup")
def seed_if_empty():
    db = SessionLocal()
    try:
        count = db.execute(select(Curriculum)).scalars().first()
        if count is None:
            cur = Curriculum(
                class_level="JSS 1",
                subject="Solar PV Installation & Maintenance",
                term="First Term"
            )
            db.add(cur)
            db.flush()
            w = Week(
                curriculum_id=cur.id,
                week_number=1,
                topic="Solar Photovoltaics (PV) System",
                performance_objectives=json.dumps([
                    "define Solar Photovoltaics",
                    "identify types of Solar Photovoltaic Modules",
                    "explain how a Solar Photovoltaic system generates electricity"
                ]),
                content_knowledge=json.dumps([
                    "Meaning of Solar Photovoltaics (PV)",
                    "Types of Solar Photovoltaics (PV) Modules",
                    "Solar PV and Electricity generation"
                ]),
                key_competencies=json.dumps(["Communication", "ICT and Digital Competencies"]),
                learning_activities=json.dumps([
                    "Discuss in groups the meaning of Solar PV system",
                    "Watch short videos on types of PV modules",
                    "Demonstrate how PV generates electricity"
                ]),
                teaching_resources=json.dumps([
                    "Internet enabled digital tools",
                    "Sample/Model of Solar PV system components",
                    "Solar Panels, Inverters, Batteries"
                ]),
                evaluation_guide=json.dumps([
                    "explain how a solar PV system generates electricity",
                    "list 2 types of Solar PV Modules"
                ])
            )
            db.add(w)
            db.commit()
            print("Seeded Week 1 of Solar PV JSS 1.")
    finally:
        db.close()


@app.get("/")
def root():
    return {"status": "alive", "service": "Lesson Note AI", "model": MODEL_NAME, "version": "step3.1"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME}


# ---------- Admin endpoints ----------
@app.post("/admin/curriculum/json")
def admin_save_json(payload: CurriculumInput, _: bool = Depends(require_admin), db: Session = Depends(get_db)):
    cur = Curriculum(class_level=payload.class_level, subject=payload.subject, term=payload.term or "First Term")
    db.add(cur)
    db.flush()
    for w in payload.weeks:
        db.add(Week(
            curriculum_id=cur.id,
            week_number=w.week_number,
            topic=w.topic,
            performance_objectives=json.dumps(w.performance_objectives),
            content_knowledge=json.dumps(w.content_knowledge),
            key_competencies=json.dumps(w.key_competencies),
            learning_activities=json.dumps(w.learning_activities),
            teaching_resources=json.dumps(w.teaching_resources),
            evaluation_guide=json.dumps(w.evaluation_guide),
        ))
    db.commit()
    return {"ok": True, "curriculum_id": cur.id, "weeks_saved": len(payload.weeks)}


@app.post("/admin/curriculum/pdf")
async def admin_extract_pdf(
    file: UploadFile = File(...),
    _: bool = Depends(require_admin)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files accepted")
    raw = await file.read()
    try:
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")

    if len(text.strip()) < 200:
        raise HTTPException(status_code=400, detail="Extracted text too short — is this a scanned PDF?")

    # Trim to ~200k chars to stay well inside Gemini's context window
    source_text = text[:200000]

    extraction_prompt = f"""
You are a curriculum parser for Nigerian NERDC documents.

Extract ALL weekly curriculum entries from the source text below into STRICT JSON.

Return ONLY valid JSON matching this schema, no other text:
{{
  "class_level": "e.g. JSS 1 / SSS 1 / SSS 2",
  "subject": "e.g. Mathematics / English Language",
  "term": "First Term",
  "weeks": [
    {{
      "week_number": 1,
      "topic": "...",
      "performance_objectives": ["...", "..."],
      "content_knowledge": ["...", "..."],
      "key_competencies": ["...", "..."],
      "learning_activities": ["...", "..."],
      "teaching_resources": ["...", "..."],
      "evaluation_guide": ["...", "..."]
    }}
  ]
}}

CRITICAL RULES:
- Extract EVERY week/topic in the document. Do not stop early.
- If the document contains 12 weeks, return 12 weeks. If 19, return 19.
- If a field is missing for a week, use []. Do NOT leave a sentence or list item unfinished.
- Do NOT truncate. Keep going until the entire document has been processed.
- Preserve the Nigerian curriculum wording where possible.
- If term is not stated, default to "First Term".
- Output pure JSON only (no markdown fences, no explanations).

=== SOURCE TEXT ===
{source_text}
"""

    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=extraction_prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=32768,
                temperature=0.1,
            ),
        )
        raw_out = resp.text.strip()
        if raw_out.startswith("```json"):
            raw_out = raw_out[7:]
        if raw_out.startswith("```"):
            raw_out = raw_out[3:]
        if raw_out.endswith("```"):
            raw_out = raw_out[:-3]
        raw_out = raw_out.strip()
        parsed = json.loads(raw_out)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e}. First 500 chars: {raw_out[:500]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI extraction failed: {e}")

    weeks = parsed.get("weeks", [])
    return {
        "ok": True,
        "weeks_extracted": len(weeks),
        "message": "Review. If correct, POST /admin/curriculum/json with X-Admin-Key.",
        "extracted": parsed
    }


@app.get("/admin/curricula")
def admin_list(_: bool = Depends(require_admin), db: Session = Depends(get_db)):
    items = db.execute(select(Curriculum).order_by(Curriculum.id)).scalars().all()
    return [
        {
            "id": c.id,
            "class_level": c.class_level,
            "subject": c.subject,
            "term": c.term,
            "weeks_count": len(c.weeks),
            "created_at": c.created_at.isoformat() if c.created_at else None
        } for c in items
    ]


@app.delete("/admin/curriculum/{curriculum_id}")
def admin_delete(curriculum_id: int, _: bool = Depends(require_admin), db: Session = Depends(get_db)):
    cur = db.get(Curriculum, curriculum_id)
    if not cur:
        raise HTTPException(status_code=404, detail="Curriculum not found")
    db.delete(cur)
    db.commit()
    return {"ok": True, "deleted": curriculum_id}


# ---------- Public read endpoints ----------
@app.get("/curriculum/classes")
def list_classes(db: Session = Depends(get_db)):
    rows = db.execute(select(Curriculum.class_level).distinct()).scalars().all()
    return sorted(rows)


@app.get("/curriculum/{class_level}/subjects")
def list_subjects(class_level: str, db: Session = Depends(get_db)):
    rows = db.execute(
        select(Curriculum.subject).where(Curriculum.class_level == class_level).distinct()
    ).scalars().all()
    return sorted(rows)


@app.get("/curriculum/{class_level}/{subject}/terms")
def list_terms(class_level: str, subject: str, db: Session = Depends(get_db)):
    rows = db.execute(
        select(Curriculum.term)
        .where(Curriculum.class_level == class_level, Curriculum.subject == subject)
        .distinct()
    ).scalars().all()
    return sorted(rows)


@app.get("/curriculum/{class_level}/{subject}/{term}/weeks")
def list_weeks(class_level: str, subject: str, term: str, db: Session = Depends(get_db)):
    cur = db.execute(
        select(Curriculum)
        .where(
            Curriculum.class_level == class_level,
            Curriculum.subject == subject,
            Curriculum.term == term
        )
    ).scalars().first()
    if not cur:
        raise HTTPException(status_code=404, detail="Curriculum not found")
    weeks = sorted(cur.weeks, key=lambda w: w.week_number)
    return [
        {"id": w.id, "week_number": w.week_number, "topic": w.topic}
        for w in weeks
    ]


@app.get("/curriculum/week/{week_id}")
def get_week(week_id: int, db: Session = Depends(get_db)):
    w = db.get(Week, week_id)
    if not w:
        raise HTTPException(status_code=404, detail="Week not found")
    return week_to_dict(w)


# ---------- Generate from DB ----------
def _req_from_week(w: Week, cur: Curriculum) -> LessonNoteRequest:
    d = week_to_dict(w)
    return LessonNoteRequest(
        class_level=cur.class_level,
        subject=cur.subject,
        term=cur.term,
        week_number=d["week_number"],
        topic=d["topic"],
        performance_objectives=d["performance_objectives"],
        content_knowledge=d["content_knowledge"],
        key_competencies=d["key_competencies"],
        learning_activities=d["learning_activities"],
        teaching_resources=d["teaching_resources"],
        evaluation_guide=d["evaluation_guide"],
    )


@app.post("/generate-from-db/{week_id}")
def generate_from_db(week_id: int, db: Session = Depends(get_db)):
    w = db.get(Week, week_id)
    if not w:
        raise HTTPException(status_code=404, detail="Week not found")
    req = _req_from_week(w, w.curriculum)
    html = generate_lesson_html(req)
    return {"html": html, "week_id": week_id, "topic": w.topic}


@app.post("/generate-from-db/{week_id}/pdf")
def generate_from_db_pdf(week_id: int, db: Session = Depends(get_db)):
    w = db.get(Week, week_id)
    if not w:
        raise HTTPException(status_code=404, detail="Week not found")
    req = _req_from_week(w, w.curriculum)
    html = generate_lesson_html(req)
    pdf_bytes = html_to_pdf(html, f"Lesson Note - {w.topic}")
    filename = f"Lesson_Note_{req.subject}_Week{w.week_number}.pdf".replace(" ", "_")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.post("/generate-from-db/{week_id}/docx")
def generate_from_db_docx(week_id: int, db: Session = Depends(get_db)):
    w = db.get(Week, week_id)
    if not w:
        raise HTTPException(status_code=404, detail="Week not found")
    req = _req_from_week(w, w.curriculum)
    html = generate_lesson_html(req)
    docx_bytes = html_to_docx(html, f"Lesson Note - {w.topic}")
    filename = f"Lesson_Note_{req.subject}_Week{w.week_number}.docx".replace(" ", "_")
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ---------- Legacy endpoints (kept) ----------
@app.post("/generate-lesson-note")
def generate_lesson_note(req: LessonNoteRequest):
    html = generate_lesson_html(req)
    return {"html": html, "topic": req.topic, "week": req.week_number}


@app.post("/generate-lesson-note/pdf")
def generate_lesson_pdf(req: LessonNoteRequest):
    html = generate_lesson_html(req)
    pdf_bytes = html_to_pdf(html, f"Lesson Note - {req.topic}")
    filename = f"Lesson_Note_{req.subject}_Week{req.week_number}.pdf".replace(" ", "_")
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.post("/generate-lesson-note/docx")
def generate_lesson_docx(req: LessonNoteRequest):
    html = generate_lesson_html(req)
    docx_bytes = html_to_docx(html, f"Lesson Note - {req.topic}")
    filename = f"Lesson_Note_{req.subject}_Week{req.week_number}.docx".replace(" ", "_")
    return Response(content=docx_bytes,
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
