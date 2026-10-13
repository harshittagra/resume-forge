"""Resume Parser — PDF/DOCX/Text → Structured JSON.

The structured JSON becomes the "master resume" — the single source
of truth that all tailored resumes are generated from.
"""

import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def extract_text_from_pdf(file) -> str:
    """Extract text from a PDF file using pdfplumber."""
    import pdfplumber
    text_parts = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def extract_text_from_docx(file) -> str:
    """Extract text from a DOCX file using python-docx."""
    from docx import Document
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text(file, filename: str) -> str:
    """Extract text from any supported file format."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        return extract_text_from_pdf(file)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file)
    else:
        content = file.read()
        return content.decode("utf-8") if isinstance(content, bytes) else content


def structure_resume_with_llm(raw_text: str) -> dict:
    """Use LLM to parse raw resume text into structured JSON."""
    from llm_provider import get_llm
    from generation.prompts import RESUME_STRUCTURE_PROMPT

    llm = get_llm(temperature=0.0)
    prompt = RESUME_STRUCTURE_PROMPT.format(resume_text=raw_text)
    response = llm.invoke(prompt)
    content = response.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return get_empty_resume()


def get_empty_resume() -> dict:
    """Return an empty resume structure template."""
    return {
        "contact": {"name": "", "email": "", "phone": "", "linkedin": "", "location": "", "website": ""},
        "summary": "",
        "experience": [],
        "skills": {"technical": [], "tools": [], "soft": [], "certifications": []},
        "education": [],
        "projects": [],
    }


def save_master_resume(data: dict) -> str:
    """Save the master resume to local JSON file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "master_resume.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_master_resume() -> Optional[dict]:
    """Load the master resume from local JSON file."""
    path = os.path.join(DATA_DIR, "master_resume.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def master_resume_exists() -> bool:
    """Check if a master resume has been saved."""
    return os.path.exists(os.path.join(DATA_DIR, "master_resume.json"))
