"""ATS Match Scorer — 100% offline, zero LLM cost.

Score = 40% keyword overlap + 60% semantic similarity (sentence-transformers).
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

_model = None

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "it", "its", "this", "that", "these", "those", "i", "we", "you",
    "he", "she", "they", "me", "us", "him", "her", "them", "my", "our",
    "your", "his", "their", "what", "which", "who", "whom", "where",
    "when", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "about", "above", "after",
    "again", "also", "as", "because", "before", "between", "during",
    "into", "over", "through", "under", "until", "up", "while",
    "work", "working", "worked", "experience", "experienced", "role",
    "ability", "team", "teams", "company", "strong", "including",
    "required", "preferred", "minimum", "years", "year", "new", "well",
    "using", "used", "use", "etc", "within", "across",
}

_WEAK_PATTERNS = [
    r"^responsible for\b", r"^helped\b", r"^assisted\b", r"^worked on\b",
    r"^worked with\b", r"^involved in\b", r"^participated in\b",
    r"^handled\b", r"^managed various\b", r"^supported\b", r"^contributed to\b",
]


def _get_model():
    """Lazy-load sentence-transformers model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Could not load sentence-transformers: {e}")
    return _model


def extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text."""
    if not text:
        return set()
    text_lower = text.lower()

    phrases = set()
    for match in re.finditer(r'\b([a-z][a-z\+\#\.]+(?:\s+[a-z][a-z\+\#\.]+){1,2})\b', text_lower):
        phrase = match.group(1).strip()
        words = phrase.split()
        if not all(w in _STOP_WORDS for w in words):
            phrases.add(phrase)

    words = set()
    for match in re.finditer(r'\b([a-z][a-z\+\#\.]{2,})\b', text_lower):
        word = match.group(1)
        if word not in _STOP_WORDS and len(word) >= 3:
            words.add(word)

    for match in re.finditer(r'\b([A-Z][A-Za-z\+\#\.]{1,})\b', text):
        words.add(match.group(1).lower())

    return words | phrases


def _flatten_resume_text(resume: dict) -> str:
    """Flatten structured resume into text for keyword extraction."""
    parts = []
    if resume.get("summary"):
        parts.append(resume["summary"])
    for exp in resume.get("experience", []):
        parts.append(f"{exp.get('title', '')} at {exp.get('company', '')}")
        parts.extend(exp.get("bullets", []))
    skills = resume.get("skills", {})
    if isinstance(skills, dict):
        for cat in skills.values():
            if isinstance(cat, list):
                parts.extend(cat)
    for proj in resume.get("projects", []):
        parts.append(proj.get("name", ""))
        parts.append(proj.get("description", ""))
        techs = proj.get("technologies", [])
        if isinstance(techs, list):
            parts.extend(techs)
    return " ".join(parts)


def _flatten_jd_text(jd: dict) -> str:
    """Flatten structured JD into text."""
    parts = [jd.get("role_title", "")]
    parts.extend(jd.get("required_skills", []))
    parts.extend(jd.get("nice_to_have_skills", []))
    parts.extend(jd.get("key_responsibilities", []))
    parts.extend(jd.get("keywords", []))
    return " ".join(parts)


def _get_resume_bullets(resume: dict) -> List[str]:
    """Get all experience bullets from a resume."""
    bullets = []
    for exp in resume.get("experience", []):
        bullets.extend(exp.get("bullets", []))
    return bullets


def _get_jd_items(jd: dict) -> List[str]:
    """Get responsibilities + skills from JD for semantic matching."""
    items = list(jd.get("key_responsibilities", []))
    items.extend([f"Experience with {s}" for s in jd.get("required_skills", [])])
    return items


def _identify_weak_bullets(bullets: List[str]) -> List[str]:
    """Identify generic/weak bullets that should be strengthened."""
    weak = []
    for bullet in bullets:
        bl = bullet.lower().strip()
        for pattern in _WEAK_PATTERNS:
            if re.match(pattern, bl):
                weak.append(bullet)
                break
    return weak


def compute_match_score(resume: dict, jd: dict) -> Dict:
    """Compute ATS match score — 40% keyword overlap + 60% semantic similarity.

    All computed locally — zero API cost.
    """
    # ── Keyword overlap (40%) ─────────────────────────────────────
    resume_text = _flatten_resume_text(resume)
    jd_text = _flatten_jd_text(jd)
    resume_kw = extract_keywords(resume_text)
    jd_kw = extract_keywords(jd_text)

    for skill in jd.get("required_skills", []) + jd.get("nice_to_have_skills", []) + jd.get("keywords", []):
        jd_kw.add(skill.lower())
    skills = resume.get("skills", {})
    if isinstance(skills, dict):
        for cat in skills.values():
            if isinstance(cat, list):
                for s in cat:
                    resume_kw.add(s.lower())

    matched = resume_kw & jd_kw
    missing = {k for k in (jd_kw - resume_kw) if len(k) > 2 and k not in _STOP_WORDS}
    kw_score = min(100, int((len(matched) / max(len(jd_kw), 1)) * 100)) if jd_kw else 0

    # ── Semantic similarity (60%) ─────────────────────────────────
    sem_score = 0
    model = _get_model()
    resume_bullets = _get_resume_bullets(resume)
    jd_items = _get_jd_items(jd)

    if model and resume_bullets and jd_items:
        try:
            from sentence_transformers.util import cos_sim
            b_emb = model.encode(resume_bullets, convert_to_tensor=True)
            j_emb = model.encode(jd_items, convert_to_tensor=True)
            sim = cos_sim(b_emb, j_emb)
            best = sim.max(dim=0).values
            sem_score = max(0, min(100, int(best.mean().item() * 100)))
        except Exception as e:
            logger.warning(f"Semantic scoring failed: {e}")
            sem_score = kw_score
    else:
        sem_score = kw_score

    overall = int(kw_score * 0.4 + sem_score * 0.6)
    weak = _identify_weak_bullets(resume_bullets)

    strong = []
    if model and resume_bullets and jd_items:
        try:
            from sentence_transformers.util import cos_sim
            b_emb = model.encode(resume_bullets, convert_to_tensor=True)
            j_emb = model.encode(jd_items, convert_to_tensor=True)
            sim = cos_sim(b_emb, j_emb)
            for i, bullet in enumerate(resume_bullets):
                if sim[i].max().item() > 0.5:
                    strong.append(bullet)
        except Exception:
            pass

    return {
        "overall_score": overall,
        "keyword_score": kw_score,
        "semantic_score": sem_score,
        "matched_keywords": sorted(list(matched)),
        "missing_keywords": sorted(list(missing)),
        "weak_bullets": weak,
        "strong_bullets": strong,
        "total_jd_keywords": len(jd_kw),
        "total_matched": len(matched),
        "total_missing": len(missing),
    }
