"""LangGraph Pipeline — Full orchestration graph.

Flow: parse_jd → score_match → tailor_resume → cover_letter → linkedin_rewrite
"""

import json
import os
import logging
from typing import Any, Dict, List, TypedDict
from datetime import datetime

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ══════════════════════════════════════════════════════════════════
# State definition
# ══════════════════════════════════════════════════════════════════

class TailorState(TypedDict):
    """Typed state flowing through every node."""
    master_resume: dict
    jd_text: str
    jd_requirements: dict
    match_score: dict
    tailored_resume: dict
    gaps: list
    changes_summary: list
    cover_letter: dict
    linkedin: dict
    processing_trace: List[str]
    error: str


# ══════════════════════════════════════════════════════════════════
# Node functions
# ══════════════════════════════════════════════════════════════════

def node_parse_jd(state: TailorState) -> Dict[str, Any]:
    """Node 1 — Parse the job description into structured requirements."""
    from parsers.jd_parser import parse_job_description

    try:
        jd_req = parse_job_description(state["jd_text"])
        role = jd_req.get("role_title", "Unknown Role")
        n_skills = len(jd_req.get("required_skills", []))
        n_resp = len(jd_req.get("key_responsibilities", []))
        return {
            "jd_requirements": jd_req,
            "processing_trace": [f"🔍 **JD Analyzed** — {role}: {n_skills} required skills, {n_resp} responsibilities extracted"],
        }
    except Exception as e:
        return {"error": f"JD parsing failed: {str(e)}", "processing_trace": [f"❌ JD parsing error: {e}"]}


def node_score_match(state: TailorState) -> Dict[str, Any]:
    """Node 2 — Compute ATS match score (offline, no LLM cost)."""
    from scoring.match_score import compute_match_score

    try:
        score = compute_match_score(state["master_resume"], state["jd_requirements"])
        return {
            "match_score": score,
            "processing_trace": [
                f"📊 **ATS Score: {score['overall_score']}/100** — "
                f"Keyword: {score['keyword_score']}% | Semantic: {score['semantic_score']}% | "
                f"{score['total_matched']} matched, {score['total_missing']} missing keywords"
            ],
        }
    except Exception as e:
        return {
            "match_score": {"overall_score": 0, "keyword_score": 0, "semantic_score": 0,
                           "matched_keywords": [], "missing_keywords": [], "weak_bullets": [],
                           "strong_bullets": [], "total_jd_keywords": 0, "total_matched": 0, "total_missing": 0},
            "processing_trace": [f"⚠️ Scoring fallback (error: {e})"],
        }


def node_tailor_resume(state: TailorState) -> Dict[str, Any]:
    """Node 3 — Tailor the resume (the core rewrite engine)."""
    from generation.tailor_resume import tailor_resume

    try:
        result = tailor_resume(state["master_resume"], state["jd_requirements"], state["match_score"])
        n_gaps = len(result.get("gaps", []))
        n_changes = len(result.get("changes_summary", []))
        return {
            "tailored_resume": result.get("tailored_resume", state["master_resume"]),
            "gaps": result.get("gaps", []),
            "changes_summary": result.get("changes_summary", []),
            "processing_trace": [f"✍️ **Resume Tailored** — {n_changes} changes made, {n_gaps} gaps flagged"],
        }
    except Exception as e:
        return {
            "tailored_resume": state["master_resume"],
            "gaps": [], "changes_summary": [],
            "processing_trace": [f"❌ Tailoring error: {e}"],
            "error": str(e),
        }


def node_cover_letter(state: TailorState) -> Dict[str, Any]:
    """Node 4 — Generate a cover letter."""
    from generation.cover_letter import generate_cover_letter

    try:
        cl = generate_cover_letter(state["tailored_resume"], state["jd_requirements"])
        return {
            "cover_letter": cl,
            "processing_trace": [f"📝 **Cover Letter Generated** — {len(cl.get('cover_letter', '').split())} words"],
        }
    except Exception as e:
        return {
            "cover_letter": {"cover_letter": "", "key_points": [], "tone": ""},
            "processing_trace": [f"⚠️ Cover letter error: {e}"],
        }


def node_linkedin(state: TailorState) -> Dict[str, Any]:
    """Node 5 — Rewrite LinkedIn headline + About."""
    from generation.linkedin_rewrite import rewrite_linkedin

    try:
        li = rewrite_linkedin(state["tailored_resume"], state["jd_requirements"])
        return {
            "linkedin": li,
            "processing_trace": [f"💼 **LinkedIn Rewritten** — Headline: {len(li.get('headline', ''))} chars"],
        }
    except Exception as e:
        return {
            "linkedin": {"headline": "", "about": "", "keywords_used": []},
            "processing_trace": [f"⚠️ LinkedIn rewrite error: {e}"],
        }


# ══════════════════════════════════════════════════════════════════
# Graph construction
# ══════════════════════════════════════════════════════════════════

def build_tailor_graph():
    """Build and compile the LangGraph workflow.

    Flow: parse_jd → score_match → tailor_resume → cover_letter → linkedin → END
    """
    workflow = StateGraph(TailorState)

    workflow.add_node("parse_jd", node_parse_jd)
    workflow.add_node("score_match", node_score_match)
    workflow.add_node("tailor_resume", node_tailor_resume)
    workflow.add_node("cover_letter", node_cover_letter)
    workflow.add_node("linkedin", node_linkedin)

    workflow.set_entry_point("parse_jd")
    workflow.add_edge("parse_jd", "score_match")
    workflow.add_edge("score_match", "tailor_resume")
    workflow.add_edge("tailor_resume", "cover_letter")
    workflow.add_edge("cover_letter", "linkedin")
    workflow.add_edge("linkedin", END)

    return workflow.compile()


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def run_tailor_pipeline(master_resume: dict, jd_text: str) -> Dict[str, Any]:
    """Run the full tailoring pipeline.

    Args:
        master_resume: Structured master resume dict.
        jd_text: Raw job description text.

    Returns dict with all pipeline outputs.
    """
    graph = build_tailor_graph()

    initial_state: TailorState = {
        "master_resume": master_resume,
        "jd_text": jd_text,
        "jd_requirements": {},
        "match_score": {},
        "tailored_resume": {},
        "gaps": [],
        "changes_summary": [],
        "cover_letter": {},
        "linkedin": {},
        "processing_trace": [],
        "error": "",
    }

    final_state = graph.invoke(initial_state)

    # Save to history
    _save_history(final_state)

    return final_state


def _save_history(state: dict):
    """Append this tailoring run to history.json."""
    os.makedirs(DATA_DIR, exist_ok=True)
    history_path = os.path.join(DATA_DIR, "history.json")

    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    jd_req = state.get("jd_requirements", {})
    match = state.get("match_score", {})

    entry = {
        "timestamp": datetime.now().isoformat(),
        "role_title": jd_req.get("role_title", "Unknown"),
        "company": jd_req.get("company", "Unknown"),
        "score_before": match.get("overall_score", 0),
        "gaps_count": len(state.get("gaps", [])),
        "changes_count": len(state.get("changes_summary", [])),
    }

    history.append(entry)

    try:
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to save history: {e}")
