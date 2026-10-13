"""Resume Tailor — the core rewrite engine.

Rewrites resume bullets to align with a target JD while NEVER
fabricating experience, metrics, tools, or outcomes.
"""

import json
import logging
from generation.prompts import TAILOR_RESUME_PROMPT, NO_FABRICATION_CONSTRAINT

logger = logging.getLogger(__name__)


def tailor_resume(master_resume: dict, jd_requirements: dict, match_score: dict) -> dict:
    """Tailor a resume to match a job description.

    Returns dict with: tailored_resume, gaps, changes_summary
    """
    from llm_provider import get_llm

    llm = get_llm(temperature=0.2)
    prompt = TAILOR_RESUME_PROMPT.format(
        no_fabrication_constraint=NO_FABRICATION_CONSTRAINT,
        master_resume=json.dumps(master_resume, indent=2),
        jd_requirements=json.dumps(jd_requirements, indent=2),
        match_score=match_score.get("overall_score", 0),
        missing_keywords=", ".join(match_score.get("missing_keywords", [])[:20]),
        weak_bullets=json.dumps(match_score.get("weak_bullets", [])[:5]),
    )

    response = llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tailored resume JSON: {e}")
        result = {
            "tailored_resume": master_resume,
            "gaps": [],
            "changes_summary": ["Error: Could not parse LLM response. Original resume returned."],
        }

    # Ensure required keys exist
    if "tailored_resume" not in result:
        result["tailored_resume"] = master_resume
    if "gaps" not in result:
        result["gaps"] = []
    if "changes_summary" not in result:
        result["changes_summary"] = []

    return result
