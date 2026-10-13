"""Skill Gap Project Builder — AI-generated weekend project ideas.

When the ATS scorer finds skills the JD requires but the resume lacks,
this module generates practical weekend project ideas that would let
the user genuinely acquire and demonstrate those skills.
"""

import json
import logging
from generation.prompts import SKILL_GAP_PROJECTS_PROMPT

logger = logging.getLogger(__name__)


def generate_skill_gap_projects(gaps: list, jd_requirements: dict) -> dict:
    """Generate weekend project ideas for each skill gap.

    Args:
        gaps: List of gap dicts from the tailor pipeline.
        jd_requirements: Structured JD requirements.

    Returns dict with project suggestions per gap.
    """
    if not gaps:
        return {"projects": []}

    from llm_provider import invoke_llm

    gap_descriptions = []
    for gap in gaps[:5]:  # Limit to top 5 gaps
        gap_descriptions.append(f"- {gap.get('requirement', 'Unknown skill')}")

    role = jd_requirements.get("role_title", "the target role")
    company = jd_requirements.get("company", "the target company")

    prompt = SKILL_GAP_PROJECTS_PROMPT.format(
        role_title=role,
        company=company,
        gaps="\n".join(gap_descriptions),
        industry=jd_requirements.get("industry", "Technology"),
    )

    try:
        content = invoke_llm(prompt, temperature=0.3)
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)
        result = json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Skill gap project generation failed: {e}")
        result = {"projects": []}

    if "projects" not in result:
        result["projects"] = []

    return result
