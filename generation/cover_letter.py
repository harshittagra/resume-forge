"""Cover Letter Generator — grounded only in the tailored resume + JD."""

import json
import logging
from generation.prompts import COVER_LETTER_PROMPT, NO_FABRICATION_CONSTRAINT

logger = logging.getLogger(__name__)


def generate_cover_letter(tailored_resume: dict, jd_requirements: dict) -> dict:
    """Generate a role-specific cover letter.

    Returns dict with: cover_letter, key_points, tone
    """
    from llm_provider import get_llm

    llm = get_llm(temperature=0.3)
    prompt = COVER_LETTER_PROMPT.format(
        no_fabrication_constraint=NO_FABRICATION_CONSTRAINT,
        tailored_resume=json.dumps(tailored_resume, indent=2),
        jd_requirements=json.dumps(jd_requirements, indent=2),
    )

    response = llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse cover letter JSON: {e}")
        return {
            "cover_letter": "Error generating cover letter. Please try again.",
            "key_points": [],
            "tone": "professional",
        }
