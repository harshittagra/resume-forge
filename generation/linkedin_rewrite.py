"""LinkedIn Rewrite — headline + About section for target role."""

import json
import logging
from generation.prompts import LINKEDIN_REWRITE_PROMPT, NO_FABRICATION_CONSTRAINT

logger = logging.getLogger(__name__)


def rewrite_linkedin(tailored_resume: dict, jd_requirements: dict) -> dict:
    """Rewrite LinkedIn headline and About section for the target role.

    Returns dict with: headline, about, keywords_used
    """
    from llm_provider import get_llm

    llm = get_llm(temperature=0.3)
    prompt = LINKEDIN_REWRITE_PROMPT.format(
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
        logger.error(f"Failed to parse LinkedIn rewrite JSON: {e}")
        return {
            "headline": "Error generating headline. Please try again.",
            "about": "Error generating About section. Please try again.",
            "keywords_used": [],
        }
