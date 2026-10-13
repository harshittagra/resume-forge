"""Job Description Parser — JD text → structured requirements via LLM."""

import json
import logging

logger = logging.getLogger(__name__)


def parse_job_description(jd_text: str) -> dict:
    """Parse a job description into structured requirements using LLM."""
    from llm_provider import get_llm
    from generation.prompts import JD_PARSE_PROMPT

    llm = get_llm(temperature=0.0)
    prompt = JD_PARSE_PROMPT.format(jd_text=jd_text)
    response = llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JD analysis as JSON: {e}")
        return get_empty_jd()


def get_empty_jd() -> dict:
    """Return an empty JD structure template."""
    return {
        "role_title": "", "company": "", "seniority": "",
        "required_skills": [], "nice_to_have_skills": [],
        "key_responsibilities": [], "keywords": [],
        "industry": "", "role_type": "", "experience_years": "",
    }
