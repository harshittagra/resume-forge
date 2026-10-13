"""Central prompt store — every LLM prompt with the no-fabrication constraint.

╔══════════════════════════════════════════════════════════════════════╗
║  CRITICAL: The NO_FABRICATION_CONSTRAINT is injected into EVERY     ║
║  generation prompt. This keeps all output defensible in interviews. ║
╚══════════════════════════════════════════════════════════════════════╝
"""

NO_FABRICATION_CONSTRAINT = """
CRITICAL CONSTRAINT — YOU MUST FOLLOW THIS EXACTLY:
You may only rephrase, reorder, or re-emphasize experience that already exists
in the provided master resume. Do not invent metrics, tools, technologies,
responsibilities, or outcomes that are not explicitly present in the source.
If the job description requires something not present in the source resume,
list it separately as a "gap" rather than fabricating it into a bullet.

NEVER:
- Add a metric (percentage, dollar amount, count) that isn't in the original
- Mention a tool, framework, or technology not in the original
- Claim a responsibility or outcome not present in the original
- Inflate scope, team size, or impact beyond what's stated

ALWAYS:
- Use only facts from the master resume
- Incorporate exact JD terminology ONLY where it truthfully matches existing experience
- Flag any JD requirement with no matching resume experience as a "gap"
""".strip()


RESUME_STRUCTURE_PROMPT = """You are a resume parser. Convert the following raw resume text into a structured JSON format.

Extract the following sections. If a section is not found, use empty strings/arrays.

Return ONLY valid JSON with this exact structure (no markdown, no code fences, no explanation):
{{
  "contact": {{
    "name": "",
    "email": "",
    "phone": "",
    "linkedin": "",
    "location": "",
    "website": ""
  }},
  "summary": "",
  "experience": [
    {{
      "company": "",
      "title": "",
      "dates": "",
      "location": "",
      "bullets": ["bullet point 1", "bullet point 2"]
    }}
  ],
  "skills": {{
    "technical": ["skill1", "skill2"],
    "tools": ["tool1", "tool2"],
    "soft": ["skill1"],
    "certifications": ["cert1"]
  }},
  "education": [
    {{
      "degree": "",
      "institution": "",
      "year": "",
      "gpa": "",
      "details": ""
    }}
  ],
  "projects": [
    {{
      "name": "",
      "description": "",
      "technologies": ["tech1"],
      "link": ""
    }}
  ]
}}

RULES:
- Extract EVERY experience entry with ALL bullet points — do not summarize or skip any
- Keep bullets exactly as written — do not rephrase
- For skills, categorize into technical (languages, frameworks), tools (software), soft (interpersonal), certifications
- If dates are partial, include what's available
- If a section doesn't exist, use empty string or empty array

RAW RESUME TEXT:
{resume_text}
"""


JD_PARSE_PROMPT = """You are a job description analyzer. Parse the following job description and extract structured requirements.

Return ONLY valid JSON with this exact structure (no markdown, no code fences, no explanation):
{{
  "role_title": "",
  "company": "",
  "seniority": "",
  "required_skills": ["skill1", "skill2"],
  "nice_to_have_skills": ["skill1"],
  "key_responsibilities": ["responsibility1", "responsibility2"],
  "keywords": ["keyword1", "keyword2"],
  "industry": "",
  "role_type": "",
  "experience_years": ""
}}

RULES:
- "required_skills": ALL explicitly required technical skills, tools, frameworks, languages, methodologies
- "nice_to_have_skills": Preferred/bonus/nice-to-have skills
- "key_responsibilities": Main duties and responsibilities (keep concise)
- "keywords": ATS-critical keywords — specific tool names, certifications, methodologies, buzzwords
- "seniority": Infer level (Entry/Junior/Mid/Senior/Staff/Lead/Principal/Director/VP)
- "industry": Domain (e.g., "FinTech", "Healthcare", "SaaS")
- "role_type": "Full-time", "Part-time", "Contract", "Internship"
- "experience_years": Required years if mentioned

JOB DESCRIPTION:
{jd_text}
"""


TAILOR_RESUME_PROMPT = """You are an expert resume tailoring assistant.

{no_fabrication_constraint}

MASTER RESUME (source of truth — you may ONLY use facts from this):
{master_resume}

TARGET JOB DESCRIPTION REQUIREMENTS:
{jd_requirements}

MATCH ANALYSIS:
- Current ATS Score: {match_score}/100
- Missing Keywords: {missing_keywords}
- Weak Bullets: {weak_bullets}

YOUR TASK:
For each experience entry, rewrite the bullets to:
1. REORDER bullets so the most JD-relevant ones come first
2. REPHRASE to incorporate exact JD terminology WHERE IT TRUTHFULLY MATCHES what was actually done
3. RE-EMPHASIZE outcomes and metrics that align with JD priorities
4. STRENGTHEN weak/generic bullets by making them more specific (using ONLY existing facts)
5. Rewrite the summary/professional summary to target this specific role

Also rewrite the skills section to:
- Reorder skills to prioritize JD-relevant ones first
- Use exact JD terminology for skills that match

Return ONLY valid JSON (no markdown, no code fences):
{{
  "tailored_resume": {{
    "contact": <keep contact unchanged from master resume>,
    "summary": "rewritten summary targeting the role",
    "experience": [
      {{
        "company": "",
        "title": "",
        "dates": "",
        "location": "",
        "bullets": ["rewritten bullet 1", "rewritten bullet 2"],
        "changes": ["what changed and why for this entry"]
      }}
    ],
    "skills": {{
      "technical": [],
      "tools": [],
      "soft": [],
      "certifications": []
    }},
    "education": <keep unchanged>,
    "projects": <keep unchanged>
  }},
  "gaps": [
    {{
      "requirement": "JD requirement with no matching experience",
      "category": "required or nice_to_have",
      "suggestion": "how to address this gap"
    }}
  ],
  "changes_summary": [
    "concise description of each significant change made and why"
  ]
}}
"""


COVER_LETTER_PROMPT = """You are an expert cover letter writer.

{no_fabrication_constraint}

TAILORED RESUME (use ONLY these facts):
{tailored_resume}

TARGET JOB REQUIREMENTS:
{jd_requirements}

INSTRUCTIONS:
1. Write a 3-4 paragraph cover letter (250-350 words)
2. Opening: Express genuine interest in the specific role and company
3. Body: Connect 2-3 of the strongest resume achievements to specific JD requirements
4. Use exact JD terminology where it matches actual experience
5. Closing: Express enthusiasm and availability
6. Tone: Professional, confident, specific — not generic

Return ONLY valid JSON (no markdown, no code fences):
{{
  "cover_letter": "the full cover letter text",
  "key_points": ["point 1 highlighted", "point 2 highlighted"],
  "tone": "professional/conversational/formal"
}}
"""


LINKEDIN_REWRITE_PROMPT = """You are a LinkedIn profile optimization expert.

{no_fabrication_constraint}

TAILORED RESUME (use ONLY these facts):
{tailored_resume}

TARGET ROLE DIRECTION:
{jd_requirements}

INSTRUCTIONS:
1. HEADLINE (max 220 characters):
   - Lead with target role title or closest truthful equivalent
   - Include 2-3 key differentiators from actual experience
   - Use vertical bars (|) or bullet dots (·) as separators

2. ABOUT SECTION (150-250 words):
   - Open with a strong positioning statement
   - Highlight 3-4 key achievements that align with target direction
   - Include relevant technical skills and domain expertise
   - End with what you're looking for
   - Write in first person, short paragraphs

Return ONLY valid JSON (no markdown, no code fences):
{{
  "headline": "the LinkedIn headline",
  "about": "the About section text",
  "keywords_used": ["keyword1", "keyword2"]
}}
"""


SKILL_GAP_PROJECTS_PROMPT = """You are a technical career coach and project mentor.

A candidate is applying for the role of **{role_title}** at **{company}** in the **{industry}** industry.

Their resume is missing these key skills/requirements:
{gaps}

For each missing skill, suggest a practical weekend project (1-2 days) that would:
1. Genuinely teach the skill through hands-on building
2. Produce a portfolio-worthy artifact (GitHub repo, deployed app, etc.)
3. Be completable in a weekend by a motivated developer
4. Result in something that could be legitimately added to a resume

Return ONLY valid JSON (no markdown, no code fences):
{{
  "projects": [
    {{
      "skill": "the missing skill",
      "project_name": "catchy project name",
      "description": "1-2 sentence overview",
      "time_estimate": "e.g. 1 weekend, 2 days",
      "steps": ["step 1", "step 2", "step 3", "step 4", "step 5"],
      "technologies": ["tech1", "tech2"],
      "resume_bullet": "suggested resume bullet after completing this project",
      "difficulty": "beginner/intermediate/advanced"
    }}
  ]
}}
"""
