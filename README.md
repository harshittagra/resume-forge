---
title: ResumeForge ATS Resume Tailor
emoji: 🔥
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
license: mit
---

# 🚀 ResumeForge — Personal ATS Resume Tailoring Tool

**ResumeForge** is a 100% free, privacy-focused, single-user web application designed to tailor your resume for any job description using AI — **without inventing any experience, skills, tools, or metrics you don't actually have.**

---

## ✨ Features

1. **Master Resume Parser**: Upload your resume in PDF, DOCX, or plain text. ResumeForge parses and structures it into a single source of truth JSON (`data/master_resume.json`).
2. **Job Description Analyzer**: Extracts key skills, required experience, ATS keywords, and seniority signals from any pasted job description.
3. **Offline ATS Match Scorer (0–100)**: Evaluates match quality using:
   - **40% Keyword Overlap**: Exact ATS keyword matching.
   - **60% Semantic Similarity**: Offline `sentence-transformers` vector similarity (`all-MiniLM-L6-v2`) — **$0 LLM cost**.
4. **AI Resume Bullet Tailor**: Re-orders, aligns, and highlights your actual experience to target job requirements under a strict **No Fabrication Constraint**.
5. **Cover Letter Generator**: Generates a tailored cover letter grounded strictly in your verified resume experience.
6. **LinkedIn Optimization**: Rewrites your LinkedIn headline and About section tailored to target roles.
7. **Interactive Diff Viewer**: Compare original vs. tailored resume side-by-side with visual diffs and change rationale.
8. **ATS-Safe PDF Export**: Downloads clean single-column PDF resumes optimized for ATS parsers (via Jinja2 & WeasyPrint).
9. **Tailoring History**: Track all your job applications and past tailored resumes locally in `data/history.json`.

---

## 🛠️ Tech Stack & Requirements

- **Frontend / UI**: [Streamlit](https://streamlit.io/) with custom dark theme CSS.
- **LLM Engine**: Dual provider with automatic fallback:
  - Primary: **Groq** (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) — Free Tier API Key
  - Fallback: **Google Gemini** (`gemini-1.5-flash`) — Free Tier API Key
- **Vector Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`) for local, zero-cost scoring.
- **Orchestration**: `LangGraph` pipeline graph.
- **PDF Generation**: `WeasyPrint` + Jinja2 single-column ATS templates.

---

## 🚀 Quickstart Guide

### 1. Install Dependencies

```bash
cd /Users/harshittagra/resume-forge
pip install -r requirements.txt
```

### 2. Configure API Keys (100% Free Tiers)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` with your free API keys:

```env
# Free Groq key: https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here

# Free Gemini key: https://aistudio.google.com/apikey
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Launch the Web Application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Directory Structure

```
resume-forge/
├── app.py                   # Main Streamlit 5-page application
├── graph.py                 # LangGraph orchestration graph
├── llm_provider.py          # Dual LLM provider with auto-fallback
├── requirements.txt         # Python dependencies
├── .env.example             # API key template
├── parsers/
│   ├── resume_parser.py     # PDF/DOCX/Text → Structured JSON
│   └── jd_parser.py         # JD text → Key requirements JSON
├── scoring/
│   └── match_score.py       # Offline 0–100 ATS Match Scorer
├── generation/
│   ├── prompts.py           # LLM prompts with No-Fabrication Rules
│   ├── tailor_resume.py     # Bullet point rewriter & aligner
│   ├── cover_letter.py      # Resume-grounded cover letter generator
│   └── linkedin_rewrite.py  # LinkedIn headline & about section generator
├── export/
│   ├── pdf_export.py        # HTML → WeasyPrint PDF renderer
│   ├── template.html        # ATS-safe HTML single-col layout
│   └── template.css         # Print CSS styles
└── data/                    # Local storage (Master resume & history)
```

---

## 🔒 Security & Privacy

All user data, resumes, history, and master files are saved locally on your machine in the `data/` folder. No third-party servers store your information.
