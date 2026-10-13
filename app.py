"""ResumeForge — Personal ATS Resume Tailoring Tool.

A 100% free, single-user app that takes your master resume + a job
description and produces a tailored resume, cover letter, and LinkedIn
rewrite — without inventing any experience you don't have.

Built with: Streamlit · LangGraph · Groq/Gemini · sentence-transformers · WeasyPrint
"""

import os
import sys
import json
import math

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Synchronize cloud secrets (Streamlit Cloud / HF) to os.environ
try:
    if hasattr(st, "secrets"):
        for _k in ["GROQ_API_KEY", "GOOGLE_API_KEY"]:
            if _k in st.secrets and _k not in os.environ:
                os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

from parsers.resume_parser import (
    extract_text, structure_resume_with_llm, save_master_resume,
    load_master_resume, master_resume_exists, get_empty_resume,
)
from parsers.jd_parser import parse_job_description
from scoring.match_score import compute_match_score
from graph import run_tailor_pipeline

# ══════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ResumeForge — ATS Resume Tailor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# Premium CSS — dark glassmorphism theme with animations
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Base reset ───────────────────────────────────── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(180deg, #050510 0%, #0a0a1f 50%, #0d0d25 100%);
    }

    /* ── Sidebar ──────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #110d2e 100%) !important;
        border-right: 1px solid rgba(99,102,241,0.12);
    }
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 2px;
    }
    section[data-testid="stSidebar"] .stRadio > div > label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 0.55rem 0.85rem !important;
        transition: all 0.25s cubic-bezier(.4,0,.2,1);
        cursor: pointer;
    }
    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(99,102,241,0.08);
        border-color: rgba(99,102,241,0.15);
    }
    section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.12)) !important;
        border-color: rgba(99,102,241,0.3) !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.1);
    }

    /* ── Hero banner ──────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #0c0c20 0%, #1a0e35 35%, #0e1a30 70%, #0c0c20 100%);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(139,92,246,0.12);
        box-shadow: 0 8px 40px rgba(0,0,0,0.5),
                    inset 0 1px 0 rgba(255,255,255,0.03);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -60%;
        right: -15%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 70%);
        border-radius: 50%;
        animation: float-orb 8s ease-in-out infinite;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -10%;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%);
        border-radius: 50%;
        animation: float-orb 10s ease-in-out infinite reverse;
    }
    @keyframes float-orb {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-20px) scale(1.05); }
    }
    .hero-banner h1 {
        background: linear-gradient(135deg, #c4b5fd 0%, #a78bfa 25%, #818cf8 50%, #6366f1 75%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 900;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.8px;
        position: relative;
        z-index: 1;
    }
    .hero-sub {
        color: #a5b4fc;
        font-size: 0.92rem;
        margin: 0;
        opacity: 0.8;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }

    /* ── Feature pills row ────────────────────────────── */
    .feature-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 1rem;
        position: relative;
        z-index: 1;
    }
    .feature-pill {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        color: #c4b5fd;
        padding: 0.3rem 0.7rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 500;
        backdrop-filter: blur(4px);
        transition: all 0.2s ease;
        letter-spacing: 0.01em;
    }
    .feature-pill:hover {
        background: rgba(99,102,241,0.16);
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(99,102,241,0.12);
    }

    /* ── Animated score gauge ─────────────────────────── */
    .score-gauge-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1.5rem 0.5rem;
    }
    .score-gauge {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        margin-bottom: 0.8rem;
    }
    .score-gauge::before {
        content: '';
        position: absolute;
        inset: 6px;
        border-radius: 50%;
        background: #0a0a1a;
    }
    .score-gauge-value {
        position: relative;
        z-index: 1;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -1px;
    }
    .score-gauge-label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }

    /* Score color variants */
    .score-low .score-gauge {
        background: conic-gradient(#ef4444 var(--score-deg), rgba(239,68,68,0.1) var(--score-deg));
        box-shadow: 0 0 30px rgba(239,68,68,0.15);
    }
    .score-low .score-gauge-value { color: #fca5a5; }
    .score-mid .score-gauge {
        background: conic-gradient(#f59e0b var(--score-deg), rgba(245,158,11,0.1) var(--score-deg));
        box-shadow: 0 0 30px rgba(245,158,11,0.15);
    }
    .score-mid .score-gauge-value { color: #fcd34d; }
    .score-high .score-gauge {
        background: conic-gradient(#22c55e var(--score-deg), rgba(34,197,94,0.1) var(--score-deg));
        box-shadow: 0 0 30px rgba(34,197,94,0.15);
    }
    .score-high .score-gauge-value { color: #86efac; }

    /* ── Glass cards ──────────────────────────────────── */
    .glass-card {
        background: linear-gradient(135deg, rgba(12,12,30,0.85), rgba(20,16,45,0.65));
        border: 1px solid rgba(99,102,241,0.1);
        border-radius: 16px;
        padding: 1.4rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
        transition: all 0.3s cubic-bezier(.4,0,.2,1);
    }
    .glass-card:hover {
        border-color: rgba(99,102,241,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transform: translateY(-1px);
    }
    .glass-card h3 {
        background: linear-gradient(90deg, #c4b5fd, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 0.6rem 0;
    }
    .glass-card p {
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.55;
        margin: 0.25rem 0;
    }

    /* ── Metric cards ─────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 14px;
        padding: 1rem 0.8rem;
        text-align: center;
        margin: 0.3rem 0;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(99,102,241,0.1);
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c4b5fd, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #a5b4fc;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-top: 0.15rem;
    }

    /* ── Score breakdown bars ──────────────────────────── */
    .score-bar-wrap {
        margin: 0.6rem 0;
    }
    .score-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }
    .score-bar-track {
        height: 6px;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.8s cubic-bezier(.4,0,.2,1);
    }
    .bar-keyword { background: linear-gradient(90deg, #818cf8, #6366f1); }
    .bar-semantic { background: linear-gradient(90deg, #a78bfa, #7c3aed); }

    /* ── Keyword tags ─────────────────────────────────── */
    .kw-matched {
        display: inline-block;
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.25);
        color: #86efac;
        padding: 0.18rem 0.55rem;
        border-radius: 12px;
        font-size: 0.72rem;
        margin: 0.12rem;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .kw-matched:hover { background: rgba(34,197,94,0.18); transform: scale(1.03); }
    .kw-missing {
        display: inline-block;
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.25);
        color: #fca5a5;
        padding: 0.18rem 0.55rem;
        border-radius: 12px;
        font-size: 0.72rem;
        margin: 0.12rem;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .kw-missing:hover { background: rgba(239,68,68,0.18); transform: scale(1.03); }
    .kw-gap {
        display: inline-block;
        background: rgba(250,204,21,0.1);
        border: 1px solid rgba(250,204,21,0.25);
        color: #fde047;
        padding: 0.18rem 0.55rem;
        border-radius: 12px;
        font-size: 0.72rem;
        margin: 0.12rem;
        font-weight: 500;
    }

    /* ── Diff view ────────────────────────────────────── */
    .diff-before {
        background: rgba(239,68,68,0.06);
        border-left: 3px solid #ef4444;
        padding: 0.6rem 0.9rem;
        border-radius: 0 10px 10px 0;
        margin: 0.25rem 0;
        font-size: 0.85rem;
        color: #fca5a5;
        line-height: 1.5;
    }
    .diff-after {
        background: rgba(34,197,94,0.06);
        border-left: 3px solid #22c55e;
        padding: 0.6rem 0.9rem;
        border-radius: 0 10px 10px 0;
        margin: 0.25rem 0;
        font-size: 0.85rem;
        color: #86efac;
        line-height: 1.5;
    }
    .diff-label {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.25rem;
    }
    .diff-label-before { color: #f87171; }
    .diff-label-after { color: #4ade80; }

    /* ── Change notes ─────────────────────────────────── */
    .change-note {
        background: rgba(99,102,241,0.06);
        border-left: 3px solid #818cf8;
        padding: 0.45rem 0.9rem;
        margin: 0.25rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.8rem;
        color: #c4b5fd;
        line-height: 1.4;
    }

    /* ── Gap cards ─────────────────────────────────────── */
    .gap-card {
        background: rgba(250,204,21,0.04);
        border: 1px solid rgba(250,204,21,0.15);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    .gap-card:hover {
        border-color: rgba(250,204,21,0.25);
        background: rgba(250,204,21,0.06);
    }
    .gap-card strong { color: #fde047; font-size: 0.9rem; }
    .gap-card p { color: #d4d4d8; font-size: 0.85rem; margin: 0.3rem 0 0 0; line-height: 1.45; }

    /* ── Project cards (skill builder) ────────────────── */
    .project-card {
        background: linear-gradient(135deg, rgba(15,15,35,0.85), rgba(25,20,50,0.65));
        border: 1px solid rgba(139,92,246,0.15);
        border-radius: 16px;
        padding: 1.3rem;
        margin: 0.6rem 0;
        transition: all 0.3s ease;
    }
    .project-card:hover {
        border-color: rgba(139,92,246,0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(139,92,246,0.08);
    }
    .project-card h4 {
        color: #c4b5fd;
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .project-card .proj-desc {
        color: #a5b4fc;
        font-size: 0.82rem;
        margin: 0 0 0.6rem 0;
        line-height: 1.4;
    }
    .project-card .proj-meta {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin-bottom: 0.6rem;
    }
    .proj-badge {
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        color: #a5b4fc;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        font-size: 0.68rem;
        font-weight: 500;
    }
    .proj-step {
        color: #94a3b8;
        font-size: 0.8rem;
        padding: 0.2rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .proj-step:last-child { border-bottom: none; }
    .proj-step strong { color: #818cf8; }
    .proj-bullet {
        background: rgba(34,197,94,0.06);
        border: 1px solid rgba(34,197,94,0.15);
        color: #86efac;
        padding: 0.5rem 0.8rem;
        border-radius: 10px;
        font-size: 0.78rem;
        margin-top: 0.6rem;
        line-height: 1.4;
    }

    /* ── Trace steps ──────────────────────────────────── */
    .trace-step {
        background: rgba(15,12,41,0.5);
        border-left: 3px solid #6366f1;
        padding: 0.5rem 0.9rem;
        margin: 0.2rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.8rem;
        color: #cbd5e1;
        line-height: 1.5;
    }

    /* ── Status badges ────────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-ready {
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.25);
        color: #86efac;
    }
    .status-empty {
        background: rgba(250,204,21,0.1);
        border: 1px solid rgba(250,204,21,0.25);
        color: #fde047;
    }
    .status-active {
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.25);
        color: #a5b4fc;
    }

    /* ── Quick action cards ────────────────────────────── */
    .quick-action {
        background: linear-gradient(135deg, rgba(12,12,30,0.9), rgba(20,16,45,0.7));
        border: 1px solid rgba(99,102,241,0.1);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        cursor: default;
        transition: all 0.3s cubic-bezier(.4,0,.2,1);
    }
    .quick-action:hover {
        border-color: rgba(99,102,241,0.25);
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.3);
    }
    .quick-action .qa-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .quick-action .qa-title {
        color: #e2e8f0;
        font-size: 0.92rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .quick-action .qa-desc {
        color: #64748b;
        font-size: 0.75rem;
        line-height: 1.4;
    }

    /* ── History timeline ──────────────────────────────── */
    .history-card {
        background: linear-gradient(135deg, rgba(12,12,30,0.85), rgba(20,16,45,0.65));
        border: 1px solid rgba(99,102,241,0.1);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: all 0.2s ease;
    }
    .history-card:hover {
        border-color: rgba(99,102,241,0.2);
        transform: translateX(3px);
    }
    .history-score {
        min-width: 50px;
        height: 50px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        font-weight: 800;
    }
    .history-score-low { background: rgba(239,68,68,0.12); color: #fca5a5; }
    .history-score-mid { background: rgba(245,158,11,0.12); color: #fcd34d; }
    .history-score-high { background: rgba(34,197,94,0.12); color: #86efac; }
    .history-details { flex: 1; }
    .history-role {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .history-meta {
        color: #64748b;
        font-size: 0.72rem;
        margin-top: 0.15rem;
    }

    /* ── Dashboard stat row ────────────────────────────── */
    .stat-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.8rem;
        margin: 1rem 0;
    }
    .stat-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    .stat-card:hover { transform: translateY(-2px); }
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c4b5fd, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        color: #a5b4fc;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    /* ── Section header ───────────────────────────────── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(99,102,241,0.1);
    }

    /* ── Fade in animation ────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeInUp 0.5s ease-out both; }

    /* ── Hide Streamlit chrome ─────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Button styling ───────────────────────────────── */
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        letter-spacing: -0.01em !important;
    }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 18px rgba(99,102,241,0.35) !important;
    }
    .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 8px 24px rgba(139,92,246,0.5) !important;
        border-color: rgba(255,255,255,0.3) !important;
    }
    .stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]), .stDownloadButton > button:not([kind="primary"]) {
        background: rgba(17, 14, 38, 0.75) !important;
        border: 1px solid rgba(139,92,246,0.25) !important;
        color: #c4b5fd !important;
        backdrop-filter: blur(10px) !important;
    }
    .stButton > button:not([kind="primary"]):hover, .stDownloadButton > button:not([kind="primary"]):hover {
        background: rgba(99,102,241,0.18) !important;
        border-color: rgba(139,92,246,0.5) !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(99,102,241,0.2) !important;
    }

    /* ── Text area & inputs ───────────────────────────── */
    .stTextArea textarea, .stTextInput input {
        background: rgba(12, 10, 28, 0.75) !important;
        border: 1px solid rgba(139,92,246,0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.2s ease !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(139,92,246,0.6) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
        background: rgba(15, 12, 35, 0.9) !important;
    }

    /* ── File uploader ────────────────────────────────── */
    .stFileUploader {
        background: rgba(13, 11, 30, 0.6) !important;
        border: 1px dashed rgba(139,92,246,0.3) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(12px) !important;
        padding: 0.8rem !important;
        transition: all 0.2s ease !important;
    }
    .stFileUploader:hover {
        border-color: rgba(139,92,246,0.6) !important;
        background: rgba(17, 14, 38, 0.75) !important;
    }

    /* ── Expander ──────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(15,12,35,0.7) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(139,92,246,0.15) !important;
        backdrop-filter: blur(10px) !important;
    }

    /* ── Tabs ──────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(139,92,246,0.15);
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(15,12,35,0.6) !important;
        border-radius: 12px 12px 0 0 !important;
        border: 1px solid rgba(139,92,246,0.15) !important;
        border-bottom: none !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15)) !important;
        border-color: rgba(139,92,246,0.4) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* ── Custom Dark Scrollbar ────────────────────────── */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #070614;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(139,92,246,0.25);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(139,92,246,0.5);
    }

    /* ── Mobile responsiveness ────────────────────────── */
    @media (max-width: 768px) {
        .hero-banner {
            padding: 1.4rem 1.2rem !important;
        }
        .hero-banner h1 {
            font-size: 1.6rem !important;
        }
        .hero-sub {
            font-size: 0.82rem !important;
        }
        .score-gauge {
            width: 120px !important;
            height: 120px !important;
        }
        .score-gauge-value {
            font-size: 2.2rem !important;
        }
        .stat-row {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 0.5rem !important;
        }
        .glass-card {
            padding: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# Helper: render score gauge HTML
# ══════════════════════════════════════════════════════════════════

def render_score_gauge(score: int, label: str = "ATS Match Score") -> str:
    """Generate HTML for the animated conic-gradient score gauge."""
    deg = int((score / 100) * 360)
    css_class = "score-low" if score < 40 else ("score-mid" if score < 70 else "score-high")
    return f"""
    <div class="score-gauge-wrap {css_class}" style="--score-deg: {deg}deg;">
        <div class="score-gauge">
            <span class="score-gauge-value">{score}</span>
        </div>
        <div class="score-gauge-label">{label}</div>
    </div>
    """


# ══════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════

defaults = {
    "master_resume": None,
    "jd_text": "",
    "jd_requirements": None,
    "match_score": None,
    "pipeline_result": None,
    "raw_resume_text": "",
    "skill_gap_projects": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Load master resume if exists
if st.session_state.master_resume is None and master_resume_exists():
    st.session_state.master_resume = load_master_resume()


# ══════════════════════════════════════════════════════════════════
# Sidebar — navigation + status
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.8rem 0 0.5rem 0;">
        <span style="font-size:1.8rem;">🔥</span>
        <div style="font-size:1.1rem; font-weight:800;
                    background: linear-gradient(135deg, #c4b5fd, #818cf8);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    margin-top: 0.2rem;">ResumeForge</div>
        <div style="font-size:0.65rem; color:#64748b; margin-top:0.1rem;">Free ATS Resume Tailor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "👤 Master Profile", "🎯 Tailor Resume",
         "🔍 Compare", "🛠️ Skill Builder", "📥 Export", "📊 History"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # API Key status
    groq_key = os.getenv("GROQ_API_KEY", "")
    has_groq = bool(groq_key and groq_key != "your_groq_api_key_here")

    with st.expander("⚙️ API Keys", expanded=not has_groq):
        new_groq = st.text_input(
            "Groq API Key", value=groq_key,
            type="password", help="Free at https://console.groq.com/keys",
        )
        if new_groq:
            os.environ["GROQ_API_KEY"] = new_groq

        new_gemini = st.text_input(
            "Gemini API Key (fallback)", value=os.getenv("GOOGLE_API_KEY", ""),
            type="password", help="Free at https://aistudio.google.com/apikey",
        )
        if new_gemini:
            os.environ["GOOGLE_API_KEY"] = new_gemini

    # Status indicators
    has_resume = st.session_state.master_resume is not None

    if has_groq:
        st.markdown('<span class="status-badge status-ready">🟢 Groq Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-empty">🟡 No API Key</span>', unsafe_allow_html=True)

    if has_resume:
        name = st.session_state.master_resume.get("contact", {}).get("name", "Unknown")
        st.markdown(f'<span class="status-badge status-ready">🟢 {name}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-empty">🟡 No Profile</span>', unsafe_allow_html=True)

    if st.session_state.match_score:
        score = st.session_state.match_score.get("overall_score", 0)
        st.markdown(f"""
        <div class="metric-card" style="margin-top:0.5rem;">
            <div class="metric-value">{score}</div>
            <div class="metric-label">Last ATS Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#475569; font-size:0.65rem; line-height:1.5;">
        Groq · LangGraph · WeasyPrint<br>
        100% free & open source 🔥
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <h1>🔥 ResumeForge</h1>
        <p class="hero-sub">
            Precision resume tailoring — your master resume × any job description.<br>
            Higher ATS scores, zero fabrication, free forever.
        </p>
        <div class="feature-row">
            <span class="feature-pill">🎯 ATS Match Scoring</span>
            <span class="feature-pill">✍️ Truthful Rewrites</span>
            <span class="feature-pill">📝 Cover Letters</span>
            <span class="feature-pill">💼 LinkedIn Rewrite</span>
            <span class="feature-pill">📄 PDF Export</span>
            <span class="feature-pill">🔍 Before/After Diff</span>
            <span class="feature-pill">🛠️ Skill Gap Builder</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    n_exp = 0
    if st.session_state.master_resume:
        n_exp = len(st.session_state.master_resume.get("experience", []))

    last_score = st.session_state.match_score.get("overall_score", "—") if st.session_state.match_score else "—"

    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "history.json")
    n_tailorings = 0
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                n_tailorings = len(json.load(f))
        except Exception:
            pass

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-value">{n_exp}</div>
            <div class="stat-label">Experience Entries</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{last_score}</div>
            <div class="stat-label">Last ATS Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{n_tailorings}</div>
            <div class="stat-label">Total Tailorings</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">$0</div>
            <div class="stat-label">Total Cost</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick actions
    st.markdown('<div class="section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)

    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        st.markdown("""
        <div class="quick-action">
            <div class="qa-icon">👤</div>
            <div class="qa-title">Master Profile</div>
            <div class="qa-desc">Upload or paste your resume to create your source of truth</div>
        </div>
        """, unsafe_allow_html=True)
    with qa2:
        st.markdown("""
        <div class="quick-action">
            <div class="qa-icon">🎯</div>
            <div class="qa-title">Tailor Resume</div>
            <div class="qa-desc">Paste a JD, get an ATS score, and tailor in one click</div>
        </div>
        """, unsafe_allow_html=True)
    with qa3:
        st.markdown("""
        <div class="quick-action">
            <div class="qa-icon">🛠️</div>
            <div class="qa-title">Skill Builder</div>
            <div class="qa-desc">Get weekend project ideas for skills you're missing</div>
        </div>
        """, unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="section-header">🔄 How It Works</div>', unsafe_allow_html=True)

    hw1, hw2, hw3, hw4 = st.columns(4)
    steps = [
        ("1️⃣", "Upload Resume", "Parse your resume into a structured master profile"),
        ("2️⃣", "Paste Job Description", "AI extracts keywords, skills, and requirements"),
        ("3️⃣", "One-Click Tailor", "Resume rewritten for the role — nothing fabricated"),
        ("4️⃣", "Download & Apply", "ATS-safe PDF + cover letter + LinkedIn rewrite"),
    ]
    for col, (icon, title, desc) in zip([hw1, hw2, hw3, hw4], steps):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; min-height:140px;">
                <div style="font-size:1.5rem; margin-bottom:0.4rem;">{icon}</div>
                <h3 style="font-size:0.88rem !important;">{title}</h3>
                <p style="font-size:0.78rem !important; color:#94a3b8;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: Master Profile
# ══════════════════════════════════════════════════════════════════

elif page == "👤 Master Profile":
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem 2rem;">
        <h1 style="font-size:1.6rem;">👤 Master Resume Profile</h1>
        <p class="hero-sub">Upload or paste your resume once. This becomes your source of truth for all tailored resumes.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 📤 Upload Resume")
        uploaded_file = st.file_uploader(
            "Upload PDF, DOCX, or TXT",
            type=["pdf", "docx", "doc", "txt"],
            help="Your resume will be parsed into structured sections",
        )

        if uploaded_file:
            if st.button("📥 Parse & Structure Resume", use_container_width=True, type="primary"):
                with st.spinner("🔍 Extracting text and structuring with AI..."):
                    try:
                        raw_text = extract_text(uploaded_file, uploaded_file.name)
                        st.session_state.raw_resume_text = raw_text
                        structured = structure_resume_with_llm(raw_text)
                        st.session_state.master_resume = structured
                        save_master_resume(structured)
                        st.success("✅ Resume parsed and saved as master profile!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        st.markdown("#### ✏️ Or Paste Resume Text")
        pasted_text = st.text_area(
            "Paste your resume content here",
            height=300,
            placeholder="Paste the full text of your resume...",
        )

        if pasted_text:
            if st.button("📝 Parse Pasted Text", use_container_width=True):
                with st.spinner("🔍 Structuring with AI..."):
                    try:
                        st.session_state.raw_resume_text = pasted_text
                        structured = structure_resume_with_llm(pasted_text)
                        st.session_state.master_resume = structured
                        save_master_resume(structured)
                        st.success("✅ Resume parsed and saved!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    with col2:
        st.markdown("#### 📋 Current Profile Preview")
        if st.session_state.master_resume:
            resume = st.session_state.master_resume
            contact = resume.get("contact", {})

            # Profile header card
            st.markdown(f"""
            <div class="glass-card">
                <h3 style="font-size:1.2rem !important;">{contact.get('name', 'N/A')}</h3>
                <p>📧 {contact.get('email', '')} · 📱 {contact.get('phone', '')} · 📍 {contact.get('location', '')}</p>
                {'<p>🔗 ' + contact.get('linkedin', '') + '</p>' if contact.get('linkedin') else ''}
            </div>
            """, unsafe_allow_html=True)

            if resume.get("summary"):
                st.markdown(f"""
                <div class="glass-card">
                    <h3>Summary</h3>
                    <p>{resume['summary'][:400]}{'...' if len(resume.get('summary','')) > 400 else ''}</p>
                </div>
                """, unsafe_allow_html=True)

            if resume.get("experience"):
                st.markdown(f"**💼 Experience ({len(resume['experience'])} entries)**")
                for exp in resume["experience"]:
                    with st.expander(f"📋 {exp.get('title', 'Role')} @ {exp.get('company', 'Company')} ({exp.get('dates', '')})"):
                        for bullet in exp.get("bullets", []):
                            st.markdown(f"• {bullet}")

            skills = resume.get("skills", {})
            if isinstance(skills, dict) and any(skills.values()):
                st.markdown("**🔧 Skills**")
                all_tags = []
                for cat, items in skills.items():
                    if items:
                        for item in items[:15]:
                            all_tags.append(f'<span class="kw-matched">{item}</span>')
                st.markdown(" ".join(all_tags), unsafe_allow_html=True)

            if resume.get("education"):
                for edu in resume["education"]:
                    st.markdown(f"🎓 {edu.get('degree', '')} — {edu.get('institution', '')} ({edu.get('year', '')})")

            if resume.get("projects"):
                st.markdown(f"**🔧 Projects ({len(resume['projects'])})**")
                for proj in resume["projects"]:
                    st.markdown(f"• **{proj.get('name', '')}** — {proj.get('description', '')[:100]}")
        else:
            st.info("👈 Upload or paste your resume to get started.")

    # Editable JSON view
    if st.session_state.master_resume:
        with st.expander("🔧 Edit Raw JSON (Advanced)", expanded=False):
            edited_json = st.text_area(
                "Edit the structured JSON directly",
                value=json.dumps(st.session_state.master_resume, indent=2),
                height=400,
            )
            if st.button("💾 Save Edited JSON"):
                try:
                    parsed = json.loads(edited_json)
                    st.session_state.master_resume = parsed
                    save_master_resume(parsed)
                    st.success("✅ Saved!")
                    st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")


# ══════════════════════════════════════════════════════════════════
# PAGE: Tailor Resume
# ══════════════════════════════════════════════════════════════════

elif page == "🎯 Tailor Resume":
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem 2rem;">
        <h1 style="font-size:1.6rem;">🎯 Tailor Your Resume</h1>
        <p class="hero-sub">Paste any job description → instant ATS score → one-click tailor. Nothing fabricated.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.master_resume:
        st.warning("⚠️ Set up your Master Profile first (👤 Master Profile page).")
    else:
        # Step 1: Paste JD
        st.markdown("#### Step 1: Paste the Job Description")
        jd_text = st.text_area(
            "Job Description",
            height=220,
            placeholder="Paste the full job description here...",
            value=st.session_state.jd_text,
        )
        st.session_state.jd_text = jd_text

        col_a, col_b = st.columns(2)

        with col_a:
            analyze_btn = st.button("🔍 Analyze JD & Score", use_container_width=True, disabled=not jd_text)
        with col_b:
            tailor_btn = st.button("🚀 Full Tailor Pipeline", use_container_width=True, type="primary", disabled=not jd_text)

        # Analyze JD only
        if analyze_btn and jd_text:
            with st.spinner("🔍 Analyzing job description and computing ATS score..."):
                try:
                    jd_req = parse_job_description(jd_text)
                    st.session_state.jd_requirements = jd_req
                    score = compute_match_score(st.session_state.master_resume, jd_req)
                    st.session_state.match_score = score
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        # Full pipeline
        if tailor_btn and jd_text:
            with st.spinner("🚀 Running full tailoring pipeline (30-60 seconds)..."):
                try:
                    result = run_tailor_pipeline(st.session_state.master_resume, jd_text)
                    st.session_state.pipeline_result = result
                    st.session_state.jd_requirements = result.get("jd_requirements", {})
                    st.session_state.match_score = result.get("match_score", {})
                    st.success("✅ Pipeline complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Pipeline error: {e}")

        # ── Show JD Analysis ──────────────────────────────────
        if st.session_state.jd_requirements:
            jd = st.session_state.jd_requirements

            st.markdown("---")
            st.markdown("#### 📋 JD Analysis")

            jd_col1, jd_col2, jd_col3 = st.columns(3)

            with jd_col1:
                st.markdown(f"""
                <div class="glass-card">
                    <h3>🎯 Role Details</h3>
                    <p><strong>Title:</strong> {jd.get('role_title', 'N/A')}</p>
                    <p><strong>Company:</strong> {jd.get('company', 'N/A')}</p>
                    <p><strong>Seniority:</strong> {jd.get('seniority', 'N/A')}</p>
                    <p><strong>Industry:</strong> {jd.get('industry', 'N/A')}</p>
                    <p><strong>Experience:</strong> {jd.get('experience_years', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

            with jd_col2:
                req_skills = jd.get("required_skills", [])
                nice_skills = jd.get("nice_to_have_skills", [])
                skills_html = " ".join([f'<span class="kw-missing">{s}</span>' for s in req_skills])
                nice_html = " ".join([f'<span class="kw-gap">{s}</span>' for s in nice_skills])
                st.markdown(f"""
                <div class="glass-card">
                    <h3>🔧 Required Skills ({len(req_skills)})</h3>
                    <p>{skills_html}</p>
                    {f'<h3 style="margin-top:0.8rem;">✨ Nice-to-Have ({len(nice_skills)})</h3><p>{nice_html}</p>' if nice_skills else ''}
                </div>
                """, unsafe_allow_html=True)

            with jd_col3:
                resp = jd.get("key_responsibilities", [])
                resp_html = "".join([f"<p>• {r}</p>" for r in resp[:6]])
                st.markdown(f"""
                <div class="glass-card">
                    <h3>📝 Responsibilities ({len(resp)})</h3>
                    {resp_html}
                </div>
                """, unsafe_allow_html=True)

        # ── Show ATS Score ────────────────────────────────────
        if st.session_state.match_score:
            score_data = st.session_state.match_score

            st.markdown("---")
            st.markdown("#### 📊 ATS Match Score")

            sc_col1, sc_col2 = st.columns([1, 2])

            with sc_col1:
                overall = score_data.get("overall_score", 0)
                st.markdown(render_score_gauge(overall), unsafe_allow_html=True)

                # Score breakdown bars
                kw_score = score_data.get("keyword_score", 0)
                sem_score = score_data.get("semantic_score", 0)
                st.markdown(f"""
                <div class="score-bar-wrap">
                    <div class="score-bar-label">
                        <span>Keyword Match</span>
                        <span style="color:#c4b5fd; font-weight:600;">{kw_score}%</span>
                    </div>
                    <div class="score-bar-track">
                        <div class="score-bar-fill bar-keyword" style="width:{kw_score}%"></div>
                    </div>
                </div>
                <div class="score-bar-wrap">
                    <div class="score-bar-label">
                        <span>Semantic Match</span>
                        <span style="color:#c4b5fd; font-weight:600;">{sem_score}%</span>
                    </div>
                    <div class="score-bar-track">
                        <div class="score-bar-fill bar-semantic" style="width:{sem_score}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with sc_col2:
                kw_sub1, kw_sub2 = st.columns(2)
                with kw_sub1:
                    st.markdown("**✅ Matched Keywords:**")
                    matched_html = " ".join([f'<span class="kw-matched">{kw}</span>' for kw in score_data.get("matched_keywords", [])[:25]])
                    st.markdown(matched_html or "*None found*", unsafe_allow_html=True)
                with kw_sub2:
                    st.markdown("**❌ Missing Keywords:**")
                    missing_html = " ".join([f'<span class="kw-missing">{kw}</span>' for kw in score_data.get("missing_keywords", [])[:25]])
                    st.markdown(missing_html or "*All present! 🎉*", unsafe_allow_html=True)

            # Weak bullets
            weak = score_data.get("weak_bullets", [])
            if weak:
                with st.expander(f"⚠️ Weak/Generic Bullets ({len(weak)}) — need strengthening"):
                    for b in weak:
                        st.markdown(f'<div class="diff-before">• {b}</div>', unsafe_allow_html=True)

        # ── Show pipeline results ─────────────────────────────
        if st.session_state.pipeline_result:
            result = st.session_state.pipeline_result

            # Processing trace
            trace = result.get("processing_trace", [])
            if trace:
                with st.expander("🔄 Pipeline Trace", expanded=False):
                    for step in trace:
                        st.markdown(f'<div class="trace-step">{step}</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Tabbed output
            out_tab1, out_tab2, out_tab3 = st.tabs(["✍️ Tailored Resume", "📝 Cover Letter", "💼 LinkedIn"])

            with out_tab1:
                tailored = result.get("tailored_resume", {})
                if tailored:
                    st.markdown("#### ✍️ Tailored Resume Preview")

                    if tailored.get("summary"):
                        st.markdown(f"""
                        <div class="glass-card">
                            <h3>Professional Summary</h3>
                            <p>{tailored['summary']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    for exp in tailored.get("experience", []):
                        st.markdown(f"**{exp.get('title', '')}** — {exp.get('company', '')} ({exp.get('dates', '')})")
                        for bullet in exp.get("bullets", []):
                            st.markdown(f"• {bullet}")

                        changes = exp.get("changes", [])
                        if changes:
                            for c in changes:
                                st.markdown(f'<div class="change-note">💡 {c}</div>', unsafe_allow_html=True)
                        st.markdown("")

                    # Gaps
                    gaps = result.get("gaps", [])
                    if gaps:
                        st.markdown("#### ⚠️ Experience Gaps")
                        st.markdown("*These JD requirements have no matching experience in your resume:*")
                        for gap in gaps:
                            st.markdown(f"""
                            <div class="gap-card">
                                <strong>🔸 {gap.get('requirement', '')}</strong>
                                <span class="kw-gap">{gap.get('category', '')}</span>
                                <p>💡 {gap.get('suggestion', 'Consider addressing this gap')}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Changes summary
                    changes_summary = result.get("changes_summary", [])
                    if changes_summary:
                        with st.expander("📋 All Changes Made"):
                            for i, change in enumerate(changes_summary, 1):
                                st.markdown(f"**{i}.** {change}")

            with out_tab2:
                cl = result.get("cover_letter", {})
                if cl and cl.get("cover_letter"):
                    st.markdown("#### 📝 Generated Cover Letter")

                    st.markdown(f"""
                    <div class="glass-card">
                        <p>{cl['cover_letter'].replace(chr(10), '<br>')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if cl.get("key_points"):
                        st.markdown("**Key Points Highlighted:**")
                        for kp in cl["key_points"]:
                            st.markdown(f"• {kp}")

                    col_cl1, col_cl2 = st.columns(2)
                    with col_cl1:
                        st.download_button(
                            "⬇️ Download as TXT",
                            data=cl["cover_letter"],
                            file_name="cover_letter.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )
                else:
                    st.info("Cover letter will appear here after running the full pipeline.")

            with out_tab3:
                li = result.get("linkedin", {})
                if li and (li.get("headline") or li.get("about")):
                    st.markdown("#### 💼 LinkedIn Headline")
                    st.code(li.get("headline", ""), language=None)

                    st.markdown("#### 💼 LinkedIn About Section")
                    st.markdown(f"""
                    <div class="glass-card">
                        <p>{li.get('about', '').replace(chr(10), '<br>')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if li.get("keywords_used"):
                        st.markdown("**Keywords Incorporated:**")
                        kw_html = " ".join([f'<span class="kw-matched">{k}</span>' for k in li["keywords_used"]])
                        st.markdown(kw_html, unsafe_allow_html=True)
                else:
                    st.info("LinkedIn rewrite will appear here after running the full pipeline.")


# ══════════════════════════════════════════════════════════════════
# PAGE: Compare (Before/After Diff)
# ══════════════════════════════════════════════════════════════════

elif page == "🔍 Compare":
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem 2rem;">
        <h1 style="font-size:1.6rem;">🔍 Before → After Comparison</h1>
        <p class="hero-sub">See exactly what changed and why — original on the left, tailored result on the right.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_result or not st.session_state.master_resume:
        st.info("Run the full tailoring pipeline first (🎯 Tailor Resume) to see the comparison.")
    else:
        original = st.session_state.master_resume
        result = st.session_state.pipeline_result
        tailored = result.get("tailored_resume", {})

        # Score impact summary
        if st.session_state.match_score:
            sc = st.session_state.match_score
            n_changes = len(result.get("changes_summary", []))
            n_gaps = len(result.get("gaps", []))
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-value">{sc.get('overall_score', 0)}</div>
                    <div class="stat-label">ATS Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{n_changes}</div>
                    <div class="stat-label">Changes Made</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{sc.get('total_matched', 0)}</div>
                    <div class="stat-label">Keywords Matched</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{n_gaps}</div>
                    <div class="stat-label">Gaps Found</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Summary comparison
        if original.get("summary") or tailored.get("summary"):
            st.markdown("#### Professional Summary")
            col_b, col_a = st.columns(2)
            with col_b:
                st.markdown('<div class="diff-label diff-label-before">BEFORE — Original</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="diff-before">{original.get("summary", "N/A")}</div>', unsafe_allow_html=True)
            with col_a:
                st.markdown('<div class="diff-label diff-label-after">AFTER — Tailored</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="diff-after">{tailored.get("summary", "N/A")}</div>', unsafe_allow_html=True)

        # Experience bullet-by-bullet comparison
        st.markdown("---")
        st.markdown("#### Experience Bullets")

        orig_exps = original.get("experience", [])
        tail_exps = tailored.get("experience", [])

        for i in range(max(len(orig_exps), len(tail_exps))):
            orig_exp = orig_exps[i] if i < len(orig_exps) else {}
            tail_exp = tail_exps[i] if i < len(tail_exps) else {}

            title = tail_exp.get("title", orig_exp.get("title", ""))
            company = tail_exp.get("company", orig_exp.get("company", ""))

            st.markdown(f"##### 📋 {title} @ {company}")

            orig_bullets = orig_exp.get("bullets", [])
            tail_bullets = tail_exp.get("bullets", [])

            col_before, col_after = st.columns(2)

            with col_before:
                st.markdown('<div class="diff-label diff-label-before">BEFORE</div>', unsafe_allow_html=True)
                for bullet in orig_bullets:
                    st.markdown(f'<div class="diff-before">• {bullet}</div>', unsafe_allow_html=True)

            with col_after:
                st.markdown('<div class="diff-label diff-label-after">AFTER</div>', unsafe_allow_html=True)
                for bullet in tail_bullets:
                    st.markdown(f'<div class="diff-after">• {bullet}</div>', unsafe_allow_html=True)

            # Change annotations
            changes = tail_exp.get("changes", [])
            if changes:
                for c in changes:
                    st.markdown(f'<div class="change-note">💡 {c}</div>', unsafe_allow_html=True)
            st.markdown("")

        # Keywords comparison
        if st.session_state.match_score:
            st.markdown("---")
            st.markdown("#### 🏷️ Keyword Impact")

            kw_col1, kw_col2 = st.columns(2)
            with kw_col1:
                st.markdown("**✅ Matched (already in resume):**")
                matched_html = " ".join([f'<span class="kw-matched">✓ {kw}</span>' for kw in st.session_state.match_score.get("matched_keywords", [])[:25]])
                st.markdown(matched_html or "None", unsafe_allow_html=True)
            with kw_col2:
                st.markdown("**🔸 Gaps Flagged (not fabricated):**")
                gaps = result.get("gaps", [])
                if gaps:
                    for gap in gaps:
                        st.markdown(f'<span class="kw-gap">⚠ {gap.get("requirement", "")}</span>', unsafe_allow_html=True)
                else:
                    st.markdown("No gaps — your resume covers all requirements! 🎉")


# ══════════════════════════════════════════════════════════════════
# PAGE: Skill Builder (NEW)
# ══════════════════════════════════════════════════════════════════

elif page == "🛠️ Skill Builder":
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem 2rem;">
        <h1 style="font-size:1.6rem;">🛠️ Skill Gap Builder</h1>
        <p class="hero-sub">Missing a skill the JD asks for? Get a weekend project idea to genuinely learn it and add it to your resume.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_result:
        st.info("Run the full tailoring pipeline first (🎯 Tailor Resume) to identify skill gaps.")
    else:
        result = st.session_state.pipeline_result
        gaps = result.get("gaps", [])

        if not gaps:
            st.success("🎉 No skill gaps found! Your resume covers all the JD requirements.")
        else:
            st.markdown(f"**{len(gaps)} skill gaps** identified from your last tailoring run:")

            for gap in gaps:
                st.markdown(f"""
                <div class="gap-card">
                    <strong>🔸 {gap.get('requirement', '')}</strong>
                    <span class="kw-gap">{gap.get('category', '')}</span>
                    <p>💡 {gap.get('suggestion', '')}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            if st.button("🛠️ Generate Weekend Project Ideas", use_container_width=True, type="primary"):
                with st.spinner("🛠️ Generating project ideas for your skill gaps..."):
                    try:
                        from generation.skill_gap_projects import generate_skill_gap_projects
                        jd_req = st.session_state.jd_requirements or {}
                        projects = generate_skill_gap_projects(gaps, jd_req)
                        st.session_state.skill_gap_projects = projects
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            # Show generated projects
            if st.session_state.skill_gap_projects:
                projects = st.session_state.skill_gap_projects.get("projects", [])
                if projects:
                    st.markdown(f'<div class="section-header">🚀 Weekend Project Ideas ({len(projects)})</div>', unsafe_allow_html=True)

                    for proj in projects:
                        steps_html = ""
                        for j, step in enumerate(proj.get("steps", []), 1):
                            steps_html += f'<div class="proj-step"><strong>{j}.</strong> {step}</div>'

                        tech_html = " ".join([f'<span class="proj-badge">{t}</span>' for t in proj.get("technologies", [])])

                        diff_class = {
                            "beginner": "kw-matched",
                            "intermediate": "kw-gap",
                            "advanced": "kw-missing",
                        }.get(proj.get("difficulty", "").lower(), "kw-gap")

                        st.markdown(f"""
                        <div class="project-card">
                            <h4>🔧 {proj.get('project_name', 'Project')}</h4>
                            <div class="proj-desc">{proj.get('description', '')}</div>
                            <div class="proj-meta">
                                <span class="proj-badge">🎯 {proj.get('skill', '')}</span>
                                <span class="proj-badge">⏱️ {proj.get('time_estimate', 'A weekend')}</span>
                                <span class="{diff_class}">{proj.get('difficulty', 'intermediate')}</span>
                            </div>
                            <div style="margin: 0.5rem 0;">{tech_html}</div>
                            {steps_html}
                            <div class="proj-bullet">📝 Resume bullet: {proj.get('resume_bullet', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: Export
# ══════════════════════════════════════════════════════════════════

elif page == "📥 Export":
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem 2rem;">
        <h1 style="font-size:1.6rem;">📥 Export Documents</h1>
        <p class="hero-sub">Download your tailored resume, cover letter, and LinkedIn content.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.pipeline_result:
        st.info("Run the full tailoring pipeline first (🎯 Tailor Resume) to export documents.")
    else:
        result = st.session_state.pipeline_result
        tailored = result.get("tailored_resume", {})

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("""
            <div class="glass-card">
                <h3>📄 Resume Export</h3>
                <p>ATS-safe formats: single column, standard typography, clean semantic headings.</p>
            </div>
            """, unsafe_allow_html=True)

            name = tailored.get("contact", {}).get("name", "resume").replace(" ", "_")
            role = result.get("jd_requirements", {}).get("role_title", "tailored").replace(" ", "_")

            # DOCX export (100% reliable, zero native C dependencies)
            try:
                from export.docx_export import export_resume_to_docx
                docx_bytes = export_resume_to_docx(tailored)
                st.download_button(
                    label="📝 Download ATS Word Document (.DOCX)",
                    data=docx_bytes,
                    file_name=f"{name}_{role}_resume.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    type="primary",
                )
            except Exception as e:
                st.error(f"DOCX export error: {e}")

            # HTML export (Open in browser, save/print to PDF via Cmd+P)
            try:
                from export.pdf_export import render_resume_html
                html_resume = render_resume_html(tailored)
                st.download_button(
                    label="🌐 Download Web Resume (.HTML)",
                    data=html_resume,
                    file_name=f"{name}_{role}_resume.html",
                    mime="text/html",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"HTML render error: {e}")

            # WeasyPrint PDF direct download
            if st.button("📄 Generate Direct PDF", use_container_width=True):
                with st.spinner("Generating PDF via WeasyPrint..."):
                    try:
                        from export.pdf_export import export_resume_to_pdf
                        pdf_bytes = export_resume_to_pdf(tailored)
                        filename = f"{name}_{role}_resume.pdf"

                        st.download_button(
                            label="⬇️ Download Direct PDF",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.success("✅ PDF generated!")
                    except ImportError:
                        st.error("❌ WeasyPrint not installed. Run: `pip install weasyprint`")
                    except Exception as e:
                        st.warning(f"⚠️ WeasyPrint note: {e}. You can also download the DOCX or HTML above (open HTML in Chrome/Safari and choose Print → Save as PDF)!")

        with exp_col2:
            # Cover letter export
            cl = result.get("cover_letter", {})
            st.markdown("""
            <div class="glass-card">
                <h3>📝 Cover Letter</h3>
                <p>Tailored cover letter grounded in your actual experience.</p>
            </div>
            """, unsafe_allow_html=True)

            if cl and cl.get("cover_letter"):
                st.download_button(
                    label="⬇️ Download Cover Letter (TXT)",
                    data=cl["cover_letter"],
                    file_name="cover_letter.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

                if st.button("📄 Generate Cover Letter PDF"):
                    with st.spinner("Generating PDF..."):
                        try:
                            from export.pdf_export import export_cover_letter_to_pdf
                            jd_req = result.get("jd_requirements", {})
                            pdf_bytes = export_cover_letter_to_pdf(
                                cl["cover_letter"],
                                role_title=jd_req.get("role_title", ""),
                                company=jd_req.get("company", ""),
                            )
                            st.download_button(
                                label="⬇️ Download Cover Letter PDF",
                                data=pdf_bytes,
                                file_name="cover_letter.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

        st.markdown("---")

        # LinkedIn export
        li = result.get("linkedin", {})
        if li and (li.get("headline") or li.get("about")):
            st.markdown("""
            <div class="glass-card">
                <h3>💼 LinkedIn Content</h3>
                <p>Copy your optimized headline and about section below.</p>
            </div>
            """, unsafe_allow_html=True)

            li_col1, li_col2 = st.columns(2)
            with li_col1:
                st.markdown("**Headline:**")
                st.code(li.get("headline", ""), language=None)
            with li_col2:
                st.download_button(
                    label="⬇️ Download LinkedIn Content (TXT)",
                    data=f"HEADLINE:\n{li.get('headline', '')}\n\nABOUT:\n{li.get('about', '')}",
                    file_name="linkedin_content.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            st.markdown("**About Section:**")
            st.code(li.get("about", ""), language=None)

        st.markdown("---")

        # JSON export
        st.markdown("""
        <div class="glass-card">
            <h3>🔧 Raw JSON Export</h3>
            <p>Download the full structured JSON for external processing.</p>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Tailored Resume JSON",
            data=json.dumps(tailored, indent=2),
            file_name="tailored_resume.json",
            mime="application/json",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════
# PAGE: History
# ══════════════════════════════════════════════════════════════════

elif page == "📊 History":
    st.markdown("""
    <div class="hero-banner" style="padding:1.5rem 2rem;">
        <h1 style="font-size:1.6rem;">📊 Tailoring History</h1>
        <p class="hero-sub">Track every job you've tailored for — scores, gaps, and changes over time.</p>
    </div>
    """, unsafe_allow_html=True)

    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "history.json")

    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)

            if history:
                # Stats row
                scores = [e.get("score_before", 0) for e in history if e.get("score_before")]
                avg_score = int(sum(scores) / len(scores)) if scores else 0
                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-card">
                        <div class="stat-value">{len(history)}</div>
                        <div class="stat-label">Total Runs</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{avg_score}</div>
                        <div class="stat-label">Avg ATS Score</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{max(scores) if scores else 0}</div>
                        <div class="stat-label">Best Score</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f'<div class="section-header">📜 All Runs ({len(history)})</div>', unsafe_allow_html=True)

                for entry in reversed(history):
                    score = entry.get("score_before", 0)
                    score_class = "history-score-low" if score < 40 else ("history-score-mid" if score < 70 else "history-score-high")
                    st.markdown(f"""
                    <div class="history-card">
                        <div class="history-score {score_class}">{score}</div>
                        <div class="history-details">
                            <div class="history-role">🎯 {entry.get('role_title', 'Unknown Role')} @ {entry.get('company', 'Unknown')}</div>
                            <div class="history-meta">📅 {entry.get('timestamp', 'N/A')[:16]} · {entry.get('gaps_count', 0)} gaps · {entry.get('changes_count', 0)} changes</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                if st.button("🗑️ Clear History"):
                    os.remove(history_path)
                    st.success("History cleared.")
                    st.rerun()
            else:
                st.info("No tailoring history yet.")
        except Exception:
            st.info("No tailoring history yet.")
    else:
        st.info("No tailoring history yet. Run your first tailoring pipeline to start tracking.")
