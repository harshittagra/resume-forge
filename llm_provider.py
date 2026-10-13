"""LLM Provider — Groq primary + Google Gemini fallback.

Both are free-tier, API key only, no credit card required.
"""

import os
import logging
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.1) -> BaseChatModel:
    """Get LLM instance — tries Groq first, falls back to Gemini."""
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and groq_key != "your_groq_api_key_here":
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=temperature,
            )
            logger.info("Using Groq (Llama 3.3 70B)")
            return llm
        except Exception as e:
            logger.warning(f"Groq init failed: {e}. Trying Gemini...")

    gemini_key = os.getenv("GOOGLE_API_KEY", "")
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=gemini_key,
                temperature=temperature,
            )
            logger.info("Using Google Gemini (2.0 Flash)")
            return llm
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}")

    raise ValueError(
        "No LLM provider available. Set at least one API key:\n"
        "  • GROQ_API_KEY (free at https://console.groq.com/keys)\n"
        "  • GOOGLE_API_KEY (free at https://aistudio.google.com/apikey)"
    )


def invoke_llm(prompt: str, temperature: float = 0.1) -> str:
    """Invoke LLM with automatic provider fallback.

    If the primary provider fails during invocation, automatically
    retries with the fallback provider.
    """
    import os

    providers = []
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and groq_key != "your_groq_api_key_here":
        providers.append(("Groq", groq_key))

    gemini_key = os.getenv("GOOGLE_API_KEY", "")
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        providers.append(("Gemini", gemini_key))

    if not providers:
        raise ValueError("No LLM provider available. Set GROQ_API_KEY or GOOGLE_API_KEY.")

    for name, key in providers:
        try:
            if name == "Groq":
                from langchain_groq import ChatGroq
                llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=key, temperature=temperature)
            else:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=key, temperature=temperature)

            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.warning(f"{name} invocation failed: {e}")
            continue

    raise RuntimeError("All LLM providers failed.")
