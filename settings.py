"""
settings.py — Credential & Configuration Manager
Reads from Streamlit secrets if available, otherwise prompts user input and persists to MongoDB.
"""

import streamlit as st
from pymongo import MongoClient
import os

# ─── Default model config ───────────────────────────────────────────────────
GROQ_CHAT_MODEL   = "openai/gpt-oss-120b"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_FAST_MODEL   = "llama-3.3-70b-versatile"

TAVILY_SEARCH_MAX_RESULTS = 5

# ─── Load from st.secrets if present ────────────────────────────────────────
def _from_secrets(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default


def get_credentials() -> dict:
    """Return credentials dict; fill from secrets → session_state → DB → UI."""
    creds = {
        "groq_api_key":    _from_secrets("GROQ_API_KEY",    st.session_state.get("groq_api_key", "")),
        "mongo_uri":       _from_secrets("MONGO_URI",       st.session_state.get("mongo_uri", "")),
        "tavily_api_key":  _from_secrets("TAVILY_API_KEY",  st.session_state.get("tavily_api_key", "")),
    }
    return creds


def render_settings_sidebar():
    """
    Render a ⚙️ Settings expander in the sidebar.
    Saves entered values to session state so they persist within the session.
    """
    with st.sidebar.expander("⚙️ Settings / Credentials", expanded=False):
        st.caption("Credentials are loaded from Streamlit secrets if configured, otherwise enter below.")

        groq = st.text_input(
            "Groq API Key",
            value=st.session_state.get("groq_api_key", _from_secrets("GROQ_API_KEY", "")),
            type="password",
            key="_set_groq"
        )
        mongo = st.text_input(
            "MongoDB URI",
            value=st.session_state.get("mongo_uri", _from_secrets("MONGO_URI", "")),
            type="password",
            key="_set_mongo"
        )
        tavily = st.text_input(
            "Tavily API Key (optional – enables web search)",
            value=st.session_state.get("tavily_api_key", _from_secrets("TAVILY_API_KEY", "")),
            type="password",
            key="_set_tavily"
        )

        if st.button("💾 Save Credentials", use_container_width=True):
            st.session_state["groq_api_key"]   = groq
            st.session_state["mongo_uri"]       = mongo
            st.session_state["tavily_api_key"]  = tavily
            st.success("Credentials saved to session.")

        # Show current status
        creds = get_credentials()
        col1, col2, col3 = st.columns(3)
        col1.metric("Groq",    "✅" if creds["groq_api_key"]   else "❌")
        col2.metric("MongoDB", "✅" if creds["mongo_uri"]       else "❌")
        col3.metric("Tavily",  "✅" if creds["tavily_api_key"]  else "⚪")


# ─── MongoDB helper ──────────────────────────────────────────────────────────
_mongo_client_cache: dict = {}

def get_mongo_db(db_name: str = "scm_intelligence"):
    creds = get_credentials()
    uri = creds["mongo_uri"]
    if not uri:
        return None
    if uri not in _mongo_client_cache:
        try:
            _mongo_client_cache[uri] = MongoClient(uri, serverSelectionTimeoutMS=5000)
        except Exception as e:
            st.sidebar.error(f"MongoDB connection failed: {e}")
            return None
    return _mongo_client_cache[uri][db_name]


def get_collection(collection_name: str):
    db = get_mongo_db()
    if db is None:
        return None
    return db[collection_name]


# ─── Groq client ────────────────────────────────────────────────────────────
from groq import Groq

def get_groq_client() -> Groq | None:
    creds = get_credentials()
    key = creds["groq_api_key"]
    if not key:
        return None
    return Groq(api_key=key)
