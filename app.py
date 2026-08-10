# ============================================================
# app.py
# Phase 4: Web UI Framework - Streamlit Frontend
# ============================================================
# Drives user visibility through a server-side Streamlit
# interface panel containing file-drop configurations,
# canvas-based visualization rendering views, and
# state-retaining recall assessment panels.
# ============================================================

import streamlit as st
import time
import logging
import os
import sys
import re
import pickle
from pathlib import Path

# ── Ensure project root is on the Python path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Patch pytz missing zoneinfo files on minimal system environments ──
import io
import builtins
def _setup_pytz_fallback():
    _orig_path_open = Path.open
    _orig_builtins_open = builtins.open

    def _mock_open_content(target):
        s = str(target)
        if "tzdata.zi" in s:
            return io.StringIO("# version 2024a\n")
        elif "zone1970.tab" in s or "zone.tab" in s or "iso3166.tab" in s:
            return io.StringIO("US\t+404251-0740023\tAmerica/New_York\nUTC\t+0000\tUTC\n")
        else:
            return io.BytesIO(b"TZif2" + b"\x00" * 40)

    def _patched_path_open(self, *args, **kwargs):
        if "zoneinfo" in str(self):
            mode = args[0] if args else kwargs.get("mode", "r")
            res = _mock_open_content(self)
            if "b" in mode and isinstance(res, io.StringIO):
                return io.BytesIO(res.getvalue().encode("utf-8"))
            return res
        return _orig_path_open(self, *args, **kwargs)

    def _patched_builtins_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, Path)) and "zoneinfo" in str(file):
            mode = args[0] if args else kwargs.get("mode", "r")
            res = _mock_open_content(file)
            if "b" in mode and isinstance(res, io.StringIO):
                return io.BytesIO(res.getvalue().encode("utf-8"))
            return res
        return _orig_builtins_open(file, *args, **kwargs)

    Path.open = _patched_path_open
    builtins.open = _patched_builtins_open
    try:
        import pytz
    finally:
        Path.open = _orig_path_open
        builtins.open = _orig_builtins_open

_setup_pytz_fallback()

from core.ingestion import PDFIngestionEngine
from core.rag_store import RAGStore
from core.llm_engine import LLMEngine
from core.visualizer import FlowchartVisualizer
from prompts.templates import PromptTemplates
from telemetry.telemetry_manager import TelemetryManager

# ── Configure Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Mentora",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING — Premium UI
# ============================================================
st.markdown("""
<style>
    /* ── Typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ══════════════════════════════════════════
       ANIMATED GRADIENT HEADER
    ══════════════════════════════════════════ */
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main-header {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite;
        padding: 2.5rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.35);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        animation: gradientShift 12s ease infinite;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 10px rgba(0,0,0,0.15);
        position: relative;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.08rem;
        margin-top: 0.6rem;
        font-weight: 400;
        position: relative;
    }

    /* ── Subtitle Chip ── */
    .header-chips {
        display: flex; gap: 0.6rem; margin-top: 1rem; flex-wrap: wrap;
        position: relative;
    }
    .header-chip {
        background: rgba(255,255,255,0.18);
        backdrop-filter: blur(6px);
        color: #fff;
        padding: 0.32rem 0.9rem;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.22);
    }

    /* ══════════════════════════════════════════
       PIPELINE PROGRESS STEPPER
    ══════════════════════════════════════════ */
    .pipeline-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0;
        margin: 1.5rem 0;
        padding: 0.8rem 1.2rem;
        background: rgba(102,126,234,0.06);
        border-radius: 14px;
        border: 1px solid rgba(102,126,234,0.10);
    }
    .pipe-step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: #a0aec0;
        transition: all 0.3s ease;
    }
    .pipe-step.done {
        color: #38a169;
        font-weight: 600;
    }
    .pipe-step.done .pipe-icon {
        background: linear-gradient(135deg, #38a169, #48bb78);
        color: #fff;
        box-shadow: 0 2px 8px rgba(56,161,105,0.35);
    }
    .pipe-step .pipe-icon {
        width: 28px; height: 28px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.82rem;
        background: #e2e8f0;
        color: #a0aec0;
        transition: all 0.3s ease;
    }
    .pipe-connector {
        flex: 1;
        height: 2px;
        background: #e2e8f0;
        margin: 0 0.3rem;
        border-radius: 2px;
    }
    .pipe-connector.done {
        background: linear-gradient(90deg, #38a169, #48bb78);
    }

    /* ══════════════════════════════════════════
       FEATURE CARDS — Glassmorphism
    ══════════════════════════════════════════ */
    .feature-card {
        background: rgba(255,255,255,0.72);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(102,126,234,0.12);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.2rem;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(102,126,234,0.18);
        border-color: rgba(102,126,234,0.30);
    }
    .feature-card h3 {
        color: #2d3748;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 0.5rem;
    }
    .feature-card p {
        color: #718096;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    /* ══════════════════════════════════════════
       METRIC CARDS
    ══════════════════════════════════════════ */
    .metric-card {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(102,126,234,0.10);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.12);
    }
    .metric-card h2 {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .metric-card p {
        color: #718096;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 0.4rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* ══════════════════════════════════════════
       SIDEBAR
    ══════════════════════════════════════════ */
    .sidebar-brand {
        background: linear-gradient(160deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.20) 50%, rgba(240,147,251,0.10) 100%);
        border: 1px solid rgba(102,126,234,0.25);
        border-radius: 18px;
        padding: 1.8rem 1.2rem 1.5rem;
        margin-bottom: 1.4rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(102,126,234,0.15), inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .sidebar-brand::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 200%; height: 100%;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.05) 50%, transparent 100%);
        animation: sidebarShimmer 4s ease-in-out infinite;
    }
    @keyframes sidebarShimmer {
        0%   { left: -100%; }
        100% { left: 100%; }
    }
    .sidebar-brand-icon {
        width: 52px; height: 52px;
        border-radius: 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 60%, #f093fb 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        margin: 0 auto 0.9rem;
        box-shadow: 0 6px 20px rgba(102,126,234,0.40);
        position: relative;
    }
    .sidebar-brand h3 {
        font-family: 'Poppins', 'Inter', sans-serif;
        color: #ffffff;
        font-weight: 800;
        font-size: 1.5rem;
        margin: 0 0 0.35rem 0;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 50%, #f0abfc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
    }
    .sidebar-brand p {
        color: rgba(255,255,255,0.60);
        font-size: 0.80rem;
        font-weight: 400;
        font-style: italic;
        letter-spacing: 0.04em;
        margin: 0;
        position: relative;
    }
    .sidebar-section {
        background: rgba(102,126,234,0.04);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(102,126,234,0.08);
    }
    .sidebar-section h4 {
        font-size: 0.78rem;
        font-weight: 600;
        color: #667eea;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.6rem 0;
    }
    .sidebar-doc-name {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #fff;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
        word-break: break-all;
    }
    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0;
        border-bottom: 1px solid rgba(102,126,234,0.08);
    }
    .sidebar-stat:last-child { border-bottom: none; }
    .sidebar-stat-label {
        font-size: 0.82rem;
        color: #718096;
        font-weight: 500;
    }
    .sidebar-stat-value {
        font-size: 0.88rem;
        font-weight: 700;
        color: #2d3748;
    }
    .sidebar-badge-row {
        display: flex; gap: 0.5rem; margin-top: 0.8rem; flex-wrap: wrap;
    }
    .sidebar-badge {
        font-size: 0.70rem;
        font-weight: 600;
        padding: 0.25rem 0.7rem;
        border-radius: 50px;
        background: rgba(102,126,234,0.10);
        color: #667eea;
    }

    /* ══════════════════════════════════════════
       QUIZ — Question Cards
    ══════════════════════════════════════════ */
    .quiz-question-card {
        background: rgba(255,255,255,0.80);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(102,126,234,0.10);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        transition: all 0.3s ease;
    }
    .quiz-question-card:hover {
        border-color: rgba(102,126,234,0.25);
        box-shadow: 0 4px 16px rgba(102,126,234,0.08);
    }
    .quiz-q-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .quiz-q-num {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #fff;
        font-size: 0.82rem;
        font-weight: 700;
        width: 32px; height: 32px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .quiz-q-text {
        font-size: 1rem;
        font-weight: 600;
        color: #2d3748;
        line-height: 1.5;
    }

    /* ══════════════════════════════════════════
       Q&A — Chat Bubbles (fixed alignment)
    ══════════════════════════════════════════ */
    .qa-chat-container {
        display: flex;
        flex-direction: column;
        width: 100%;
    }
    .qa-entry {
        display: flex;
        flex-direction: column;
        width: 100%;
        margin-bottom: 1.5rem;
    }
    .qa-question-bubble {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #fff;
        padding: 0.9rem 1.3rem;
        border-radius: 16px 16px 4px 16px;
        font-weight: 500;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
        max-width: 85%;
        margin-left: auto;
        word-wrap: break-word;
        box-sizing: border-box;
    }
    .qa-answer-bubble {
        background: rgba(102,126,234,0.12);
        border: 1px solid rgba(102,126,234,0.25);
        padding: 1.2rem 1.5rem;
        border-radius: 16px 16px 16px 4px;
        font-size: 0.95rem;
        color: #e2e8f0;
        line-height: 1.7;
        margin-bottom: 0.6rem;
        max-width: 90%;
        margin-right: auto;
        word-wrap: break-word;
        box-sizing: border-box;
    }
    .qa-sources {
        margin-bottom: 0.5rem;
    }

    /* ══════════════════════════════════════════
       FLOWCHART CONTAINER
    ══════════════════════════════════════════ */
    .flowchart-container {
        background: rgba(255,255,255,0.90);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(102,126,234,0.10);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }

    /* ══════════════════════════════════════════
       TAB STYLING
    ══════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(102,126,234,0.04);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 500;
        font-size: 0.88rem;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        font-weight: 600;
    }

    /* ══════════════════════════════════════════
       STATUS BADGES
    ══════════════════════════════════════════ */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 1rem;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-connected {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        box-shadow: 0 2px 8px rgba(56,161,105,0.15);
    }
    .status-disconnected {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        box-shadow: 0 2px 8px rgba(220,53,69,0.15);
    }

    /* ══════════════════════════════════════════
       UPLOAD ZONE
    ══════════════════════════════════════════ */
    .upload-zone {
        border: 2px dashed rgba(102,126,234,0.30);
        border-radius: 18px;
        padding: 2.5rem;
        text-align: center;
        background: rgba(102,126,234,0.03);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    .upload-zone:hover {
        border-color: rgba(102,126,234,0.50);
        background: rgba(102,126,234,0.06);
    }
    .upload-zone-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .upload-zone p {
        color: #718096;
        font-size: 0.92rem;
    }

    /* ── Score Display ── */
    .score-display {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 8px 28px rgba(102,126,234,0.35);
    }

    /* ── Section Titles ── */
    .section-title {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.3rem;
    }
    .section-title-icon {
        font-size: 1.6rem;
    }
    .section-title h2 {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0;
    }
    .section-subtitle {
        color: #718096;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }

    /* ── Footer ── */
    .footer-text {
        text-align: center;
        font-size: 0.75rem;
        color: #a0aec0;
        padding: 1.5rem 0;
        border-top: 1px solid rgba(0,0,0,0.05);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    """Initialize all Streamlit session state variables."""
    defaults = {
        # ── Pipeline Components ──
        "ingestion_engine": None,
        "rag_store": None,
        "llm_engine": None,
        "visualizer": None,
        "telemetry": None,
        # ── Document State ──
        "document_loaded": False,
        "document_name": "",
        "document_chunks": [],
        "document_id": None,
        "num_pages": 0,
        # ── Feature Results ──
        "summary_notes": "",
        "flowchart_dot": "",
        "flowchart_image": None,
        "qa_history": [],
        "quiz_questions_raw": "",
        "quiz_questions": "",
        "parsed_questions": [],
        "answer_key": {},
        "user_answers": {},
        "quiz_submitted": False,
        "quiz_feedback": "",
        # ── UI State ──
        "active_tab": "upload",
        "processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def initialize_components():
    """Lazy-initialize all pipeline components."""
    if st.session_state.ingestion_engine is None:
        st.session_state.ingestion_engine = PDFIngestionEngine(
            chunk_size=500, chunk_overlap=50
        )
    if st.session_state.rag_store is None:
        st.session_state.rag_store = RAGStore(
            model_name="all-MiniLM-L6-v2",
            cache_dir="cache"
        )
    if st.session_state.llm_engine is None:
        st.session_state.llm_engine = LLMEngine(model_name="llama3.2:1b")
    if st.session_state.visualizer is None:
        st.session_state.visualizer = FlowchartVisualizer(output_dir="cache")
    if st.session_state.telemetry is None:
        st.session_state.telemetry = TelemetryManager()


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    """Render the application sidebar with info and controls."""
    with st.sidebar:
        # ── Brand Header ──
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🎓</div>
            <h3>Mentora</h3>
            <p>Your Personal Study Buddy</p>
        </div>
        """, unsafe_allow_html=True)

        # ── System Status ──
        llm = st.session_state.llm_engine
        is_connected = llm and llm._check_connection()
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h4>⚡ System Status</h4>', unsafe_allow_html=True)
        if is_connected:
            st.markdown(
                '<span class="status-badge status-connected">● Ollama Connected</span>',
                unsafe_allow_html=True
            )
            st.caption(f"Model: `{llm.model_name}`")
        else:
            st.markdown(
                '<span class="status-badge status-disconnected">● Disconnected</span>',
                unsafe_allow_html=True
            )
            st.caption("Run `ollama serve` then `ollama pull llama3.2:1b`")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Document Info ──
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h4>📄 Document</h4>', unsafe_allow_html=True)
        if st.session_state.document_loaded:
            st.markdown(
                f'<div class="sidebar-doc-name">📎 {st.session_state.document_name}</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"""
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">Pages</span>
                <span class="sidebar-stat-value">{st.session_state.num_pages}</span>
            </div>
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">Chunks</span>
                <span class="sidebar-stat-value">{len(st.session_state.document_chunks)}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🗑️ Clear Document", use_container_width=True):
                reset_document_state()
                st.rerun()
        else:
            st.caption("No document loaded. Upload a PDF to get started.")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Study Progress ──
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h4>📊 Study Progress</h4>', unsafe_allow_html=True)
        telemetry = st.session_state.telemetry
        if telemetry:
            total_docs = telemetry.get_total_documents()
            avg_score = telemetry.get_average_quiz_score()
            st.markdown(f"""
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">Documents Studied</span>
                <span class="sidebar-stat-value">{total_docs}</span>
            </div>
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">Avg Quiz Score</span>
                <span class="sidebar-stat-value">{avg_score:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Footer Badges ──
        st.markdown("""
        <div class="sidebar-badge-row">
            <span class="sidebar-badge">🔒 Offline</span>
            <span class="sidebar-badge">🦙 Llama 3.2</span>
            <span class="sidebar-badge">⚡ FAISS</span>
        </div>
        """, unsafe_allow_html=True)


def reset_document_state():
    """Reset all document-related session state."""
    st.session_state.document_loaded = False
    st.session_state.document_name = ""
    st.session_state.document_chunks = []
    st.session_state.document_id = None
    st.session_state.num_pages = 0
    st.session_state.summary_notes = ""
    st.session_state.flowchart_dot = ""
    st.session_state.flowchart_image = None
    st.session_state.qa_history = []
    st.session_state.quiz_questions_raw = ""
    st.session_state.quiz_questions = ""
    st.session_state.parsed_questions = []
    st.session_state.answer_key = {}
    st.session_state.user_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.quiz_feedback = ""
    if st.session_state.rag_store:
        st.session_state.rag_store.clear()


# ============================================================
# HEADER
# ============================================================
def render_header():
    """Render the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Mentora</h1>
        <p>Transform your PDF textbooks into interactive learning roadmaps — 
        completely offline, completely private.</p>
        <div class="header-chips">
            <span class="header-chip">📄 PDF Parsing</span>
            <span class="header-chip">📝 Smart Notes</span>
            <span class="header-chip">🗺️ Study Roadmaps</span>
            <span class="header-chip">💬 Q&A</span>
            <span class="header-chip">📝 Quizzes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PHASE 1: PDF UPLOAD & INGESTION
# ============================================================


def render_upload_tab():
    """Render the PDF upload and processing interface."""
    st.markdown("""
    <div class="section-title">
        <span class="section-title-icon">📤</span>
        <h2>Upload Your Document</h2>
    </div>
    <p class="section-subtitle">Drop a PDF to start your study session. It will be parsed, chunked, and indexed locally for instant retrieval.</p>
    """, unsafe_allow_html=True)

    if not st.session_state.document_loaded:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-zone-icon">📄</div>
            <p>Drag & drop your PDF below, or click Browse Files</p>
        </div>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF textbook, research paper, or study material.",
        key="pdf_uploader",
        label_visibility="collapsed"
    )

    if uploaded_file is not None and not st.session_state.document_loaded:
        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            process_document(uploaded_file)

    # ── Cache loader removed — not needed for this deployment ──

    if st.session_state.document_loaded:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h2>{}</h2>
                <p>Pages Extracted</p>
            </div>
            """.format(st.session_state.num_pages), unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h2>{}</h2>
                <p>Text Chunks</p>
            </div>
            """.format(len(st.session_state.document_chunks)), unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h2>✅</h2>
                <p>Ready to Study</p>
            </div>
            """, unsafe_allow_html=True)

        st.success(f"**{st.session_state.document_name}** is processed and indexed. "
                   f"Navigate to the tabs above to start studying!")


def process_document(uploaded_file):
    """Process the uploaded PDF through Phase 1 & 2 pipeline."""
    start_time = time.time()
    progress_bar = st.progress(0, text="Starting document processing...")

    try:
        # ── Phase 1: Ingestion & Parsing ──
        progress_bar.progress(10, text="📖 Extracting text from PDF...")
        engine = st.session_state.ingestion_engine
        pdf_bytes = uploaded_file.read()

        progress_bar.progress(30, text="✂️ Chunking text into context blocks...")
        chunks = engine.process_pdf(pdf_bytes)

        if not chunks:
            st.error("❌ No text could be extracted from this PDF. "
                     "The document may be image-only or corrupted.")
            return

        # ── Count unique pages ──
        unique_pages = set(c["page_num"] for c in chunks)
        st.session_state.num_pages = len(unique_pages)

        # ── Phase 2: RAG Store Indexing ──
        progress_bar.progress(60, text="🔢 Computing vector embeddings...")
        rag = st.session_state.rag_store
        rag.clear()
        rag.add_documents(chunks)

        progress_bar.progress(85, text="💾 Saving index to disk cache...")
        doc_base_name = uploaded_file.name.replace(".pdf", "")
        rag.save_index(filename=doc_base_name)

        # ── Save document chunks to disk (for Jetson deployment) ──
        chunks_path = Path("cache") / f"{doc_base_name}.chunks"
        with open(chunks_path, "wb") as f:
            pickle.dump({
                "chunks": chunks,
                "document_name": uploaded_file.name,
                "num_pages": len(unique_pages),
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved %d document chunks to '%s'", len(chunks), chunks_path)

        # ── Update Session State ──
        st.session_state.document_loaded = True
        st.session_state.document_name = uploaded_file.name
        st.session_state.document_chunks = chunks

        # ── Phase 5: Telemetry Logging ──
        processing_time = (time.time() - start_time) * 1000
        doc_id = st.session_state.telemetry.log_document_upload(
            filename=uploaded_file.name,
            file_size_bytes=len(pdf_bytes),
            num_pages=st.session_state.num_pages,
            num_chunks=len(chunks),
            processing_time_ms=processing_time
        )
        st.session_state.document_id = doc_id

        progress_bar.progress(100, text="✅ Document processed successfully!")
        time.sleep(0.5)
        progress_bar.empty()

        logger.info(f"Document processed: {uploaded_file.name} → "
                    f"{len(chunks)} chunks in {processing_time:.0f}ms")
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Error processing document: {str(e)}")
        logger.error(f"Document processing error: {e}", exc_info=True)


# ============================================================
# FEATURE 1: SUMMARY NOTES & FLOWCHARTS
# ============================================================
def render_summary_tab():
    """Render the Summary Notes generation interface."""
    if not st.session_state.document_loaded:
        st.warning("⚠️ Please upload a PDF document first.")
        return

    st.markdown("""
    <div class="section-title">
        <span class="section-title-icon">📝</span>
        <h2>Summary Notes</h2>
    </div>
    <p class="section-subtitle">Generate comprehensive study notes covering every topic in your document.</p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("#### 📋 Summary Notes")
    st.caption("AI-powered notes covering every topic in your PDF")
    if st.button("✨ Generate Summary Notes", type="primary",
                  use_container_width=True, key="btn_summary"):
        generate_summary_notes()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.summary_notes:
        st.markdown("---")
        st.markdown(st.session_state.summary_notes)


def generate_summary_notes():
    """Generate summary notes from chunks spread across the entire document.
    
    Samples chunks evenly from start to end so ALL topics are represented,
    then makes a single LLM call for fast generation.
    """
    start_time = time.time()

    with st.spinner("🤖 Generating summary notes..."):
        try:
            all_chunks = st.session_state.document_chunks
            llm = st.session_state.llm_engine
            
            # ── Sample chunks spread across the entire PDF ──
            # Take up to 25 chunks, evenly spaced, to cover all topics
            max_chunks = 25
            if len(all_chunks) > max_chunks:
                step = len(all_chunks) / max_chunks
                selected = [all_chunks[int(i * step)] for i in range(max_chunks)]
            else:
                selected = all_chunks
            
            context = "\n\n".join([c["text"] for c in selected])

            # ── Single LLM call ──
            system_prompt, user_prompt = PromptTemplates.get_summary_prompt(context)
            response = llm.generate(prompt=user_prompt, system_prompt=system_prompt)
            
            st.session_state.summary_notes = response

            # ── Telemetry ──
            elapsed = (time.time() - start_time) * 1000
            st.session_state.telemetry.log_feature_usage(
                document_id=st.session_state.document_id,
                feature_name="summary_notes",
                response_time_ms=elapsed,
                success=True
            )
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error generating summary: {str(e)}")
            st.session_state.telemetry.log_feature_usage(
                document_id=st.session_state.document_id,
                feature_name="summary_notes",
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e)
            )


def generate_flowchart():
    """Generate a Graphviz DOT study roadmap flowchart.
    
    Strategy:
    1. Try LLM-generated DOT code
    2. If LLM DOT is invalid, extract topics from document chunks
       and build a programmatic study roadmap (never shows generic chart)
    """
    start_time = time.time()

    with st.spinner("🤖 Generating study roadmap flowchart..."):
        try:
            # ── Gather context ──
            all_chunks = st.session_state.document_chunks
            if len(all_chunks) > 20:
                step = max(1, len(all_chunks) // 20)
                selected = all_chunks[::step][:20]
            else:
                selected = all_chunks
            context = "\n\n".join([c["text"] for c in selected])

            # ── Get prompt templates ──
            system_prompt, user_prompt = PromptTemplates.get_flowchart_prompt(context)

            # ── Generate DOT via LLM ──
            llm = st.session_state.llm_engine
            response = llm.generate(prompt=user_prompt, system_prompt=system_prompt)

            # ── Extract and sanitize DOT code ──
            dot_code = llm._extract_dot_code(response)
            visualizer = st.session_state.visualizer
            dot_code = visualizer.sanitize_dot_source(dot_code)

            # ── Validate DOT syntax ──
            is_valid, error_msg = visualizer.validate_dot_syntax(dot_code)
            if not is_valid:
                logger.warning(f"Invalid DOT from LLM: {error_msg}. Building topic-based fallback.")
                topics = _extract_topics_from_chunks(all_chunks)
                dot_code = visualizer.create_default_flowchart(
                    title=st.session_state.document_name,
                    topics=topics
                )

            # ── Render to PNG ──
            png_bytes = visualizer.render_dot_to_png(dot_code, filename="flowchart")
            if png_bytes:
                st.session_state.flowchart_dot = dot_code
                st.session_state.flowchart_image = png_bytes

            # ── Telemetry ──
            elapsed = (time.time() - start_time) * 1000
            st.session_state.telemetry.log_feature_usage(
                document_id=st.session_state.document_id,
                feature_name="flowchart",
                response_time_ms=elapsed,
                success=True
            )
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error generating flowchart: {str(e)}")
            st.session_state.telemetry.log_feature_usage(
                document_id=st.session_state.document_id,
                feature_name="flowchart",
                response_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e)
            )


def _extract_topics_from_chunks(chunks: list[dict]) -> list[str]:
    """Extract topic headings from document chunks using text analysis.
    
    Looks for headings, bold text, numbered sections, and key phrases
    to build a list of topics in document order.
    """
    topics = []
    seen = set()
    
    for chunk in chunks:
        text = chunk.get("text", "")
        lines = text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 4:
                continue
            
            topic = None
            
            # Match markdown headers: ## Topic, ### Sub-topic
            m = re.match(r'^#{1,4}\s+(.+)', line)
            if m:
                topic = m.group(1).strip()
            
            # Match numbered sections: 1.1 Topic, 2. Topic
            if not topic:
                m = re.match(r'^\d+\.\d*\s+([A-Z][^.!?]{3,60})$', line)
                if m:
                    topic = m.group(1).strip()
            
            # Match ALL-CAPS headings (common in textbook PDFs)
            if not topic and line.isupper() and 4 < len(line) < 60:
                topic = line.title()
            
            # Match bold-like headings
            if not topic:
                m = re.match(r'^\*\*(.+?)\*\*', line)
                if m:
                    topic = m.group(1).strip()
            
            # Clean and deduplicate
            if topic:
                topic = re.sub(r'[\*#_]', '', topic).strip()
                topic = topic[:50]
                lower_topic = topic.lower()
                if lower_topic not in seen and len(topic) > 3:
                    seen.add(lower_topic)
                    topics.append(topic)
    
    # If not enough topics from headings, use first line of each chunk
    if len(topics) < 4:
        topics = []
        seen = set()
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            first_line = text.split("\n")[0].strip()
            first_line = re.sub(r'[\*#_]', '', first_line).strip()
            if len(first_line) > 10:
                topic = first_line[:55]
                if topic.endswith(' '):
                    topic = topic.rsplit(' ', 1)[0]
                lower_topic = topic.lower()
                if lower_topic not in seen:
                    seen.add(lower_topic)
                    topics.append(topic)
    
    # Limit to 8-12 topics for a clean roadmap
    if len(topics) > 12:
        step = len(topics) / 12
        topics = [topics[int(i * step)] for i in range(12)]
    
    return topics


# ============================================================
# FEATURE 2: CONTEXTUAL Q&A
# ============================================================
def render_qa_tab():
    """Render the Contextual Q&A interface with RAG retrieval."""
    if not st.session_state.document_loaded:
        st.warning("⚠️ Please upload a PDF document first.")
        return

    st.markdown("""
    <div class="section-title">
        <span class="section-title-icon">💬</span>
        <h2>Contextual Q&A</h2>
    </div>
    <p class="section-subtitle">Ask any question about your document. Answers are grounded in the actual content using retrieval-augmented generation.</p>
    """, unsafe_allow_html=True)

    # ── Question Input ──
    question = st.text_input(
        "Ask a question about your document:",
        placeholder="e.g., What are the main topics covered in this chapter?",
        key="qa_input"
    )

    if st.button("🔍 Get Answer", type="primary",
                  disabled=not question, key="btn_qa"):
        answer_question(question)

    # ── Chat History ──
    if st.session_state.qa_history:
        st.markdown("---")
        # Build all Q&A entries as a single HTML block for stable alignment
        chat_html = '<div class="qa-chat-container">'
        for entry in reversed(st.session_state.qa_history):
            source_chips = " ".join([
                f'<span class="sidebar-badge">📄 Page {s["page_num"]}</span>'
                for s in entry["sources"]
            ])
            chat_html += (
                f'<div class="qa-entry">'
                f'<div class="qa-question-bubble">❓ {entry["question"]}</div>'
                f'<div class="qa-answer-bubble">{entry["answer"]}</div>'
                f'<div class="qa-sources"><div class="sidebar-badge-row">{source_chips}</div></div>'
                f'</div>'
            )
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)


def answer_question(question: str):
    """Answer a question using RAG retrieval + LLM generation."""
    start_time = time.time()

    with st.spinner("🔍 Searching document and generating answer..."):
        try:
            # ── RAG Retrieval (Phase 2) ──
            rag = st.session_state.rag_store
            results = rag.search(query=question, top_k=5)

            if not results:
                st.warning("No relevant context found in the document for this question.")
                return

            # ── Build context from retrieved chunks ──
            context = "\n\n".join([
                f"[Page {r['page_num']}]: {r['text']}" for r in results
            ])

            # ── Get prompt templates ──
            system_prompt, user_prompt = PromptTemplates.get_qa_prompt(question, context)

            # ── Generate answer via LLM (Phase 3) ──
            llm = st.session_state.llm_engine
            answer = llm.generate(prompt=user_prompt, system_prompt=system_prompt)

            # ── Store in history ──
            st.session_state.qa_history.append({
                "question": question,
                "answer": answer,
                "sources": results
            })

            # ── Telemetry ──
            elapsed = (time.time() - start_time) * 1000
            st.session_state.telemetry.log_feature_usage(
                document_id=st.session_state.document_id,
                feature_name="qa",
                response_time_ms=elapsed,
                success=True
            )
            st.session_state.telemetry.log_qa_session(
                document_id=st.session_state.document_id,
                question=question,
                answer_length=len(answer)
            )
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")
            logger.error(f"Q&A error: {e}", exc_info=True)


# ============================================================
# FEATURE 3: REVISION QUIZ & FEEDBACK
# ============================================================

# ============================================================
# QUIZ PIPELINE — Helper Functions
# ============================================================

def parse_quiz_questions_new(quiz_text: str) -> tuple[list[dict], dict]:
    """Universal parser: handles Q: format, numbered format, ANSWER/CORRECT answers.
    Also extracts EVIDENCE and SOURCE fields for grounding verification."""
    questions: list[dict] = []
    answer_key: dict = {}

    # Split into blocks based on format detected
    if re.search(r'\bQ\s*:', quiz_text, re.IGNORECASE):
        raw_blocks = re.split(r'(?=\bQ\s*:)', quiz_text.strip(), flags=re.IGNORECASE)
    else:
        raw_blocks = re.split(r'(?=^\s*\d+[\.\)]\s)', quiz_text.strip(),
                              flags=re.MULTILINE)

    q_counter = 1
    for block in raw_blocks:
        block = block.strip()
        if not block or len(block) < 10:
            continue

        # Extract question text
        q_text = None
        m = re.match(r'Q\s*:\s*(.+?)(?=\n\s*[a-d]\s*[\)\.])',
                     block, re.DOTALL | re.IGNORECASE)
        if m:
            q_text = m.group(1).strip()
        else:
            m = re.match(r'^\s*\d+[\.\)]\s*(.+?)(?=\n\s*[a-d]\s*[\)\.])',
                         block, re.DOTALL | re.IGNORECASE)
            if m:
                q_text = m.group(1).strip()
            else:
                q_lines = []
                for line in block.split('\n'):
                    if re.match(r'^\s*[a-d]\s*[\)\.]', line, re.IGNORECASE):
                        break
                    q_lines.append(line)
                q_text = re.sub(r'^\d+[\.\)]\s*', '', ' '.join(q_lines)).strip()

        if not q_text or len(q_text) < 5:
            continue

        # Extract options line by line
        options: dict[str, str] = {}
        for line in block.split('\n'):
            s = line.strip()
            om = re.match(r'^([a-d])[\)\.\  :]+(.+)', s, re.IGNORECASE)
            if om:
                letter = om.group(1).lower()
                text = re.sub(r'\s*(ANSWER|CORRECT|EVIDENCE|SOURCE)\s*:.*$', '',
                              om.group(2), flags=re.IGNORECASE).strip()
                if text:
                    options[letter] = text

        # Extract answer
        am = re.search(r'(?:ANSWER|CORRECT(?:\s+ANSWER)?)\s*:\s*([a-d])',
                       block, re.IGNORECASE)
        correct = am.group(1).lower() if am else None

        # Extract evidence (NEW)
        evidence = ""
        ev_match = re.search(
            r'EVIDENCE\s*:\s*["\']?(.+?)["\']?\s*(?=SOURCE\s*:|$)',
            block, re.IGNORECASE | re.DOTALL
        )
        if ev_match:
            evidence = ev_match.group(1).strip().strip('"\'')

        # Extract source chunk (NEW)
        source_chunk = None
        src_match = re.search(r'SOURCE\s*:\s*(\d+)', block, re.IGNORECASE)
        if src_match:
            source_chunk = int(src_match.group(1))

        if q_text and len(options) >= 2:
            questions.append({
                'number': q_counter,
                'question': q_text,
                'options': options,
                'evidence': evidence,
                'source_chunk': source_chunk
            })
            if correct:
                answer_key[q_counter] = correct
            q_counter += 1

    return questions, answer_key


def _structural_validate_question(
    q: dict, answer_key: dict,
    all_chunks: list[dict] | None = None
) -> tuple[bool, str]:
    """Enhanced validation with evidence grounding checks.

    Checks:
    1. Question text is non-empty and at least 10 chars.
    2. Exactly 4 options a/b/c/d present.
    3. An answer key entry exists for this question.
    4. The answer key letter is one of the actual options.
    5. No option is a duplicate of another.
    6. Question is not trivially short (< 5 words).
    7. (NEW) Evidence field exists and is non-empty.
    8. (NEW) Evidence is found in source chunks (fuzzy match).
    9. (NEW) Question is self-contained — no garbage patterns.
    10. (NEW) All options are meaningful (>= 3 chars).
    """
    q_num = q['number']
    q_text = q['question'].strip()
    options = q['options']

    if len(q_text) < 10:
        return False, "Question text too short"

    if len(q_text.split()) < 5:
        return False, "Question has fewer than 5 words"

    required_letters = {'a', 'b', 'c', 'd'}
    if not required_letters.issubset(set(options.keys())):
        missing = required_letters - set(options.keys())
        return False, f"Missing options: {missing}"

    # Check all option texts are meaningful (>= 3 chars)
    for letter, text in options.items():
        if not text or len(text.strip()) < 3:
            return False, f"Option {letter}) is empty or too short (< 3 chars)"

    # Check answer key exists
    if q_num not in answer_key:
        return False, "No answer key entry for this question"

    correct = answer_key[q_num]
    if correct not in options:
        return False, f"Answer '{correct}' not in available options"

    # Check for duplicate option texts
    opt_texts = [t.strip().lower() for t in options.values()]
    if len(set(opt_texts)) < len(opt_texts):
        return False, "Duplicate option texts detected"

    # (NEW) Check for garbage patterns — random IDs, hex strings, table fragments
    garbage_patterns = [
        r'[0-9a-f]{8,}',            # hex strings
        r'^\s*\d+\s*$',             # just numbers as question
        r'(?:table|fig(?:ure)?)\s+\d+', # table/figure references
        r'as shown (above|below)',   # context-dependent references
        r'the above',
        r'see page',
    ]
    for pattern in garbage_patterns:
        if re.search(pattern, q_text, re.IGNORECASE):
            return False, f"Question contains garbage/context-dependent pattern"

    # Hard-reject weak/meta answer constructions.
    banned = ("all of the above", "none of the above", "both a and b",
              "both a & b", "a and b", "b and c", "c and d")
    for letter, text in options.items():
        low = text.strip().lower()
        if any(x == low or x in low for x in banned):
            return False, f"Option {letter}) uses a banned meta/combination answer"

    # Catch fabricated normal-form acronyms seen in small-model hallucinations.
    standard_nf = {"1NF","2NF","3NF","BCNF","4NF","5NF","DKNF","6NF"}
    nf_tokens = set(re.findall(r'\b[A-Z]{2,5}NF\b|\b[1-6]NF\b', q_text.upper()))
    for text in options.values():
        nf_tokens.update(re.findall(r'\b[A-Z]{2,5}NF\b|\b[1-6]NF\b', text.upper()))
    unknown_nf = nf_tokens - standard_nf
    if unknown_nf:
        return False, f"Suspicious/nonstandard normal-form term(s): {sorted(unknown_nf)}"

    # (NEW) Evidence validation
    evidence = q.get('evidence', '')
    if not evidence or len(evidence.strip()) < 10:
        return False, "Missing or too-short evidence field"

    # (NEW) Verify evidence is grounded in source chunks
    if all_chunks:
        evidence_lower = evidence.strip().lower()
        # Build full text from all chunks for fuzzy matching
        full_text_lower = ' '.join(
            c.get('text', '') for c in all_chunks
        ).lower()
        # Check if a significant portion of the evidence appears in the source
        # Use sliding window: check if any 30-char substring of evidence is found
        evidence_grounded = False
        check_len = min(30, len(evidence_lower))
        if check_len >= 10:
            for start in range(0, len(evidence_lower) - check_len + 1, 10):
                snippet = evidence_lower[start:start + check_len]
                if snippet in full_text_lower:
                    evidence_grounded = True
                    break
        else:
            evidence_grounded = evidence_lower in full_text_lower

        if not evidence_grounded:
            return False, "Evidence not found in source document chunks"

    return True, ""


def _candidate_to_text(q: dict, correct: str) -> str:
    lines = [f"Q: {q['question']}"]
    for letter in ("a","b","c","d"):
        lines.append(f"{letter.upper()}) {q['options'].get(letter, '')}")
    lines.append(f"CLAIMED ANSWER: {correct.upper()}")
    lines.append(f"EVIDENCE: {q.get('evidence','')}")
    return "\n".join(lines)


def _semantic_validate_question(llm, q: dict, correct: str, source_text: str):
    """Semantic validation with retry logic for 1B model consistency.

    The 1B model is non-deterministic, so we try the verification up to
    3 times. If the independent answer matches the generated answer on
    ANY attempt, the question is accepted. This maintains quality while
    accounting for the small model's randomness.
    """
    for try_num in range(3):
        try:
            verdict = llm.validate_quiz_question(
                source_text, _candidate_to_text(q, correct),
                PromptTemplates.QUESTION_VALIDATION_SYSTEM
            ).strip()
            m = re.match(r'^VALID\s*:\s*([A-D])\b', verdict, re.IGNORECASE)
            if m:
                independent = m.group(1).lower()
                if independent == (correct or "").lower():
                    return True, ""
                # Answer mismatch — try again (model may be inconsistent)
                logger.info(
                    "Semantic try %d: mismatch generator=%s validator=%s",
                    try_num + 1, correct, independent
                )
            else:
                logger.info("Semantic try %d: no VALID verdict: %s", try_num + 1, verdict[:80])
        except Exception as e:
            logger.warning("Semantic validation attempt %d error: %s", try_num + 1, e)
    return False, "Semantic validation failed after 3 attempts"


def _get_rag_context_for_quiz(num_questions: int) -> tuple[str, list[dict]]:
    """Select quiz context using RAG retrieval with chunk metadata.

    Returns
    -------
    tuple[str, list[dict]]
        (context_string, list_of_chunk_metadata_dicts)
        Each dict has 'text', 'chunk_index', 'page_num'.
    """
    all_chunks = st.session_state.document_chunks
    rag = st.session_state.rag_store

    if not all_chunks:
        return "", []

    # ── Step 1: Extract candidate concept queries ──
    target_queries = min(num_questions * 3, 20)
    step = max(1, len(all_chunks) // target_queries)
    candidate_chunks = all_chunks[::step][:target_queries]

    queries = []
    for chunk in candidate_chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        first_sentence = re.split(r'[.!?\n]', text)[0].strip()
        if len(first_sentence) > 15:
            queries.append(first_sentence[:120])

    # Fallback
    if len(queries) < num_questions:
        for chunk in all_chunks[:30]:
            for sent in re.split(r'[.!?]', chunk.get("text", "")):
                sent = sent.strip()
                if len(sent) > 20:
                    queries.append(sent[:120])
                    if len(queries) >= num_questions * 3:
                        break
            if len(queries) >= num_questions * 3:
                break

    # ── Step 2: RAG retrieval for each query ──
    seen_texts: set[str] = set()
    retrieved_chunks: list[dict] = []

    for query in queries:
        try:
            results = rag.search(query=query, top_k=2)
            for r in results:
                text_key = r["text"][:80]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    retrieved_chunks.append(r)
        except Exception:
            pass

    # ── Step 3: If too few, pad with evenly-sampled ──
    if len(retrieved_chunks) < num_questions * 2:
        step2 = max(1, len(all_chunks) // (num_questions * 2))
        for chunk in all_chunks[::step2]:
            text_key = chunk["text"][:80]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                retrieved_chunks.append(chunk)
            if len(retrieved_chunks) >= num_questions * 3:
                break

    # Limit total chunks
    max_chunks = min(len(retrieved_chunks), num_questions * 3, 25)
    final_chunks = retrieved_chunks[:max_chunks]

    # ── Step 4: Build chunk metadata list (NEW — includes index and page) ──
    chunk_metadata: list[dict] = []
    for c in final_chunks:
        chunk_metadata.append({
            'text': c['text'],
            'chunk_index': c.get('chunk_index', 0),
            'page_num': c.get('page_num', 0),
        })

    # ── Step 5: Build combined context string with chunk numbers ──
    numbered_parts = []
    for i, cm in enumerate(chunk_metadata, 1):
        numbered_parts.append(
            f"[Chunk {i}] (Page {cm['page_num']}): {cm['text']}"
        )
    full_context = "\n\n".join(numbered_parts)

    logger.info(
        "RAG quiz context: %d unique chunks retrieved via %d queries",
        len(final_chunks), len(queries)
    )
    return full_context, chunk_metadata


def render_quiz_tab():
    """Render the Revision Quiz & Feedback interface with interactive MCQ."""
    if not st.session_state.document_loaded:
        st.warning("⚠️ Please upload a PDF document first.")
        return

    st.markdown("""
    <div class="section-title">
        <span class="section-title-icon">📝</span>
        <h2>Revision Quiz</h2>
    </div>
    <p class="section-subtitle">Test your understanding with auto-generated MCQ questions. Select your answer for each, then submit for instant feedback.</p>
    """, unsafe_allow_html=True)

    # ── Quiz Generation ──
    if not st.session_state.parsed_questions:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            num_questions = st.slider("Number of questions", 3, 10, 5, key="quiz_num")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎯 Generate Quiz", type="primary",
                          use_container_width=True, key="btn_quiz"):
                generate_quiz(num_questions)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # ── Display Quiz with Interactive Radio Buttons ──
        if not st.session_state.quiz_submitted:
            total_q = len(st.session_state.parsed_questions)
            answered = len(st.session_state.user_answers)
            st.markdown(
                f"<p style='color:#718096; font-size:0.88rem; margin-bottom:1rem;'>"
                f"📋 <b>{total_q} evidence-grounded questions</b> generated "
                f"&nbsp;•&nbsp; "
                f"✏️ <b>{answered}/{total_q}</b> answered</p>",
                unsafe_allow_html=True
            )
            
            for q in st.session_state.parsed_questions:
                st.markdown(
                    f'<div class="quiz-question-card">'
                    f'<div class="quiz-q-header">'
                    f'<span class="quiz-q-num">{q["number"]}</span>'
                    f'<span class="quiz-q-text">{q["question"]}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
                
                if q['options']:
                    option_labels = []
                    for letter in sorted(q['options'].keys()):
                        option_labels.append(f"{letter}) {q['options'][letter]}")
                    
                    selected = st.radio(
                        f"Select your answer for Q{q['number']}:",
                        options=option_labels,
                        key=f"quiz_q_{q['number']}",
                        index=None,
                        label_visibility="collapsed"
                    )
                    
                    if selected:
                        selected_letter = selected[0]
                        st.session_state.user_answers[q['number']] = selected_letter
                
                st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
            
            # ── Submit & New Quiz buttons ──
            col1, col2 = st.columns(2)
            with col1:
                all_answered = len(st.session_state.user_answers) == len(st.session_state.parsed_questions)
                if st.button("📤 Submit Answers", type="primary",
                              use_container_width=True, key="btn_submit_quiz"):
                    if st.session_state.user_answers:
                        grade_quiz_interactive()
                    else:
                        st.warning("⚠️ Please select an answer for at least one question.")
                
                if not all_answered and st.session_state.user_answers:
                    st.caption(f"✏️ Answered {len(st.session_state.user_answers)} of "
                              f"{len(st.session_state.parsed_questions)} questions")
            with col2:
                if st.button("🔄 New Quiz", use_container_width=True, key="btn_new_quiz"):
                    reset_quiz_state()
                    st.rerun()

        # ── Display Feedback After Submission ──
        if st.session_state.quiz_submitted and st.session_state.quiz_feedback:
            st.markdown("---")
            st.markdown("### 📊 Your Results & Feedback")
            st.markdown(st.session_state.quiz_feedback)

            if st.button("🔄 Take Another Quiz", type="primary",
                          use_container_width=True, key="btn_retake"):
                reset_quiz_state()
                st.rerun()


def reset_quiz_state():
    """Reset all quiz-related session state."""
    st.session_state.quiz_questions_raw = ""
    st.session_state.quiz_questions = ""
    st.session_state.parsed_questions = []
    st.session_state.answer_key = {}
    st.session_state.user_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.quiz_feedback = ""


def generate_quiz(num_questions: int):
    """Generate validated quiz questions — one MCQ per concept chunk.

    Pipeline (per the improved design):
    1. Select diverse concept chunks via RAG (or even sampling).
    2. For each chunk, generate exactly ONE MCQ using get_single_question_prompt.
    3. Validate each MCQ deterministically (4 options, answer present, evidence).
    4. On FAIL: regenerate with a different chunk (max 2 retries per slot).
    5. Quality > Quantity: show fewer valid questions rather than garbage.
    """
    import random as _random
    start_time = time.time()
    status_placeholder = st.empty()

    with st.spinner("Generating quiz questions..."):
        try:
            llm = st.session_state.llm_engine
            all_chunks = st.session_state.document_chunks

            if not all_chunks:
                st.error("No document chunks found. Please re-upload the PDF.")
                status_placeholder.empty()
                return

            # ── Step 1: Pick candidate chunks (diverse, spread across doc) ──
            status_placeholder.info("Selecting content from document...")

            # Select 3x num_questions chunks spread evenly across the document
            pool_size = min(len(all_chunks), num_questions * 8)
            if len(all_chunks) <= pool_size:
                candidate_chunks = list(all_chunks)
            else:
                step = len(all_chunks) / pool_size
                candidate_chunks = [all_chunks[int(i * step)] for i in range(pool_size)]

            # Shuffle so we don't always start from the beginning
            _random.shuffle(candidate_chunks)

            # ── Step 2: Generate ONE MCQ per chunk slot ──
            valid_questions: list[dict] = []
            valid_answer_key: dict = {}
            chunk_idx_used: set = set()

            # We iterate over candidate_chunks; each slot gets max 3 attempts
            chunk_queue = list(candidate_chunks)
            slot = 1  # question slot number

            while slot <= num_questions and chunk_queue:
                status_placeholder.info(
                    f"Generating question {slot}/{num_questions}..."
                )

                # Try up to 8 chunks for this slot
                success = False
                for attempt in range(8):
                    if not chunk_queue:
                        break
                    chunk = chunk_queue.pop(0)
                    chunk_text = chunk.get("text", "").strip()
                    if not chunk_text or len(chunk_text) < 50:
                        continue

                    chunk_num = chunk.get("chunk_index", slot)
                    page_num = chunk.get("page_num", None)

                    try:
                        sys_p, usr_p = PromptTemplates.get_single_question_prompt(
                            chunk_text,
                            chunk_number=chunk_num,
                            page_num=page_num
                        )
                        raw = llm.generate(prompt=usr_p, system_prompt=sys_p)
                        parsed, ak = parse_quiz_questions_new(raw)

                        if not parsed:
                            logger.warning("Slot %d attempt %d: no parse", slot, attempt+1)
                            continue

                        q = parsed[0]
                        q["number"] = slot
                        correct = ak.get(1, ak.get(slot, ""))
                        test_ak = {slot: correct}

                        # Full quality validation: structural + semantic
                        is_valid, reason = _structural_validate_question(
                            q, test_ak, all_chunks=[chunk]
                        )
                        if is_valid:
                            is_valid, reason = _semantic_validate_question(
                                llm, q, correct, chunk_text
                            )

                        if is_valid:
                            q["source_chunk"] = chunk_num
                            q["source_page"] = page_num
                            valid_questions.append(q)
                            valid_answer_key[slot] = correct
                            logger.info("Slot %d: structurally + semantically valid.", slot)
                            success = True
                            break
                        else:
                            logger.warning(
                                "Slot %d attempt %d failed validation: %s",
                                slot, attempt + 1, reason
                            )

                    except Exception as e:
                        logger.warning("Slot %d attempt %d error: %s", slot, attempt+1, e)

                if not success:
                    logger.warning(
                        "Slot %d: all attempts failed. Skipping (quality > quantity).",
                        slot
                    )

                slot += 1  # always advance, whether success or not

            # ── Step 3: Renumber without losing question-answer pairing ──
            accepted_pairs = [(q, valid_answer_key.get(q["number"], "")) for q in valid_questions]
            final_ak: dict = {}
            for i, (q, correct) in enumerate(accepted_pairs, start=1):
                q["number"] = i
                final_ak[i] = correct

            if not valid_questions:
                st.error(
                    "Could not generate valid questions. "
                    "Please try again — the model may need another attempt."
                )
                status_placeholder.empty()
                return

            if len(valid_questions) < num_questions:
                st.info(
                    f"Generated {len(valid_questions)} high-quality questions "
                    f"(requested {num_questions}). "
                    "Fewer validated questions are better than low-quality ones."
                )

            # ── Step 4: Build display text ──
            trimmed_quiz_text = ""
            for q in valid_questions:
                trimmed_quiz_text += f"{q['number']}. {q['question']}\n"
                for letter in sorted(q["options"].keys()):
                    trimmed_quiz_text += f"{letter}) {q['options'][letter]}\n"
                trimmed_quiz_text += "\n"

            # ── Step 5: Persist in session state ──
            st.session_state.quiz_questions_raw = ""
            st.session_state.quiz_questions = trimmed_quiz_text.strip()
            st.session_state.parsed_questions = valid_questions
            st.session_state.answer_key = final_ak
            st.session_state.user_answers = {}

            logger.info(
                "Quiz ready: %d valid questions, answer key: %s",
                len(valid_questions), final_ak
            )

            # ── Telemetry ──
            elapsed = (time.time() - start_time) * 1000
            st.session_state.telemetry.log_feature_usage(
                document_id=st.session_state.document_id,
                feature_name="quiz_generation",
                response_time_ms=elapsed,
                success=True
            )
            status_placeholder.empty()
            st.rerun()

        except Exception as e:
            status_placeholder.empty()
            st.error(f"Error generating quiz: {str(e)}")
            logger.error("Quiz generation error: %s", e, exc_info=True)



def _regenerate_single_question(
    llm,
    chunk_metadata: list[dict],
    target_number: int,
    max_retries: int = 2,
    all_chunks: list[dict] | None = None
) -> tuple[dict | None, str | None]:
    """Attempt to regenerate a single valid quiz question with evidence.

    Tries up to ``max_retries`` different context chunks, returning the
    first question that passes enhanced validation.
    """
    import random
    if not chunk_metadata:
        return None, None

    samples = random.sample(
        chunk_metadata,
        min(max_retries, len(chunk_metadata))
    )

    for i, cm in enumerate(samples):
        try:
            chunk_num = i + 1
            sys_p, usr_p = PromptTemplates.get_single_question_prompt(
                cm['text'],
                chunk_number=chunk_num,
                page_num=cm.get('page_num')
            )
            regen_raw = llm.generate(prompt=usr_p, system_prompt=sys_p)
            regen_parsed, regen_ak = parse_quiz_questions_new(regen_raw)

            if regen_parsed:
                q = regen_parsed[0]
                q['number'] = target_number
                correct = regen_ak.get(1, regen_ak.get(target_number, ''))
                test_ak = {target_number: correct}
                is_valid, _ = _structural_validate_question(
                    q, test_ak, all_chunks=all_chunks
                )
                if is_valid:
                    return q, correct
        except Exception as e:
            logger.warning("Regeneration attempt failed: %s", e)

    return None, None


def grade_quiz_interactive():
    """Grade quiz using pure Python letter comparison with evidence display.

    Compares user's selected letter against the stored answer key.
    Shows evidence quote from source material for each question.
    """
    start_time = time.time()

    try:
        user_answers = st.session_state.user_answers
        parsed_qs = st.session_state.parsed_questions
        answer_key = st.session_state.answer_key

        correct_count = 0
        total_questions = len(parsed_qs)
        feedback_parts = []

        for q in parsed_qs:
            q_num = q['number']
            q_text = q['question']
            options = q['options']
            evidence = q.get('evidence', '')
            source_chunk = q.get('source_chunk', None)

            # Normalize both to lowercase for reliable comparison
            user_letter = (user_answers.get(q_num, '') or '').strip().lower()
            correct_letter = (answer_key.get(q_num, '') or '').strip().lower()

            # Build user answer display
            if user_letter and user_letter in options:
                user_display = f"{user_letter}) {options[user_letter]}"
            elif user_letter:
                user_display = f"{user_letter}) (option text unavailable)"
            else:
                user_display = "⏭︎ Not answered"

            # Build correct answer display
            if correct_letter and correct_letter in options:
                correct_display = f"{correct_letter}) {options[correct_letter]}"
            elif correct_letter:
                correct_display = f"{correct_letter}) (option text unavailable)"
            else:
                correct_display = (
                    "⚠️ Answer key unavailable for this question. "
                    "The LLM could not reliably determine the correct answer — "
                    "please verify manually."
                )

            # Score: both letters must be non-empty and match
            is_correct = (
                bool(user_letter)
                and bool(correct_letter)
                and user_letter == correct_letter
            )

            if is_correct:
                correct_count += 1
                score_icon = "✅ Correct (+1)"
            elif not user_letter:
                score_icon = "⏭️ Skipped (0)"
            else:
                score_icon = "❌ Incorrect (0)"

            # Build feedback with evidence (NEW)
            fb = (
                f"### Question {q_num}\n"
                f"> {q_text}\n\n"
                f"- **Score:** {score_icon}\n"
                f"- **Your Answer:** {user_display}\n"
                f"- **Correct Answer:** {correct_display}\n"
            )
            if evidence:
                fb += f'- **Evidence:** "{evidence}"\n'
            if source_chunk is not None:
                fb += f"- **Source:** Chunk {source_chunk}\n"

            feedback_parts.append(fb)

        # Calculate percentage
        percentage = (correct_count / total_questions * 100) if total_questions > 0 else 0

        # Grade message
        if percentage >= 80:
            grade_emoji, grade_msg = "🌟", "Excellent work! You have a strong grasp of the material."
        elif percentage >= 60:
            grade_emoji, grade_msg = "👍", "Good effort! Review the incorrect answers to strengthen your understanding."
        elif percentage >= 40:
            grade_emoji, grade_msg = "📖", "Keep studying! Focus on the topics you got wrong and try again."
        else:
            grade_emoji, grade_msg = "💪", "Don't give up! Revisit the document and retake the quiz to improve."

        # Build final feedback
        feedback = "\n".join(feedback_parts)
        feedback += (
            f"\n---\n"
            f"## 📊 Overall Results\n\n"
            f"- **Score:** {grade_emoji} **{correct_count} / {total_questions}** "
            f"({percentage:.0f}%)\n"
            f"- **Assessment:** {grade_msg}\n"
        )

        # Topics to review
        wrong_topics = []
        for q in parsed_qs:
            u = user_answers.get(q['number'], '')
            c = answer_key.get(q['number'], '')
            if u.lower() != c.lower():
                wrong_topics.append(q['question'])

        if wrong_topics:
            feedback += "\n### 📝 Topics to Review\n"
            for wq in wrong_topics:
                feedback += f"- {wq}\n"

        st.session_state.quiz_feedback = feedback
        st.session_state.quiz_submitted = True

        # Telemetry
        elapsed = (time.time() - start_time) * 1000
        st.session_state.telemetry.log_feature_usage(
            document_id=st.session_state.document_id,
            feature_name="quiz_grading",
            response_time_ms=elapsed,
            success=True
        )
        try:
            st.session_state.telemetry.log_quiz_result(
                document_id=st.session_state.document_id,
                num_questions=total_questions,
                score_percent=percentage,
                weak_areas=", ".join(wrong_topics[:3]) if wrong_topics else ""
            )
        except Exception:
            pass

        st.rerun()

    except Exception as e:
        st.error(f"❌ Error grading quiz: {str(e)}")


# ============================================================
# MAIN APPLICATION LAYOUT
# ============================================================
def main():
    """Main application entry point."""
    # ── Initialize ──
    init_session_state()
    initialize_components()

    # ── Render Layout ──
    render_sidebar()
    render_header()

    # ── Pipeline Progress Indicator ──
    if st.session_state.document_loaded:
        steps = [
            ("PDF Uploaded", True),
            ("Summary Notes", bool(st.session_state.summary_notes)),
            ("Q&A", bool(st.session_state.qa_history)),
            ("Quiz Complete", st.session_state.quiz_submitted),
        ]
        step_icons = ["📄", "📝", "💬", "🎯"]
        html_parts = []
        for i, ((label, done), icon) in enumerate(zip(steps, step_icons)):
            cls = "pipe-step done" if done else "pipe-step"
            icon_cls = "pipe-icon"
            checkmark = "✓" if done else str(i + 1)
            html_parts.append(
                f'<div class="{cls}">'
                f'<span class="{icon_cls}">{checkmark}</span>'
                f'{label}</div>'
            )
            if i < len(steps) - 1:
                conn_cls = "pipe-connector done" if done else "pipe-connector"
                html_parts.append(f'<div class="{conn_cls}"></div>')
        st.markdown(
            f'<div class="pipeline-bar">{" ".join(html_parts)}</div>',
            unsafe_allow_html=True
        )

    # ── Main Tabs ──
    tab_upload, tab_summary, tab_qa, tab_quiz = st.tabs([
        "📤 Upload",
        "📝 Summary Notes",
        "💬 Q&A",
        "🎯 Quiz"
    ])

    with tab_upload:
        render_upload_tab()     

    with tab_summary:
        render_summary_tab()  

    with tab_qa:
        render_qa_tab()

    with tab_quiz:
        render_quiz_tab()

    # ── Footer ──   
    st.markdown(    
        '<div class="footer-text">'
        '🎓 Mentora &nbsp;•&nbsp; Powered by Llama 3.2 &nbsp;•&nbsp; '
        '100% Offline &nbsp;•&nbsp; Your Data Stays Private'
        '</div>',
        unsafe_allow_html=True
    )

# Run Application 

if __name__ == "__main__":
    main()
       