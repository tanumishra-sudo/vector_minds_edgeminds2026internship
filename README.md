# 🎓 AI Study Buddy

> A fully localized, privacy-preserving academic assistant that transforms static PDF textbooks into dynamic, interactive, and feedback-driven learning roadmaps.

## Overview

AI Study Buddy deploys a memory-optimized 1B parameter large language model (Llama 3.2 1B via Ollama) completely on edge hardware to:

- **Eliminate information overload** from lengthy PDF documents
- **Eliminate cloud resource costs** with fully offline processing
- **Create a closed-loop active recall framework** guiding students from document exposure to concept mastery

## 3-Feature Academic Path

| Step | Feature | Description |
|------|---------|-------------|
| 1 | **Upload PDF** | Ingest and parse multi-page documents |
| 2 | **Summary Notes & Flowcharts** | Auto-generated structured notes with DOT mind maps |
| 3 | **Contextual Q&A** | RAG-powered question answering from document context |
| 4 | **Revision Quiz & Feedback** | Active recall testing with targeted grading feedback |

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| UI Framework | Streamlit | Dashboard design, interactive inputs, session retention |
| Inference | Ollama + Llama 3.2 1B | Local 4-bit quantized neural network inference |
| Data Processing | PyMuPDF, EasyOCR, Sentence Transformers | Text extraction and vector embeddings |
| Vector Database | FAISS | High-speed offline localized vector retrieval |
| Graphic Network | Graphviz | Programmatic DOT syntax to PNG compilation |
| Persistence | SQLite | Embedded engine for system telemetry and logging |

## Setup Instructions

### Prerequisites

1. **Python 3.10+** installed
2. **Ollama** installed and running locally
3. **Graphviz** system binary installed

### Environment Setup

```bash
# Clone or navigate to the project directory
cd ai_study_buddy

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Pull the Llama 3.2 1B model via Ollama
ollama pull llama3.2:1b
```

### Jetson Orin Nano Optimization (Optional - Phase 5)

```bash
# Set maximum 15W high-performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Extend system swap space (recommended 8GB)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Running the Application

```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501`

## Project Directory Structure

```
ai_study_buddy/
├── app.py                    # Streamlit Frontend View & Multi-Stage Layout Control
├── requirements.txt          # Package Dependency Manifest
├── README.md                 # This file
├── core/
│   ├── __init__.py           # Core package initializer
│   ├── ingestion.py          # PyMuPDF Processing & Recursive Text Chunking
│   ├── rag_store.py          # SentenceTransformers & FAISS Database
│   ├── llm_engine.py         # Local Ollama Inference Management
│   └── visualizer.py         # Graphviz Core Integration & Image Compilation
├── prompts/
│   ├── __init__.py           # Prompts package initializer
│   └── templates.py          # System Prompt Scripts (DOT, Notes, Grading)
├── telemetry/
│   └── study_telemetry.db    # Local SQLite Database (auto-created)
└── cache/                    # FAISS Index Saved Disk Profiles (auto-created)
```

## 5-Phase Execution Model

1. **Ingestion & Parsing Layer** — PyMuPDF + EasyOCR → token-aware recursive text chunking
2. **Local RAG Store** — Sentence Transformers → FAISS volatile index with disk caching
3. **LLM Engine & Structural Interface** — Ollama/Llama 3.2 1B → Markdown notes + DOT flowcharts
4. **Web UI Framework** — Streamlit → file-drop, canvas visualization, state-retaining panels
5. **Performance Tuning & Persistence** — CUDA/Tegra config, swap extension, SQLite telemetry

## License

This project is for educational purposes. All processing is performed locally.
