# 🚀 Mentora — Jetson Orin Setup & Deployment Guide

> Complete instructions to set up and run the **AI Study Buddy (Mentora)** application on NVIDIA Jetson Orin hardware.

---

## Table of Contents

1. [Hardware Requirements](#1--hardware-requirements)
2. [Software Prerequisites](#2--software-prerequisites)
3. [JetPack SDK & OS Setup](#3--jetpack-sdk--os-setup)
4. [System-Level Dependencies](#4--system-level-dependencies)
5. [Python Environment Setup](#5--python-environment-setup)
6. [Ollama & LLM Configuration](#6--ollama--llm-configuration)
7. [FAISS Index Pre-Build (Laptop → Jetson)](#7--faiss-index-pre-build-laptop--jetson)
8. [Performance Tuning for Jetson Orin](#8--performance-tuning-for-jetson-orin)
9. [Running the Application](#9--running-the-application)
10. [Troubleshooting](#10--troubleshooting)

---

## 1 — Hardware Requirements

| Component             | Minimum Specification                              |
|-----------------------|----------------------------------------------------|
| **Board**             | NVIDIA Jetson Orin Nano (8 GB) or Jetson Orin NX   |
| **Storage**           | 64 GB+ microSD / NVMe SSD (SSD strongly recommended) |
| **RAM**               | 8 GB unified (LPDDR5)                              |
| **Power Supply**      | USB-C PD or barrel-jack adapter (15 W minimum)     |
| **Networking**        | Ethernet or Wi-Fi (for initial setup only — app runs fully offline) |
| **Display (optional)**| HDMI monitor for local GUI, or headless via SSH     |

---

## 2 — Software Prerequisites

| Software              | Version               | Purpose                                     |
|-----------------------|-----------------------|---------------------------------------------|
| **JetPack SDK**       | 6.0+ (L4T R36.x)     | Board Support Package, CUDA, cuDNN, TensorRT |
| **Ubuntu**            | 22.04 LTS (Jammy)     | Base OS shipped with JetPack 6              |
| **Python**            | 3.10 or 3.11          | Runtime for the application                 |
| **pip**               | 23.0+                 | Python package manager                      |
| **Ollama**            | Pre-installed on board | Local LLM inference server                  |
| **Graphviz**          | 2.43+                 | DOT → PNG flowchart rendering               |
| **Git**               | 2.34+                 | Source code cloning                         |

---

## 3 — JetPack SDK & OS Setup

If your Jetson Orin board is **not** already flashed with JetPack:

### 3.1 Flash the Board

1. Download [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) on a host Ubuntu PC (x86_64).
2. Connect the Jetson Orin to the host via USB-C in recovery mode.
3. In SDK Manager, select:
   - **Target Hardware**: Jetson Orin Nano / Orin NX
   - **JetPack Version**: 6.0+
   - Components: Jetson OS + Jetson SDK Components (CUDA, cuDNN, TensorRT)
4. Flash and complete the on-device setup (language, user account, network).

### 3.2 Verify CUDA

```bash
# Confirm CUDA is available
nvcc --version
# Expected: cuda_12.2 or later

# Confirm GPU is detected
sudo tegrastats
# Look for GR3D (GPU utilisation) and RAM entries
```

---

## 4 — System-Level Dependencies

Install these on the Jetson Orin before setting up Python packages:

```bash
# Update the package index
sudo apt update && sudo apt upgrade -y

# Python build essentials
sudo apt install -y python3-pip python3-venv python3-dev build-essential

# Graphviz (required for flowchart rendering)
sudo apt install -y graphviz libgraphviz-dev

# Image processing libraries (required by Pillow & EasyOCR)
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev libwebp-dev

# MuPDF system dependencies (used by PyMuPDF)
sudo apt install -y libmupdf-dev libfreetype-dev libharfbuzz-dev

# SQLite (usually pre-installed, needed by telemetry module)
sudo apt install -y sqlite3 libsqlite3-dev

# Networking utilities
sudo apt install -y curl wget git
```

---

## 5 — Python Environment Setup

### 5.1 Create a Virtual Environment

```bash
# Navigate to the project directory
cd ~/vector_minds_edgeminds2026internship

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 5.2 Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 5.3 Install PyTorch (Jetson-specific build)

> ⚠️ **Do NOT install PyTorch from PyPI on Jetson.** The standard `pip install torch` will download an x86 wheel that will not work. Use NVIDIA's pre-built Jetson wheel instead.

```bash
# PyTorch 2.3+ for JetPack 6 (check NVIDIA forum for latest URL)
pip install --no-cache-dir \
  torch==2.3.0 \
  --index-url https://developer.download.nvidia.com/compute/redist/jp/v60/
```

If the above index URL has changed, find the latest Jetson PyTorch wheel at:
- [NVIDIA PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [Jetson Zoo](https://elinux.org/Jetson_Zoo)

### 5.4 Install Project Dependencies

```bash
# Install all remaining Python packages
pip install -r requirements.txt
```

#### Full Package List (`requirements.txt`)

| Package                | Version   | Purpose                                          |
|------------------------|-----------|--------------------------------------------------|
| `streamlit`            | ≥ 1.32.0  | Web UI framework                                |
| `PyMuPDF`              | ≥ 1.24.0  | PDF text extraction (Phase 1)                   |
| `easyocr`              | ≥ 1.7.1   | OCR for scanned PDF pages (Phase 1)             |
| `Pillow`               | ≥ 10.2.0  | Image processing for OCR pipeline               |
| `sentence-transformers`| ≥ 2.5.0   | Embedding model (`all-MiniLM-L6-v2`) for RAG    |
| `faiss-cpu`            | ≥ 1.8.0   | Vector similarity search (FAISS index)           |
| `ollama`               | ≥ 0.2.0   | Python client for the Ollama REST API            |
| `graphviz`             | ≥ 0.20.3  | Python bindings for Graphviz DOT rendering       |
| `numpy`                | ≥ 1.26.0  | Numerical computation                            |
| `torch`                | ≥ 2.2.0   | Deep learning framework (Jetson wheel — see §5.3)|

> **Note on `faiss-cpu`:** On Jetson, `faiss-cpu` works well for the index sizes in this application. If you need GPU-accelerated FAISS, build `faiss-gpu` from source — but this is not required for Mentora.

---

## 6 — Ollama & LLM Configuration

### ⚠️ Ollama is Already Running — Do NOT Reinstall

The Ollama server is **pre-installed and already running** on the Jetson board. All approved models are pre-loaded. **Do NOT** run `ollama pull` or attempt to restart the Ollama service — it will cause unnecessary memory pressure.

### 6.1 Approved Model

| Parameter       | Value                       |
|-----------------|-----------------------------|
| **Model**       | `llama3.2:1b`               |
| **API Endpoint**| `http://172.17.0.1:11434`   |
| **Context Size**| 1024 tokens (saves ~500 MB RAM) |
| **GPU Devices** | 1 (single Jetson GPU)       |
| **Memory Map**  | Enabled (`use_mmap: True`)  |

> ⛔ **`llama3.2:1b` is the ONLY permitted model on the Jetson board — no exceptions.**
> All final demos run on `llama3.2:1b`, regardless of track. Do not attempt to pull or run any other model on the board.

### 6.2 Verify Ollama is Running

```bash
# Check the Ollama service status
curl http://172.17.0.1:11434/api/tags

# You should see llama3.2:1b in the model list
```

### 6.3 API Call Template

The application's `core/llm_engine.py` communicates with Ollama via its Python SDK. For direct REST testing, use:

```python
import requests

API_URL = "http://172.17.0.1:11434/api/generate"

payload = {
    "model": "llama3.2:1b",          # Only approved model for Jetson deployment
    "prompt": "Explain how backpropagation works in one paragraph.",
    "stream": False,
    "options": {
        "num_ctx": 1024,             # Keep context small — saves ~500 MB RAM
        "num_gpu": 1,                # 1 GPU device on Jetson Orin
        "use_mmap": True             # Memory-mapped loading for Jetson
    }
}

response = requests.post(API_URL, json=payload)
if response.status_code == 200:
    print(response.json()['response'])
else:
    print(f"Error {response.status_code}: {response.text}")
```

### 6.4 Configure Ollama Host in the Application

If the app uses the Ollama Python SDK (which connects to `localhost:11434` by default), set the environment variable to point to the correct host:

```bash
export OLLAMA_HOST=http://172.17.0.1:11434
```

Add this to your `~/.bashrc` for persistence:

```bash
echo 'export OLLAMA_HOST=http://172.17.0.1:11434' >> ~/.bashrc
source ~/.bashrc
```

---

## 7 — FAISS Index Pre-Build (Laptop → Jetson)

The Sentence Transformers embedding model and FAISS index building are memory-intensive. It is **strongly recommended** to pre-build the index on a laptop and transfer the cache to the Jetson.

### 7.1 On Your Laptop

```bash
# 1. Run the app and upload your PDF — the app auto-generates:
#      cache/<document_name>.index   — FAISS binary index
#      cache/<document_name>.meta    — Pickled RAG metadata
#      cache/<document_name>.chunks  — Pickled document chunks

streamlit run app.py

# 2. After processing, copy the cache/ directory
scp -r cache/ jetson_user@<jetson_ip>:~/vector_minds_edgeminds2026internship/cache/
```

### 7.2 On the Jetson

The app detects pre-built cache files automatically and offers a **"Load Cached Document"** option in the UI — no PDF re-processing required.

---

## 8 — Performance Tuning for Jetson Orin

### 8.1 Set Maximum Performance Mode

```bash
# Set 15W high-performance power mode (mode 0 = MAXN)
sudo nvpmodel -m 0

# Lock CPU/GPU/EMC clocks to maximum frequency
sudo jetson_clocks
```

### 8.2 Extend Swap Space

The Jetson Orin Nano has 8 GB of unified RAM. Extending swap prevents OOM kills during embedding model loading:

```bash
# Create an 8 GB swap file
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make it persistent across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 8.3 Verify Resource Allocation

```bash
# Monitor GPU, CPU, and memory usage in real-time
sudo tegrastats

# Check swap is active
free -h
```

### 8.4 Disable Unnecessary Services (Optional)

Free up RAM by stopping the desktop environment if running headless:

```bash
# Stop the GUI
sudo systemctl stop gdm3          # GNOME
# or
sudo systemctl stop lightdm       # LightDM

# Disable it from starting on boot
sudo systemctl disable gdm3
```

---

## 9 — Running the Application

### 9.1 Start the App

```bash
# Activate the virtual environment
cd ~/vector_minds_edgeminds2026internship
source venv/bin/activate

# Set Ollama host (if not in bashrc)
export OLLAMA_HOST=http://172.17.0.1:11434

# Launch Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### 9.2 Access the UI

| Access Method     | URL                                 |
|-------------------|-------------------------------------|
| **Local browser** | `http://localhost:8501`             |
| **Remote (LAN)**  | `http://<jetson_ip>:8501`           |

### 9.3 Recommended Workflow on Jetson

1. **Load a cached document** (pre-built on laptop — see §7) to avoid heavy PDF processing on the board.
2. Use **Summary Notes & Flowcharts** — generated via `llama3.2:1b` through Ollama.
3. Use **Contextual Q&A** — RAG-powered answers from the document.
4. Use **Revision Quiz** — active recall with automated grading.

---

## 10 — Troubleshooting

### Ollama Connection Refused

```
Error: Connection refused at http://172.17.0.1:11434
```

- **Cause:** Ollama service is not running or is on a different port.
- **Fix:** Check the service: `systemctl status ollama` or verify the Docker bridge IP.

### Out of Memory (OOM) Errors

```
Killed (signal 9)
```

- **Cause:** Insufficient RAM for model + embeddings.
- **Fix:**
  1. Ensure swap is active: `free -h`
  2. Use pre-built FAISS cache (§7) to skip embedding on-device
  3. Close unnecessary processes: `sudo systemctl stop gdm3`

### PyTorch Import Error on Jetson

```
Illegal instruction (core dumped)
```

- **Cause:** You installed x86 PyTorch from PyPI instead of the Jetson ARM64 wheel.
- **Fix:** Uninstall and reinstall using the NVIDIA Jetson wheel (§5.3):
  ```bash
  pip uninstall torch -y
  pip install torch==2.3.0 --index-url https://developer.download.nvidia.com/compute/redist/jp/v60/
  ```

### Graphviz "dot not found"

```
graphviz.backend.execute.ExecutableNotFound: failed to execute 'dot'
```

- **Cause:** Graphviz system binary not installed.
- **Fix:** `sudo apt install -y graphviz`

### Streamlit Not Accessible Remotely

- **Cause:** Streamlit binds to `localhost` by default.
- **Fix:** Use `--server.address 0.0.0.0`:
  ```bash
  streamlit run app.py --server.address 0.0.0.0
  ```

### EasyOCR CUDA Errors

- **Cause:** EasyOCR may attempt to use CUDA but fail on Jetson due to memory constraints.
- **Fix:** Force CPU mode by setting the environment variable before launching:
  ```bash
  export EASYOCR_GPU=False
  ```

---

## Quick Reference — Full Setup Commands

```bash
# ── System Dependencies ──
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv python3-dev build-essential \
  graphviz libgraphviz-dev libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
  libmupdf-dev libfreetype-dev libharfbuzz-dev sqlite3 libsqlite3-dev curl wget git

# ── Performance Tuning ──
sudo nvpmodel -m 0
sudo jetson_clocks
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# ── Python Environment ──
cd ~/vector_minds_edgeminds2026internship
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip setuptools wheel

# ── PyTorch (Jetson ARM64 wheel) ──
pip install torch==2.3.0 --index-url https://developer.download.nvidia.com/compute/redist/jp/v60/

# ── Project Dependencies ──
pip install -r requirements.txt

# ── Ollama Host ──
export OLLAMA_HOST=http://172.17.0.1:11434

# ── Launch ──
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

> **Built for EdgeMinds 2026 Internship** · Runs fully offline on NVIDIA Jetson Orin · Powered by Llama 3.2 1B via Ollama
