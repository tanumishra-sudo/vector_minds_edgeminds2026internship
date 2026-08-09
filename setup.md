# 🚀 Mentora — Jetson Orin Deployment Guide

> Step-by-step commands to deploy **AI Study Buddy (Mentora)** on the Jetson Orin board.
>
> **Environment:** Docker container on Jetson Orin · Read-only root filesystem · Python 3.10 · Ollama pre-installed

---

## Quick Start — Copy-Paste All Commands

If you want to run everything at once, copy this entire block into your Jetson terminal:

```bash
# ── Step 1: Navigate to project ──
cd ~/vector_minds_edgeminds2026internship

# ── Step 2: Check what's already installed ──
python3 --version
pip3 --version
python3 -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null || echo "PyTorch: NOT installed"
which dot 2>/dev/null && echo "Graphviz: installed" || echo "Graphviz: NOT installed"

# ── Step 3: Upgrade pip (user-level) ──
pip3 install --user --upgrade pip setuptools wheel

# ── Step 4: Install all Python dependencies (user-level) ──
pip3 install --user streamlit PyMuPDF easyocr Pillow sentence-transformers faiss-cpu ollama graphviz numpy

# ── Step 5: Add ~/.local/bin to PATH (where pip --user installs scripts) ──
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# ── Step 6: Set Ollama endpoint ──
export OLLAMA_HOST=http://172.17.0.1:11434
echo 'export OLLAMA_HOST=http://172.17.0.1:11434' >> ~/.bashrc

# ── Step 7: Verify Ollama connection ──
curl -s http://172.17.0.1:11434/api/tags | python3 -m json.tool

# ── Step 8: Launch the app ──
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## Detailed Step-by-Step Guide

### Step 1 — Navigate to the Project

```bash
cd ~/vector_minds_edgeminds2026internship
```

If the repo isn't cloned yet:

```bash
cd ~
git clone https://github.com/tanumishra-sudo/vector_minds_edgeminds2026internship.git
cd vector_minds_edgeminds2026internship
```

---

### Step 2 — Pre-Flight Checks

Run these to see what's already available in the container:

```bash
# Check Python version (should be 3.10+)
python3 --version

# Check pip is available
pip3 --version

# Check if PyTorch is pre-installed (common in Jetson containers)
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "PyTorch is NOT installed"

# Check if Graphviz binary exists
which dot 2>/dev/null && echo "Graphviz dot: FOUND" || echo "Graphviz dot: NOT FOUND"

# Check available disk space
df -h ~
```

---

### Step 3 — Install Python Packages (User-Level)

> ⚠️ **Why `--user`?** The container has a **read-only root filesystem** (`/usr/lib/` is immutable). Using `pip install --user` installs packages to `~/.local/lib/python3.10/site-packages/` which is writable.
>
> ⚠️ **No venv needed.** `python3 -m venv` requires `ensurepip` which cannot be installed via apt on a read-only filesystem. `--user` installs achieve the same isolation.

#### 3a. Upgrade pip

```bash
pip3 install --user --upgrade pip setuptools wheel
```

#### 3b. Install PyTorch (if NOT already pre-installed)

First check:

```bash
python3 -c "import torch; print(torch.__version__)"
```

- **If PyTorch prints a version** → skip this step, it's already in the container.
- **If it says `ModuleNotFoundError`** → install the Jetson ARM64 wheel:

```bash
pip3 install --user --no-cache-dir torch --index-url https://developer.download.nvidia.com/compute/redist/jp/v60/
```

#### 3c. Install All Project Dependencies

```bash
pip3 install --user streamlit PyMuPDF easyocr Pillow sentence-transformers faiss-cpu ollama graphviz numpy
```

> This installs every package from `requirements.txt` without touching the read-only system directories.

#### 3d. Add `~/.local/bin` to PATH

pip `--user` installs executable scripts (like `streamlit`) to `~/.local/bin`. Make sure it's on your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Make it permanent:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### 3e. Verify Installation

```bash
# Should print the streamlit version
streamlit --version

# Should import without errors
python3 -c "import streamlit, fitz, easyocr, PIL, sentence_transformers, faiss, ollama, graphviz; print('All packages OK')"
```

---

### Step 4 — Handle Graphviz System Binary

The Python `graphviz` package is a wrapper — it needs the `dot` binary on the system.

```bash
which dot
```

- **If it prints a path** (e.g., `/usr/bin/dot`) → you're good, skip ahead.
- **If "not found"** → try installing:

```bash
sudo apt install -y graphviz 2>/dev/null
```

If that also fails (read-only filesystem), the flowchart feature won't render PNG images, but all other features (Summary Notes, Q&A, Quiz) will work fine.

---

### Step 5 — Configure Ollama Endpoint

The Ollama server runs on the **host machine** (outside the container), accessible via the Docker bridge IP:

```bash
export OLLAMA_HOST=http://172.17.0.1:11434
```

Make it permanent:

```bash
echo 'export OLLAMA_HOST=http://172.17.0.1:11434' >> ~/.bashrc
source ~/.bashrc
```

#### Verify Ollama is Reachable

```bash
curl -s http://172.17.0.1:11434/api/tags
```

You should see `llama3.2:1b` in the output. If not, try:

```bash
# Alternative Docker bridge IPs
curl -s http://host.docker.internal:11434/api/tags
curl -s http://172.17.0.1:11434/api/tags
curl -s http://localhost:11434/api/tags
```

Use whichever IP returns a response and update `OLLAMA_HOST` accordingly.

#### Quick LLM Test

```bash
python3 -c "
import requests
r = requests.post('http://172.17.0.1:11434/api/generate', json={
    'model': 'llama3.2:1b',
    'prompt': 'Say hello in one sentence.',
    'stream': False,
    'options': {'num_ctx': 1024, 'num_gpu': 1, 'use_mmap': True}
})
print(r.json()['response'] if r.status_code == 200 else f'Error: {r.text}')
"
```

---

### Step 6 — Launch the Application

```bash
cd ~/vector_minds_edgeminds2026internship
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

#### Access the UI

| Method              | URL                                |
|---------------------|------------------------------------|
| **On the Jetson**   | `http://localhost:8501`            |
| **From your laptop**| `http://<jetson_ip>:8501`          |

To find the Jetson's IP:

```bash
hostname -I
```

---

### Step 7 — (Optional) Pre-Build FAISS Cache on Laptop

Building the embedding index is RAM-heavy. To avoid OOM on Jetson, pre-build on your laptop:

#### On Your Laptop

```bash
cd vector_minds_edgeminds2026internship
pip install -r requirements.txt
streamlit run app.py
# Upload your PDF → app generates cache/*.index, *.meta, *.chunks files
```

#### Copy Cache to Jetson

```bash
scp -r cache/ codex@<jetson_ip>:~/vector_minds_edgeminds2026internship/cache/
```

The app will detect cached files and offer **"Load Cached Document"** — no PDF re-processing needed.

---

## Troubleshooting

### `streamlit: command not found`

```bash
# pip --user scripts aren't on PATH
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

### `Read-only file system` on apt install

This is expected in the container. Use `pip install --user` for all Python packages. System binaries like `graphviz` may already be pre-installed — check with `which dot`.

### `ModuleNotFoundError: No module named 'xyz'`

```bash
# Reinstall the missing package
pip3 install --user <package_name>
```

### `Connection refused` to Ollama

```bash
# Try different endpoints
curl http://172.17.0.1:11434/api/tags
curl http://host.docker.internal:11434/api/tags
curl http://localhost:11434/api/tags

# Use whichever works
export OLLAMA_HOST=http://<working_ip>:11434
```

### Out of Memory (OOM) — `Killed (signal 9)`

```bash
# Check memory
free -h

# Use pre-built FAISS cache (Step 7) to skip embedding on-device
# Close other processes if possible
```

### `Illegal instruction (core dumped)` on import torch

```bash
# Wrong PyTorch build (x86 instead of ARM64). Reinstall:
pip3 uninstall torch -y
pip3 install --user --no-cache-dir torch --index-url https://developer.download.nvidia.com/compute/redist/jp/v60/
```

### EasyOCR CUDA Errors

```bash
# Force CPU mode
export EASYOCR_GPU=False
```

---

## Model Configuration Reference

| Parameter       | Value                         | Notes                           |
|-----------------|-------------------------------|---------------------------------|
| **Model**       | `llama3.2:1b`                 | ⛔ Only permitted model         |
| **API Endpoint**| `http://172.17.0.1:11434`     | Docker bridge to host Ollama    |
| **Context**     | 1024 tokens                   | Saves ~500 MB RAM               |
| **GPU Devices** | 1                             | Single Jetson GPU               |
| **Memory Map**  | `True`                        | Required for Jetson             |

> ⛔ **`llama3.2:1b` is the ONLY permitted model — no exceptions.** Do not pull or run any other model.

---

> **Built for EdgeMinds 2026 Internship** · Runs fully offline on NVIDIA Jetson Orin · Powered by Llama 3.2 1B via Ollama
