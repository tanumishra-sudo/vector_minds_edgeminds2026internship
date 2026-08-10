#!/bin/bash
# ============================================================
# Mentora - One-Command Startup Script
# ============================================================

# ── 1. Kill any existing instances ──
pkill -9 -f ngrok 2>/dev/null || true
pkill -9 -f streamlit 2>/dev/null || true

# ── 2. Configure Environment ──
mkdir -p ~/tmp ~/.cache/pip
export TMPDIR=$HOME/tmp
export PIP_CACHE_DIR=$HOME/.cache/pip
export PATH="$HOME/.local/bin:$PATH"

# ── 3. Navigate to Project Directory ──
PROJECT_DIR="$HOME/vector_minds_edgeminds2026internship"
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
elif [ -d ".git" ]; then
    PROJECT_DIR="$(pwd)"
else
    git clone https://github.com/tanumishra-sudo/vector_minds_edgeminds2026internship.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# ── 4. Pull Latest Code ──
git pull origin main 2>/dev/null || true

# ── 4b. Ensure Python Dependencies ──
pip3 install --user --no-cache-dir tzdata pytz -r requirements.txt pyngrok >/dev/null 2>&1 || true

# ── 5. Ensure Ollama Service is Active ──
if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama service..."
    ollama serve >/dev/null 2>&1 &
    sleep 4
fi

# ── 6. Start ngrok Tunnel in Background (silent logging) ──
python3 -c "from pyngrok import ngrok; ngrok.set_auth_token('3HiI2mchF3YRFEwXvZxr7DcCTB8_4iaN1k6yZhVdMWv32BXTh'); t = ngrok.connect('127.0.0.1:8501'); print('\n========================================\n🌐 YOUR NGROK PUBLIC URL:\n', t.public_url, '\n========================================\n', flush=True); ngrok.get_ngrok_process().proc.wait()" > /tmp/ngrok.log 2>&1 &

sleep 3

# ── 7. Display Public URL ──
echo ""
echo "========================================"
echo "🌐 YOUR NGROK PUBLIC URL:"
python3 -c "from pyngrok import ngrok; tunnels = ngrok.get_tunnels(); print(tunnels[0].public_url if tunnels else 'Tunnel initializing...')" 2>/dev/null
echo "========================================"
echo ""

# ── 8. Launch Streamlit Server ──
echo "🚀 Launching Streamlit App..."
streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.enableCORS false --server.enableXsrfProtection false
