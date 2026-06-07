#!/bin/bash

# ── Paths ────────────────────────────────────────────────────────────────
source "$(dirname "$0")/venv/bin/activate"
export PATH="$(python3 -m site --user-base)/bin:$PATH"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "🚀 Starting Document Classifier..."
echo ""

# ── Kill any existing process on port 8000 (Python only) ─────────────────
PIDS=$(lsof -ti:8000)
if [ -n "$PIDS" ]; then
  for PID in $PIDS; do
    NAME=$(ps -p $PID -o comm= 2>/dev/null)
    if [[ "$NAME" == *"python"* ]]; then
      kill -9 $PID 2>/dev/null
    fi
  done
  sleep 1
fi

# ── Kill any existing process on port 3000 ───────────────────────────────
lsof -ti:3000 | xargs kill -9 2>/dev/null
sleep 1

# ── Start backend ─────────────────────────────────────────────────────────
cd "$PROJECT_DIR"
uvicorn app.main:app --port 8000 &
BACKEND_PID=$!

# ── Wait for backend to be ready ──────────────────────────────────────────
echo "⏳ Waiting for backend..."
for i in {1..30}; do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend ready"
    break
  fi
  sleep 1
done

# ── Start frontend ────────────────────────────────────────────────────────
cd "$PROJECT_DIR/frontend"
python3 -m http.server 3000 &
FRONTEND_PID=$!

echo ""
echo "✅ Document Classifier is running"
echo "👉 Open this in your browser: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop everything"
echo ""

# ── Keep running until Ctrl+C ─────────────────────────────────────────────
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'; exit" SIGINT SIGTERM
wait
