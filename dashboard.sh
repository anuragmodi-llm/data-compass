#!/bin/bash

set -e
source "$(dirname "$0")/venv/bin/activate"

# ============================================================
# dashboard.sh — Start the monitoring dashboard on port 3001
# Dependency check runs every time to catch missing packages
# ============================================================

DASHBOARD_PORT=3001
BACKEND_URL="http://localhost:8000"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_DIR="$SCRIPT_DIR/dashboard"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Document Classifier — Dashboard        ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Dependency check ──────────────────────────────────────────
echo "▶ Checking dependencies..."

# Python 3
if ! command -v python3 &>/dev/null; then
  echo "✗ python3 not found. Install Python 3 first."
  exit 1
fi
echo "  ✓ python3 $(python3 --version 2>&1 | awk '{print $2}')"

# pip packages
MISSING_PKGS=()
for pkg in supabase python-dotenv; do
  if ! python3 -c "import ${pkg//-/_}" &>/dev/null; then
    MISSING_PKGS+=("$pkg")
  fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
  echo "  ✗ Missing packages: ${MISSING_PKGS[*]}"
  echo "  Installing..."
  pip3 install "${MISSING_PKGS[@]}" --quiet
  echo "  ✓ Packages installed"
else
  echo "  ✓ All Python packages present"
fi

# .env file
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  echo ""
  echo "✗ .env file not found at: $SCRIPT_DIR/.env"
  echo "  Create it with:"
  echo "    SUPABASE_URL=https://your-project.supabase.co"
  echo "    SUPABASE_SERVICE_ROLE_KEY=your-key"
  exit 1
fi
echo "  ✓ .env present"

# Backend reachability (warn only — don't block dashboard from starting)
if ! curl -s --max-time 2 "$BACKEND_URL/docs" &>/dev/null; then
  echo ""
  echo "  ⚠ Backend not reachable at $BACKEND_URL"
  echo "    Dashboard will load but live data won't appear until ./start.sh is running."
  echo ""
fi

# ── Port cleanup ───────────────────────────────────────────────
echo ""
echo "▶ Checking port $DASHBOARD_PORT..."

EXISTING=$(lsof -ti tcp:$DASHBOARD_PORT 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
  echo "  Killing existing process on port $DASHBOARD_PORT..."
  kill -9 $EXISTING 2>/dev/null || true
  sleep 0.5
fi
echo "  ✓ Port $DASHBOARD_PORT free"

# ── Start dashboard ────────────────────────────────────────────
echo ""
echo "▶ Starting dashboard..."
echo ""
echo "  Dashboard →  http://localhost:$DASHBOARD_PORT"
echo "  Backend   →  $BACKEND_URL  (must be running separately)"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

cd "$DASHBOARD_DIR"
python3 -m http.server $DASHBOARD_PORT
