#!/bin/bash
# Double-click this file in Finder to launch AlgoForge Strategy Lab
cd "$(dirname "$0")"

# Activate venv
source venv/bin/activate

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "  ⚠  ANTHROPIC_API_KEY is not set."
  echo "  AI translation will not work until you run:"
  echo "  export ANTHROPIC_API_KEY=sk-ant-..."
  echo ""
fi

echo ""
echo "  AlgoForge Strategy Lab"
echo "  → http://localhost:5001"
echo ""

# Open browser after 1s
(sleep 1 && open http://localhost:5001) &

python app.py
