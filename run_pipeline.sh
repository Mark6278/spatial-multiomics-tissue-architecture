#!/usr/bin/env bash
# Fast execution script using global Python environment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect Python command
if command -v python &>/dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python is not installed."
    exit 1
fi

# Install dependencies globally only if scanpy or squidpy are missing
if ! $PYTHON_CMD -c "import scanpy, squidpy, mudata, skimage" &>/dev/null; then
    echo "Installing required libraries globally..."
    $PYTHON_CMD -m pip install --upgrade pip -q
    $PYTHON_CMD -m pip install -r requirements.txt
else
    echo "Core libraries already installed. Launching pipeline..."
fi

mkdir -p logs figures
$PYTHON_CMD src/pipeline.py "$@" | tee logs/pipeline_run.log
echo "Done! Figures saved to ./figures/"
