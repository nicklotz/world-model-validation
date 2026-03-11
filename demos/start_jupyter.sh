#!/bin/bash
# Start Jupyter + FiftyOne for remote demo access
# Usage: bash start_jupyter.sh

set -e

export PATH="$HOME/.local/bin:$PATH"

# FiftyOne: bind App to all interfaces so it's accessible via VM IP
export FIFTYONE_DEFAULT_APP_ADDRESS=0.0.0.0

# Kill any existing Jupyter/FiftyOne processes
pkill -f "jupyter-notebook" 2>/dev/null || true
pkill -f "jupyter-lab" 2>/dev/null || true
sleep 1

cd ~/world-model-validation-talk

# Start Jupyter notebook server
nohup env \
  PATH="$HOME/.local/bin:$PATH" \
  FIFTYONE_DEFAULT_APP_ADDRESS=0.0.0.0 \
  jupyter notebook \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --NotebookApp.token="worldmodel2026" \
    --NotebookApp.allow_origin="*" \
  > /tmp/jupyter.log 2>&1 &

JPID=$!
echo "Jupyter started with PID $JPID"
sleep 4

# Show access URLs
EXTERNAL_IP=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H "Metadata-Flavor: Google")
echo ""
echo "================================================"
echo "  Jupyter:    http://${EXTERNAL_IP}:8888/?token=worldmodel2026"
echo "  FiftyOne:   http://${EXTERNAL_IP}:5151"
echo "================================================"
echo ""
echo "Open the Jupyter URL, navigate to demos/, and run a notebook."
echo "When a cell calls fo.launch_app(), the FiftyOne App will be"
echo "accessible at the FiftyOne URL above."
echo ""
tail -3 /tmp/jupyter.log
