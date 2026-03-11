#!/usr/bin/env bash
# setup.sh — Pre-warm models and datasets for the live demos
# Run on GPU VM before the talk to avoid download delays during presentation
set -euo pipefail

echo "=== World Model Validation Talk — Demo Setup ==="
echo "This script pre-downloads models and datasets so demos run instantly."
echo ""

# ── 1. Python dependencies ──────────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
pip install -q fiftyone torch torchvision transformers decord qwen-vl-utils pillow

# ── 2. Pre-download FiftyOne datasets ───────────────────────────────
echo "[2/4] Pre-downloading FiftyOne datasets..."
python3 -c "
import fiftyone.zoo as foz

# Demo 1: quickstart-video (small, instant)
print('  Downloading quickstart-video...')
foz.load_zoo_dataset('quickstart-video', drop_existing=True)

# Demo 2: activitynet-200 validation split (subset)
# Fallback: if ActivityNet download fails, demo 2 uses quickstart-video
print('  Downloading activitynet-200 (validation, max 20 samples)...')
try:
    foz.load_zoo_dataset(
        'activitynet-200',
        split='validation',
        max_samples=20,
        drop_existing=True,
    )
    print('  ActivityNet loaded successfully.')
except Exception as e:
    print(f'  ActivityNet download failed: {e}')
    print('  Demo 2 will use quickstart-video with synthetic temporal labels.')
"

# ── 3. Pre-download Qwen3-VL models ────────────────────────────────
echo "[3/4] Pre-caching Qwen3-VL models..."
python3 -c "
from transformers import AutoModel, AutoTokenizer, AutoProcessor

models = [
    'Qwen/Qwen3-VL-2B-Instruct',
]

for model_name in models:
    print(f'  Downloading {model_name}...')
    try:
        AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f'  {model_name} cached.')
    except Exception as e:
        print(f'  Warning: {model_name} failed: {e}')

# Embedding model
print('  Downloading Qwen3-VL-Embedding-2B...')
try:
    from transformers import AutoModel
    AutoModel.from_pretrained('Qwen/Qwen3-VL-Embedding-2B', trust_remote_code=True)
    print('  Embedding model cached.')
except Exception as e:
    print(f'  Warning: embedding model failed: {e}')
    print('  Demo 1 will fall back to CLIP embeddings.')
"

# ── 4. Verify GPU availability ──────────────────────────────────────
echo "[4/4] Verifying GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  GPU: {gpu} ({mem:.1f} GB)')
else:
    print('  WARNING: No GPU detected! Demos will be slow.')
"

echo ""
echo "=== Setup complete ==="
echo "Run the demos with: jupyter notebook --no-browser --port=8888"
