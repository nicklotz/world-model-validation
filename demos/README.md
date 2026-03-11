# Demo Setup & Run Instructions

## Hardware Requirements

- **GPU:** NVIDIA L4 (24 GB) or T4 (16 GB) minimum
- **RAM:** 32 GB recommended (n1-standard-8)
- **Disk:** 200 GB (models + datasets)
- **Tested on:** GCP `n1-standard-8` + 1x L4, PyTorch DLVM image

## Quick Start

```bash
# 1. Run setup (downloads models + datasets, ~10 min first time)
bash demos/setup.sh

# 2. Launch Jupyter
jupyter notebook --no-browser --port=8888

# 3. In a separate terminal, launch FiftyOne App
#    (or let the notebook launch it automatically)
fiftyone app launch --port 5151
```

## Demo 1: Embedding-Guided Failure Triage

**File:** `demo1_embedding_triage.ipynb`
**Duration:** ~5 minutes
**Placement:** Between slides 8 and 9

Shows how to use embeddings + UMAP to triage 10K model outputs without watching them all. Uses `quickstart-video` dataset with simulated physics scores.

### Key APIs used
- `fiftyone.brain.compute_visualization()` — UMAP projection
- `fiftyone.brain.compute_similarity()` — similarity index
- `dataset.sort_by_similarity()` — text-based semantic search
- Embeddings panel in FiftyOne App

### Fallback if Qwen3-VL fails
Switch to CLIP embeddings:
```python
fob.compute_visualization(dataset, model="clip-vit-base32-torch", brain_key="clip_viz")
```

## Demo 2: Temporal Forensics — VLM-Powered Video Audit

**File:** `demo2_temporal_forensics.ipynb`
**Duration:** ~5 minutes
**Placement:** Between slides 13 and 14

Demonstrates using a VLM to audit generated video for temporal events, then evaluating against ground truth with ActivityNet-style temporal IoU.

### Key APIs used
- `fo.utils.transformers.apply_model()` — VLM inference
- `dataset.evaluate_detections()` — temporal IoU evaluation
- `dataset.to_clips()` — create reviewable clips from events
- Timeline visualization in FiftyOne App

### Fallback if ActivityNet unavailable
The notebook automatically falls back to `quickstart-video` with synthetic temporal labels.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Restart kernel, reduce batch size, or use `torch.cuda.empty_cache()` |
| FiftyOne App won't launch | Run `fiftyone app launch --port 5151` in a separate terminal |
| Model download hangs | Check internet connection; models are ~4 GB each |
| `ModuleNotFoundError` | Re-run `bash demos/setup.sh` |
| ActivityNet download fails | Demo 2 auto-falls back to quickstart-video |
