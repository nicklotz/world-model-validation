# Demo 1: Finding What Breaks — Embedding-Guided Failure Triage

**Hook:** You have thousands of model outputs. How do you find the broken ones without watching them all?

**What this demo shows:**
- Compute video embeddings and project to 2D (UMAP) for instant visual triage
- Failure modes cluster together in embedding space — lasso a cluster, find a bug category
- Text-based semantic search lets you query by *description*, not metadata

**Duration:** ~5 minutes | **Dataset:** `quickstart-video` (10 video segments — in production, this same workflow handles 100K+)

## 1. Load dataset & simulate validation scores

In production, these scores come from your physics validation pipeline.
Here we simulate them to show the triage workflow.


```python
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import numpy as np

# Load the quickstart-video dataset (10 video segments, instant)
dataset = foz.load_zoo_dataset("quickstart-video")
print(f"Loaded {len(dataset)} video samples")
print(dataset)
```


```python
# Simulate world model validation fields
# In practice these come from physics checks, drift detectors, etc.
np.random.seed(42)

failure_types = ["clean", "clean", "clean", "physics_violation", 
                 "temporal_drift", "object_hallucination", "clean",
                 "physics_violation", "clean", "temporal_drift"]

for i, sample in enumerate(dataset):
    ft = failure_types[i % len(failure_types)]
    
    if ft == "clean":
        sample["physics_score"] = np.random.uniform(0.85, 1.0)
        sample["drift_horizon"] = np.random.randint(40, 50)
    elif ft == "physics_violation":
        sample["physics_score"] = np.random.uniform(0.15, 0.4)
        sample["drift_horizon"] = np.random.randint(5, 15)
    elif ft == "temporal_drift":
        sample["physics_score"] = np.random.uniform(0.4, 0.6)
        sample["drift_horizon"] = np.random.randint(10, 25)
    else:  # object_hallucination
        sample["physics_score"] = np.random.uniform(0.3, 0.5)
        sample["drift_horizon"] = np.random.randint(15, 30)
    
    sample["failure_type"] = ft
    sample.save()

print("Added fields: physics_score, drift_horizon, failure_type")
print(f"Failure distribution: {dataset.count_values('failure_type')}")
```

## 2. Compute video embeddings

Embeddings map each video to a high-dimensional vector that captures its semantic content.
Videos that *look* similar (same scene type, same objects, same motion patterns) land
near each other in this space — including videos that fail in similar ways.


```python
# Compute embeddings using CLIP (robust, supports text+image search)
# CLIP is ideal here because sort_by_similarity with text queries
# requires a model that jointly embeds text and images.
model = "clip-vit-base32-torch"
model_name = "clip"
print(f"Using {model}")

# Compute and store embeddings + similarity index
fob.compute_similarity(
    dataset,
    model=model,
    brain_key=f"{model_name}_sim",
)
print("Similarity index built.")
```

## 3. UMAP visualization — the failure map

UMAP projects those high-dimensional embeddings down to 2D so you can *see*
the structure. Clusters = groups of semantically similar videos. Color by
`physics_score` and the failures light up.


```python
# Compute 2D UMAP projection for the Embeddings panel
# We reference the similarity index so it reuses those embeddings
fob.compute_visualization(
    dataset,
    brain_key=f"{model_name}_viz",
    embeddings=f"{model_name}_sim",
    method="umap",
    num_dims=2,
)
print("UMAP visualization computed.")
```

## 4. Launch the App — interactive triage

**What to do in the App:**
1. Open the **Embeddings panel** (icon in the top toolbar)
2. Select the visualization brain key (`clip_viz` or `qwen3-vl-embedding_viz`)
3. **Color by** `physics_score` — low scores (red) will cluster together
4. **Lasso** a red cluster — you've just found a failure category without watching a single video
5. Click individual samples to inspect the actual video

This is the core insight: **aggregate metrics say the model is broken. Embeddings show you *how* it's broken.**


```python
# Launch FiftyOne App
session = fo.launch_app(dataset, port=5151)
print("App launched at http://localhost:5151")
print("\n>> Open the Embeddings panel and color by 'physics_score'")
print(">> Lasso a cluster of low-scoring samples to isolate a failure mode")
```

## 5. Semantic search — query by description

Instead of filtering by metadata fields, you can search by *what the video looks like*.
This is powerful when you don't have pre-existing labels for the failure mode you're hunting.


```python
# Text-based semantic search: find videos matching a description
results = dataset.sort_by_similarity(
    "vehicles on road with many objects",
    brain_key=f"{model_name}_sim",
    k=5,
)

print(f"Top {len(results)} results for 'vehicles on road with many objects':")
for sample in results:
    print(f"  {sample.filepath.split('/')[-1]}  "
          f"physics={sample.physics_score:.2f}  "
          f"type={sample.failure_type}")

# Update the App view to show search results
session.view = results
```


```python
# Filter to just the failures — this is your review queue
from fiftyone import ViewField as F

failures = dataset.match(F("physics_score") < 0.5)
print(f"\nFailure queue: {len(failures)} samples need review")
print(f"Failure types: {failures.count_values('failure_type')}")

session.view = failures
print("\n>> App now shows only failures. Review them in the grid.")
```

## Takeaway

**Aggregate metrics say the model is broken. Embeddings show you *how* it's broken.**

With FiftyOne's embedding visualization + semantic search, you can:
- Triage thousands of model outputs in minutes, not hours
- Discover failure modes you didn't know to look for
- Build targeted review queues without manual labeling

This scales from 10 samples (this demo) to 100K+ samples (production).
