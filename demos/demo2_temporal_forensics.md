# Demo 2: Temporal Forensics — VLM-Powered Video Audit

**Hook:** A VLM can audit your generated video — what happens, when it starts, when it stops — in seconds per clip.

**What this demo shows:**
- Use a VLM (Qwen3-VL-2B) to automatically localize temporal events in video
- Evaluate VLM predictions against ground truth using temporal IoU
- Find disagreements: events the VLM missed (false negatives) or hallucinated (false positives)
- Create reviewable clips from disagreement regions

**Duration:** ~5 minutes | **Dataset:** ActivityNet-200 (6 samples) or quickstart-video fallback

## 1. Load dataset with temporal ground truth

ActivityNet provides temporal activity annotations: for each video, we know
*what* happens and *when* (start/end timestamps). This gives us ground truth
to evaluate VLM predictions against.


```python
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.video as fouv
from fiftyone import ViewField as F
import numpy as np

# Try ActivityNet first, fall back to quickstart-video
USE_ACTIVITYNET = True

try:
    dataset = foz.load_zoo_dataset(
        "activitynet-200",
        split="validation",
        max_samples=6,
    )
    print(f"Loaded ActivityNet-200: {len(dataset)} samples")
except Exception as e:
    print(f"ActivityNet unavailable ({e}), using quickstart-video with synthetic labels")
    USE_ACTIVITYNET = False
    dataset = foz.load_zoo_dataset("quickstart-video")

# Ensure metadata is computed (needed for frame_rate, total_frame_count)
dataset.compute_metadata()

if not USE_ACTIVITYNET:
    # Add synthetic temporal ground truth
    np.random.seed(42)
    for sample in dataset:
        duration = sample.metadata.total_frame_count / sample.metadata.frame_rate
        # Simulate 1-2 ground truth events per video
        events = []
        t1 = np.random.uniform(0.1, 0.3) * duration
        t2 = t1 + np.random.uniform(2, 5)
        events.append(fo.TemporalDetection(
            label="activity",
            support=[int(t1 * sample.metadata.frame_rate) + 1,
                     int(min(t2, duration) * sample.metadata.frame_rate) + 1],
        ))
        if np.random.random() > 0.4:
            t3 = np.random.uniform(0.5, 0.7) * duration
            t4 = t3 + np.random.uniform(1, 4)
            events.append(fo.TemporalDetection(
                label="interaction",
                support=[int(t3 * sample.metadata.frame_rate) + 1,
                         int(min(t4, duration) * sample.metadata.frame_rate) + 1],
            ))
        sample["ground_truth_events"] = fo.TemporalDetections(detections=events)
        sample.save()

print(dataset)
```

## 2. Run VLM temporal localization

We ask Qwen3-VL-2B two questions per video:
1. **Temporal localization:** "What events happen, and when do they start/stop?"
2. **Description:** "Describe what happens in this video."

The VLM acts as an automated auditor — like having a junior annotator who
never gets tired, watching every video at machine speed.


```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# Load Qwen3-VL-2B for temporal analysis
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    vlm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    VLM_AVAILABLE = True
    print(f"Loaded {MODEL_ID}")
except Exception as e:
    print(f"VLM not available ({e}), using simulated predictions")
    VLM_AVAILABLE = False
```


```python
import json
from decord import VideoReader, cpu

def extract_frames(video_path, num_frames=8):
    """Extract evenly-spaced frames from a video."""
    vr = VideoReader(video_path, ctx=cpu(0))
    indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    return frames, indices

def run_vlm_temporal(sample, processor, vlm_model):
    """Run VLM temporal event localization on a single video sample."""
    frames, frame_indices = extract_frames(sample.filepath, num_frames=8)
    duration = sample.metadata.total_frame_count / sample.metadata.frame_rate
    fps = sample.metadata.frame_rate
    
    # Build prompt for temporal localization
    prompt = (
        f"This video is {duration:.1f} seconds long at {fps:.0f} fps. "
        f"List each distinct event or activity. For each, provide: "
        f"event name, start time (seconds), end time (seconds). "
        f"Format as JSON: [{{\"label\": ..., \"start\": ..., \"end\": ...}}]"
    )
    
    # Process with VLM
    from PIL import Image
    images = [Image.fromarray(f) for f in frames]
    inputs = processor(
        text=prompt,
        images=images,
        return_tensors="pt",
    ).to(vlm_model.device)
    
    with torch.no_grad():
        output_ids = vlm_model.generate(**inputs, max_new_tokens=512)
    
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return response, duration, fps

def parse_vlm_events(response, duration, fps):
    """Parse VLM output into TemporalDetections."""
    detections = []
    try:
        # Try to extract JSON from the response
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start >= 0 and json_end > json_start:
            events = json.loads(response[json_start:json_end])
            for evt in events:
                start_frame = int(float(evt.get('start', 0)) * fps) + 1
                end_frame = int(min(float(evt.get('end', duration)), duration) * fps) + 1
                if end_frame > start_frame:
                    detections.append(fo.TemporalDetection(
                        label=str(evt.get('label', 'event')).lower().strip(),
                        support=[start_frame, end_frame],
                    ))
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return detections

print("Helper functions defined.")
```


```python
# Run VLM on each sample (or simulate if VLM unavailable)
np.random.seed(99)
for sample in dataset:
    print(f"Processing: {sample.filepath.split('/')[-1]}...")
    
    if VLM_AVAILABLE:
        response, duration, fps = run_vlm_temporal(sample, processor, vlm_model)
        vlm_dets = parse_vlm_events(response, duration, fps)
        sample["vlm_description"] = response
    else:
        # Simulate VLM predictions with some intentional errors
        fps = sample.metadata.frame_rate
        duration = sample.metadata.total_frame_count / fps
        vlm_dets = []
        
        # Slightly offset from ground truth (simulates imperfect VLM)
        if sample.has_field("ground_truth_events") and sample.ground_truth_events:
            for gt in sample.ground_truth_events.detections:
                # 70% chance: detect with temporal offset
                if np.random.random() < 0.7:
                    offset = np.random.uniform(-0.5, 1.5) * fps
                    vlm_dets.append(fo.TemporalDetection(
                        label=gt.label,
                        support=[
                            max(1, int(gt.support[0] + offset)),
                            int(gt.support[1] + offset * 0.5),
                        ],
                    ))
            # 30% chance: hallucinate an extra event
            if np.random.random() < 0.3:
                t_start = int(np.random.uniform(0.6, 0.8) * duration * fps) + 1
                t_end = t_start + int(np.random.uniform(1, 3) * fps)
                vlm_dets.append(fo.TemporalDetection(
                    label="hallucinated_event",
                    support=[t_start, min(t_end, int(duration * fps))],
                ))
        
        sample["vlm_description"] = "(Simulated — VLM not available)"
    
    sample["vlm_events"] = fo.TemporalDetections(detections=vlm_dets)
    sample.save()

print(f"\nProcessed {len(dataset)} videos")
print(f"Total VLM events detected: {dataset.count('vlm_events.detections')}")
```

## 3. Evaluate: temporal IoU

Now we compare VLM predictions against ground truth using **temporal IoU** —
the same metric used in ActivityNet benchmarks. This tells us:
- **True positives:** events the VLM correctly found
- **False negatives:** events the VLM missed
- **False positives:** events the VLM hallucinated


```python
# Determine ground truth field name
gt_field = "ground_truth_events" if not USE_ACTIVITYNET else "temporal_detections"

# Check which field actually has ground truth
if not dataset.has_sample_field(gt_field):
    # ActivityNet may use different field names
    possible_fields = [f for f in dataset.get_field_schema() 
                       if "temporal" in f.lower() or "activity" in f.lower() or "detection" in f.lower()]
    if possible_fields:
        gt_field = possible_fields[0]
    else:
        gt_field = "ground_truth_events"

print(f"Ground truth field: {gt_field}")
print(f"Prediction field: vlm_events")
```


```python
# Evaluate with temporal IoU (ActivityNet-style evaluation for temporal detections)
results = dataset.evaluate_detections(
    "vlm_events",
    gt_field=gt_field,
    eval_key="temporal_eval",
)

# Print aggregate metrics
results.print_report()
print(f"\nmAP: {results.mAP():.3f}")
```

## 4. Find disagreements

The most valuable output isn't the mAP number — it's *where* the VLM disagrees
with ground truth. These disagreement regions are exactly what a human reviewer
needs to look at.


```python
# Find disagreement samples
# After evaluation, each detection gets a per-label eval field.
# We look for samples that have any false positives or false negatives.
from collections import Counter

missed_ids = []
hallucinated_ids = []

for sample in dataset:
    # Check ground truth detections for false negatives (unmatched GT)
    gt = sample[gt_field]
    if gt is not None:
        for det in gt.detections:
            if hasattr(det, "temporal_eval") and det.temporal_eval == "fn":
                missed_ids.append(sample.id)
                break
    
    # Check predictions for false positives (unmatched predictions)
    preds = sample["vlm_events"]
    if preds is not None:
        for det in preds.detections:
            if hasattr(det, "temporal_eval") and det.temporal_eval == "fp":
                hallucinated_ids.append(sample.id)
                break

missed = dataset.select(missed_ids) if missed_ids else dataset.limit(0)
hallucinated = dataset.select(hallucinated_ids) if hallucinated_ids else dataset.limit(0)

print(f"Samples with missed events (FN): {len(missed)}")
print(f"Samples with hallucinated events (FP): {len(hallucinated)}")

# All disagreement samples
all_disagree_ids = list(set(missed_ids + hallucinated_ids))
disagreements = dataset.select(all_disagree_ids) if all_disagree_ids else dataset
print(f"\nTotal disagreement samples: {len(disagreements)}")
print("These are your priority review queue.")
```

## 5. Launch the App — timeline visualization

**What to do in the App:**
1. Look at the video player — you'll see **dual timeline bars**:
   - Green: ground truth events
   - Blue: VLM predictions
2. Spot disagreements visually — where bars don't overlap
3. Click on a disagreement region to jump to that timestamp


```python
# Launch App showing disagreement samples
session = fo.launch_app(disagreements, port=5151)
print("App launched with disagreement samples")
print("\n>> Look for misaligned timeline bars (green=GT, blue=VLM)")
print(">> Click a sample to see the video with temporal annotations")
```

## 6. Create reviewable clips from disagreement regions

`to_clips()` is the power move: it converts the full video dataset into a
clip-level dataset where each clip corresponds to one VLM-predicted event.
Now reviewers see *just* the relevant segment, not the whole video.


```python
# Create clip-level view from VLM events
clips = dataset.to_clips("vlm_events")

print(f"Created {len(clips)} reviewable clips from VLM events")
for clip in clips:
    label = clip.vlm_events.label
    support = clip.support
    print(f"  [{support[0]:>5d} - {support[1]:>5d}] {label}")

# Show clips in the App
session.view = clips
print("\n>> App now shows individual clips for review")
```

## Takeaway

**A validation pipeline you can build in an afternoon, not a quarter.**

What we just built:
1. VLM-powered temporal event localization (automated "what happens when")
2. Quantitative evaluation against ground truth (temporal IoU / mAP)
3. Automatic disagreement detection (missed events + hallucinations)
4. Clip-level review queue (reviewers see only what matters)

This same pipeline works for auditing world model outputs:
- Replace ActivityNet with your generated video
- Ground truth = what *should* happen given the action input
- VLM = automated auditor checking what *actually* happens
- Disagreements = where your world model is wrong
