# Analyzer vs Impact Point — Comparison & Recommended Approach

## The Core Difference (TL;DR)

| Aspect | [analyzer.py](file:///d:/CricVision_Scratch/analyzer.py) (More Accurate) | [impact_point.py](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py) (Less Accurate) |
|---|---|---|
| **Architecture** | Stateful, frame-by-frame processing | Batch post-hoc analysis on full path |
| **Impact detection** | Angle-change counting (max 2) with distance gating | 4 independent sub-detectors, earliest-wins |
| **Pitch detection** | Built-in, with strict `>` check | Delegated to [pitch_point.py](file:///d:/CricVision_Scratch/ball_tracking/pitch_point.py) (uses `>=`, weaker) |
| **Case classification** | 6 classified cases with full-toss guards | No case classification, just returns a point |
| **Noise handling** | 40px min-gap between angle changes | Segment filters commented out ⚠️ |

---

## Why [analyzer.py](file:///d:/CricVision_Scratch/analyzer.py) Is More Accurate

### 1. Single unified angle-change model

[analyzer.py](file:///d:/CricVision_Scratch/analyzer.py) uses **one algorithm**: track up to **2 major angle changes** (≥25°) with a minimum 40px gap between them. Then in [evaluate_event()](file:///d:/CricVision_Scratch/analyzer.py#149-326):

- **2 angle changes + pitch** → 1st = pitch confirmation, 2nd = impact (Case 1)
- **2 angle changes, no pitch** → 1st = impact (Case 2)
- **1 angle change + pitch** → compare distance to decide
- **0 angle changes** → missed ball (Case 3)

This is elegant because the **same signal** (angle change) is used for both pitch and impact, and the **case logic** disambiguates them.

### 2. Frame-by-frame processing prevents look-ahead bias

[analyzer.py](file:///d:/CricVision_Scratch/analyzer.py) processes each frame incrementally. It detects the pitch as soon as `curr.y > prev.y AND curr.y > next.y` (strict `>`), and locks it (only first pitch is kept). This means **later noisy Y-peaks can't overwrite the true pitch**.

### 3. Full-toss guard built into evaluation

When Case 1 fires (pitch + 2 angle changes), [analyzer.py](file:///d:/CricVision_Scratch/analyzer.py) checks if the pitch point is actually just the bat impact in disguise:

```
if dist_to_second <= 40px → it's actually a Full Toss (Case 1→2 downgrade)
```

[impact_point.py](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py) has no equivalent guard — it trusts whatever [pitch_point.py](file:///d:/CricVision_Scratch/ball_tracking/pitch_point.py) gives it.

### 4. Minimum distance gating

The 40px minimum gap (`MIN_ANGLE_CHANGE_DISTANCE_PX`) between angle-change points prevents frame-to-frame noise from creating a false second deflection right next to the first. [impact_point.py](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py)'s segment filters are **commented out**, so jitter can produce false positives.

---

## Why [impact_point.py](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py) Is Less Accurate

### 1. Too many competing sub-detectors

It runs **4 independent methods** and picks the earliest:

| Sub-detector | What it does | Problem |
|---|---|---|
| [_turning_point](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py#22-47) | Y-reversal | Often fires at the **pitch** point, not bat contact |
| [_consecutive_change](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py#49-71) | Angle > 30° | Good idea, but no min-gap → noise sensitive |
| [_anchored_change](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py#73-107) | Cumulative drift > 45° | Too broad, often fires late or on wrong point |
| bat-bbox point | Ball inside bat | Good direct evidence but used alongside conflicting candidates |

The [_pick_earliest](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py#109-133) strategy is the fundamental flaw: the **earliest** ≠ the **most correct**. On a standard delivery, the pitch bounce often triggers [_turning_point](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py#22-47) or [_consecutive_change](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py#49-71) *before* the actual bat impact.

### 2. Segment filters are disabled

Lines 38-39, 63-64, 91-92, 95-96 — all the `min_seg` / `max_seg` guards are commented out. This means:
- Sub-pixel jitter (< 4px movements) triggers false detections
- Large detection jumps (> 150px) aren't filtered

### 3. No case classification

[impact_point.py](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py) returns a single point or `False` — it has no concept of "what kind of delivery was this?". So it can't reason about full-toss vs standard bounce, and can't apply delivery-type-specific logic.

### 4. Weak full-toss handling

For full-toss (no pitch), it only runs [_consecutive_change](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py#49-71) + bat-bbox. The [_turning_point](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py#22-47) and [_anchored_change](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py#73-107) are skipped. This is actually *correct* in principle, but without the case-classification framework of [analyzer.py](file:///d:/CricVision_Scratch/analyzer.py), the result has no context.

---

## Recommended Approach

> [!IMPORTANT]
> The best approach is to **bring [analyzer.py](file:///d:/CricVision_Scratch/analyzer.py)'s angle-change counting model into [impact_point.py](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py)**, rather than trying to fix the 4-sub-detector approach.

### Strategy: Replace the multi-detector approach with analyzer's 2-angle-change model

Here's the concrete plan:

#### Step 1 — Port the core angle-change detection

Replace the 4 sub-detectors with a single function that walks the `post_pitch` points and finds up to **2 major angle changes** (≥ 25°), enforcing a **40px minimum gap** between them:

```python
def _find_angle_changes(
    points: list[tuple[int, int]],
    threshold_deg: float = 25.0,
    min_gap_px: float = 40.0,
) -> list[tuple[int, int]]:
    """Find up to 2 major angle-change points (analyzer.py's model)."""
    changes = []
    for i in range(1, len(points) - 1):
        v1 = (points[i][0] - points[i-1][0], points[i][1] - points[i-1][1])
        v2 = (points[i+1][0] - points[i][0], points[i+1][1] - points[i][1])
        angle = _angle_deg(v1, v2)
        if angle >= threshold_deg:
            if not changes or _dist(points[i], changes[-1]) >= min_gap_px:
                changes.append(points[i])
                if len(changes) == 2:
                    break
    return changes
```

#### Step 2 — Add case-based evaluation in [find_impact_point](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py#137-195)

Instead of "earliest wins", use the **number of angle changes + pitch presence** to decide:

```
has_pitch + 2 angle changes → impact = 2nd angle change (with full-toss guard)
has_pitch + 1 angle change  → impact = that angle change (if far from pitch)
no pitch  + angle changes   → impact = 1st angle change
no pitch  + no changes      → impact = bat-bbox point (fallback)
```

#### Step 3 — Re-enable segment filters

Uncomment the `min_seg` / `max_seg` guards in the remaining helper functions, or apply them in the new `_find_angle_changes` function.

#### Step 4 — Keep the bat-bbox as a **fallback**, not a primary candidate

The bat-bbox containment is valuable direct evidence, but it should be used only when the angle-change model produces **no result** — not as an equal-weight candidate.

### What NOT to do

- ❌ Don't try to "tune the thresholds" on the existing 4-sub-detector approach — the architecture is fundamentally flawed
- ❌ Don't replace [impact_point.py](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py) entirely with [analyzer.py](file:///d:/CricVision_Scratch/analyzer.py) — [analyzer.py](file:///d:/CricVision_Scratch/analyzer.py) is stateful (frame-by-frame) while [impact_point.py](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py) is called once on the full path; both patterns are valid, but the orchestrator currently expects the batch API
- ❌ Don't remove the bat-bbox logic entirely — it's genuinely useful as a fallback/tiebreaker

---

## Verification

Since there are no automated tests in this project, the best way to verify is:

1. Run the pipeline on the same test videos before and after changes
2. Compare the printed `Impact point:` output
3. Check the output video to visually confirm the impact marker lands at the correct frame
