# CricVision — Time-Complexity Optimizations & Professional Logging

Optimize the CricVision video-processing pipeline for speed and memory, **without changing any detection/analysis logic**. Replace all scattered `print()` calls with a production-grade `logging` system that writes to a rotating log file for server deployment.

---

## Proposed Changes

### Logging System

#### [NEW] [logger.py](file:///d:/CricVision_Scratch/ball_tracking/logger.py)

Centralised logging configuration module. Features:

- Creates a named logger `"CricVision"` used by every module
- Configures two handlers:
  - **Console** (`StreamHandler`) — `INFO` level, concise format
  - **File** (`RotatingFileHandler`) — `DEBUG` level, appends to `logs/cricvision.log` with rotation (5 MB × 5 backups)
- Structured log format: `[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s`
- Exposes `get_logger(name)` factory so each module gets a child logger (e.g. `CricVision.orchestrator`)

---

#### [MODIFY] [main.py](file:///d:/CricVision_Scratch/main.py)

- Replace all `print()` with `logger.info()` / `logger.warning()`
- Add a startup banner logging session metadata (timestamp, input folder, output folder)
- Add per-video structured summary log

---

#### [MODIFY] [orchestrator.py](file:///d:/CricVision_Scratch/ball_tracking/orchestrator.py)

**Logging:**
- Replace all `print()` with `logger.info()` / `logger.debug()`
- Add timing instrumentation: log elapsed time for Pass 1 (detection), computation, and Pass 2 (writing)

**Time-complexity / performance optimisations (logic unchanged):**

| Optimisation | Before | After | Impact |
|---|---|---|---|
| **Eliminate in-memory frame store** | All annotated frames stored in a Python `list` (O(N) memory — e.g. 3000 frames × 6 MB each ≈ 18 GB) | Write annotated frames to a **temporary video file** in Pass 1, then replay from disk in Pass 2 | Memory: O(N) → O(1). Huge for long videos |
| **Remove duplicate computation** | [compute_full_path](file:///d:/CricVision_Scratch/ball_tracking/ball_path.py#97-126) and [_find_impact_frame_idx](file:///d:/CricVision_Scratch/ball_tracking/orchestrator.py#17-39) called twice (lines 135-136 then 139-140) | Call once, conditioned on the fulltoss branch | ~2× faster for these functions |
| **[_find_impact_frame_idx](file:///d:/CricVision_Scratch/ball_tracking/orchestrator.py#17-39) — O(n) dict scan** | Linear scan over `frame_ball_map` dict | Short-circuit on exact match; keep existing fallback | Best-case O(1) |

---

#### [MODIFY] [impact_point.py](file:///d:/CricVision_Scratch/ball_tracking/impact_point.py)

- Replace all `print()` with `logger.debug()` / `logger.info()`
- **Remove hot-loop `print`** on line 49 (`print(f"triplet: …")`). This fires on every single triplet (~hundreds of times per video) and is the single biggest log-spam offender. Convert to `logger.debug()` so it's still available at DEBUG level but won't fire in normal runs.

---

#### [MODIFY] [ball_utils.py](file:///d:/CricVision_Scratch/ball_tracking/ball_utils.py)

- Replace `print()` with `logger.debug()`

---

#### [MODIFY] [pitch_point.py](file:///d:/CricVision_Scratch/ball_tracking/pitch_point.py)

- Fix the loop range: currently iterates `for i in range(0, …)` but reads `ball_path_points[i - 1]` on the first iteration (i=0 → reads index -1, which silently wraps to the last element). Change start to `range(1, …)` to match the intended triplet logic.

> [!CAUTION]
> This is a **bug fix**, not a logic change. The current code accidentally reads the *last* element on its first iteration. The fix changes the loop start from `0` to `1` so it reads proper consecutive triplets. This should not change results for paths > 3 points because the heuristic is looking for a Y-peak pattern that is unlikely to match the wrap-around, but it is technically a subtle behavioral change worth noting.

---

#### [MODIFY] [infer.py](file:///d:/CricVision_Scratch/models/infer.py)

- Replace `print()` in [load_model](file:///d:/CricVision_Scratch/models/infer.py#22-30) with `logger.info()`

---

#### [MODIFY] [detections.py](file:///d:/CricVision_Scratch/ball_tracking/detections.py)

- Minor: no `print()` calls to replace, but the [detect_all](file:///d:/CricVision_Scratch/ball_tracking/detections.py#48-54) function calls [detect_stump](file:///d:/CricVision_Scratch/ball_tracking/detections.py#42-46) even though `DETECT_STUMP = False`. Skip disabled detectors early by checking the flag inline.

---

## Verification Plan

### Manual Verification

Since there are no automated tests in this project, verification will be manual:

1. **Run the pipeline** on the same input video(s) you currently use:
   ```
   cd d:\CricVision_Scratch
   python main.py
   ```
2. **Check log output** — confirm structured logs appear on console and in `logs/cricvision.log`
3. **Compare output video** — open the new output `.mp4` and verify it looks identical to a previous run (same ball path, same freeze-frame animation, same annotations)
4. **Check memory** — for longer videos, observe that RAM usage stays roughly constant during processing instead of growing proportionally to frame count

> [!IMPORTANT]
> I'll need you to do a visual comparison of the output video since the pipeline relies on ML model inference and video output — there's no automated way to diff these. Please confirm the output looks the same after the changes.
