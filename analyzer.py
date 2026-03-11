"""
analyzer
========
Stateful frame-by-frame analyser that identifies the **pitch point** and
**bat impact point** from ball-tracking data in a cricket video.

Processing flow
───────────────
1. Call ``process_frame(ball_pos, bat_bbox)`` for every frame once the
   ball is first detected.
2. After the trajectory is complete (ball lost / path locked), call
   ``evaluate_event()`` to get the final result.

Evaluation cases (checked in order):
  • Case 1 — Pitch and Hit  (standard bounce)
  • Case 2 — No Pitch       (full-toss / direct hit)
  • Case 3 — Missed Ball
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass

from .utils import compute_velocity_vector, compute_deflection_angle

log: logging.Logger = logging.getLogger(__name__)

# ── Default Thresholds ───────────────────────────────────────────────────────

DEFAULT_PIXEL_TOLERANCE: float = 40.0
DEFAULT_ANGLE_CHANGE_THRESHOLD_DEG: float = 25.0
MIN_ANGLE_CHANGE_DISTANCE_PX: float = 40.0
PITCH_MATCH_TOLERANCE_PX: float = 40.0


# ── Result Container ─────────────────────────────────────────────────────────

@dataclass
class AnalyzerResult:
    """Immutable result returned by ``BallTrackerAnalyzer.evaluate_event()``."""

    case: int                                        # 1, 2, or 3
    case_label: str                                  # human-readable label
    pitch_point: tuple[int, int] | None = None
    impact_point: tuple[int, int] | None = None
    is_full_toss: bool = False
    missed_by_batsman: bool = False
    detail: str = ""


# ── BallTrackerAnalyzer ──────────────────────────────────────────────────────

class BallTrackerAnalyzer:
    """Stateful analyser populated frame-by-frame, evaluated once at the end.

    Parameters
    ----------
    pixel_tolerance : float
        Maximum Euclidean distance (px) for two points to be considered
        "matching" when comparing the pitch point against angle-change
        points (Case 1).
    angle_change_threshold_deg : float
        Minimum deflection angle (degrees) for a trajectory point to be
        recorded as a *major* angle change.
    """

    def __init__(
        self,
        pixel_tolerance: float = DEFAULT_PIXEL_TOLERANCE,
        angle_change_threshold_deg: float = DEFAULT_ANGLE_CHANGE_THRESHOLD_DEG,
    ) -> None:
        # ── Configurable thresholds ──────────────────────────────────────
        self.pixel_tolerance: float = pixel_tolerance
        self.angle_change_threshold_deg: float = angle_change_threshold_deg
        self.MIN_ANGLE_CHANGE_DISTANCE_PX: float = MIN_ANGLE_CHANGE_DISTANCE_PX

        # ── State variables (§1 of spec) ────────────────────────────────
        # Ball coordinates that fall strictly inside the bat bbox.
        self.bat_bbox_balls: list[tuple[int, int]] = []

        # Up to 2 points where the trajectory angle changes significantly.
        self.major_angle_changes: list[tuple[int, int]] = []

        # Identified pitch point (Y-coordinate peak).
        self.pitch_point: Optional[tuple[int, int]] = None

        # ── Internal bookkeeping ─────────────────────────────────────────
        # Rolling window of the last 3 ball positions for triplet checks.
        self._history: list[tuple[int, int]] = []

    # ─────────────────────────────────────────────────────────────────────────
    #  Frame-by-Frame Processing (§1 + §2)
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(
        self,
        ball_pos: tuple[int, int],
        bat_bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        """Ingest one frame's ball position and (optionally) the bat bbox."""
        ball_pos = (int(ball_pos[0]), int(ball_pos[1]))
        self._history.append(ball_pos)

        # ── Bat-bbox containment check ────────────────────────────────────
        if bat_bbox is not None:
            bx1, by1, bx2, by2 = bat_bbox
            if bx1 < ball_pos[0] < bx2 and by1 < ball_pos[1] < by2:
                self.bat_bbox_balls.append(ball_pos)

        # ── Pitch point — Y-coordinate peak (needs 3 consecutive) ──────
        #   current.y > previous.y AND current.y > next.y
        if len(self._history) >= 3 and self.pitch_point is None:
            prev_pt: tuple[int, int] = self._history[-3]
            curr_pt: tuple[int, int] = self._history[-2]
            next_pt: tuple[int, int] = self._history[-1]
            if curr_pt[1] > prev_pt[1] and curr_pt[1] > next_pt[1]:
                self.pitch_point = curr_pt
                log.info("├── Analyzer: pitch detected at %s", self.pitch_point)

        # ── Major angle changes (max 2) ─────────────────────────────────────
        if len(self.major_angle_changes) < 2 and len(self._history) >= 3:
            pt_a: tuple[int, int] = self._history[-3]
            pt_b: tuple[int, int] = self._history[-2]
            pt_c: tuple[int, int] = self._history[-1]

            vel_before: tuple[float, float] = compute_velocity_vector(pt_a, pt_b)
            vel_after: tuple[float, float] = compute_velocity_vector(pt_b, pt_c)
            angle: float = compute_deflection_angle(vel_before, vel_after)

            if angle >= self.angle_change_threshold_deg:
                candidate: tuple[int, int] = pt_b  # the inflection point
                # Avoid duplicating a point already recorded AND enforce
                # a minimum 40 px gap between the two angle-change points
                # so that frame-to-frame noise doesn't produce a false
                # second point right next to the first.
                if not self.major_angle_changes or self._pixel_distance(
                    candidate, self.major_angle_changes[-1]
                ) >= self.MIN_ANGLE_CHANGE_DISTANCE_PX:
                    self.major_angle_changes.append(candidate)
                    log.info(
                        "├── Analyzer: angle change #%d at %s (%.1f°)",
                        len(self.major_angle_changes), candidate, angle,
                    )

    # ── Evaluation (3 Cases) ───────────────────────────────────────────────────

    def evaluate_event(self) -> AnalyzerResult:
        """Evaluate collected state and return the final analysis result."""

        has_pitch: bool = self.pitch_point is not None
        num_angle_changes: int = len(self.major_angle_changes)

        # ── Case 1: Pitch and Hit (Standard Bounce) ─────────────────────
        # Condition: pitch_point exists AND 2 major angle changes.
        if has_pitch and num_angle_changes == 2:
            first_angle_pt: tuple[int, int] = self.major_angle_changes[0]
            second_angle_pt: tuple[int, int] = self.major_angle_changes[1]

            dist_to_first: float = self._pixel_distance(self.pitch_point, first_angle_pt)
            dist_to_second: float = self._pixel_distance(first_angle_pt, second_angle_pt)

            # ── Full-toss guard: pitch matches 2nd angle change ─────────
            if dist_to_second <= PITCH_MATCH_TOLERANCE_PX:
                log.info(
                    "├── Case 1→2: pitch=%s matches 2nd angle change "
                    "(dist=%.1fpx ≤ %.0fpx) → Full Toss",
                    self.pitch_point, dist_to_second, PITCH_MATCH_TOLERANCE_PX,
                )
                return AnalyzerResult(
                    case=2,
                    case_label="Full Toss",
                    impact_point=second_angle_pt,
                    is_full_toss=True,
                    detail=(
                        f"Pitch at {self.pitch_point} matches 2nd angle change "
                        f"{second_angle_pt} (dist={dist_to_second:.1f}px) → "
                        f"false pitch (Y-peak is actually bat impact). "
                        f"Delivery flagged as Full Toss."
                    ),
                )

            # ── Standard bounce: pitch within tolerance of 1st angle ─────
            if dist_to_first <= PITCH_MATCH_TOLERANCE_PX:
                log.info(
                    "├── Case 1: pitch=%s confirmed (dist=%.1fpx), impact=%s",
                    self.pitch_point, dist_to_first, second_angle_pt,
                )
                return AnalyzerResult(
                    case=1,
                    case_label="Pitch and Hit (Standard Bounce)",
                    pitch_point=self.pitch_point,
                    impact_point=second_angle_pt,
                    detail=(
                        f"Pitch at {self.pitch_point} matched 1st angle change "
                        f"(dist={dist_to_first:.1f}px). Impact at {second_angle_pt}."
                    ),
                )

            # ── Pitch doesn't match either angle change ──────────────────
            log.info(
                "├── Case 1 (no-match): pitch=%s ≃40px from 1st angle "
                "%s (dist=%.1fpx). Keeping pitch, impact=%s.",
                self.pitch_point, first_angle_pt, dist_to_first,
                second_angle_pt,
            )
            return AnalyzerResult(
                case=1,
                case_label="Pitch and Hit (Standard Bounce)",
                pitch_point=self.pitch_point,
                impact_point=second_angle_pt,
                detail=(
                    f"Pitch at {self.pitch_point} did not match 1st angle "
                    f"change {first_angle_pt} (dist={dist_to_first:.1f}px). "
                    f"Pitch retained; impact at {second_angle_pt}."
                ),
            )

        # ── Case 2: No Pitch, but Angle Changes (Full Toss / Direct) ───
        if not has_pitch and num_angle_changes > 0:
            if num_angle_changes == 2:
                impact: tuple[int, int] = self.major_angle_changes[0]
                log.info("├── Case 2 (2 angle changes): impact=%s", impact)
                return AnalyzerResult(
                    case=2,
                    case_label="No Pitch – Direct Hit",
                    impact_point=impact,
                    detail=(
                        f"No pitch. 2 angle changes; 1st point {impact} used as impact."
                    ),
                )
            else:
                impact = self.major_angle_changes[0]
                log.info("├── Case 2 (1 angle change, Full Toss): impact=%s", impact)
                return AnalyzerResult(
                    case=2,
                    case_label="Full Toss",
                    impact_point=impact,
                    is_full_toss=True,
                    detail=(
                        f"No pitch. Single angle change at {impact}; Full Toss."
                    ),
                )

        # ── Case 3: Missed Ball ──────────────────────────────────────────
        if not has_pitch and num_angle_changes == 0:
            if self.bat_bbox_balls:
                last_bat_ball: tuple[int, int] = self.bat_bbox_balls[-1]
                log.info(
                    "Analyzer Case 3 (Missed): last bat_bbox_ball=%s",
                    last_bat_ball,
                )
                return AnalyzerResult(
                    case=3,
                    case_label="Missed by Batsman",
                    impact_point=last_bat_ball,
                    missed_by_batsman=True,
                    detail=(
                        f"No pitch, no angle changes. Last ball inside bat "
                        f"bbox at {last_bat_ball}. Flagged as Missed by Batsman."
                    ),
                )
            else:
                log.info("Analyzer Case 3: no data at all (no bat contact)")
                return AnalyzerResult(
                    case=3,
                    case_label="Missed by Batsman (no bat contact)",
                    missed_by_batsman=True,
                    detail="No pitch, no angle changes, and no bat-bbox balls.",
                )

        # ── Edge case: pitch exists but <2 angle changes ───────────────
        if has_pitch and num_angle_changes == 1:
            impact = self.major_angle_changes[0]
            dist: float = self._pixel_distance(self.pitch_point, impact)
            if dist <= self.pixel_tolerance:
                log.info(
                    "├── Edge: pitch=%s matches sole angle change (no impact)",
                    self.pitch_point,
                )
                return AnalyzerResult(
                    case=1,
                    case_label="Pitch only (no bat impact)",
                    pitch_point=self.pitch_point,
                    detail=(
                        f"Pitch at {self.pitch_point} matches the only angle "
                        f"change. No second angle change → no bat impact."
                    ),
                )
            else:
                log.info(
                    "├── Edge: pitch=%s, sole angle change %s → impact "
                    "(dist=%.1fpx)",
                    self.pitch_point, impact, dist,
                )
                return AnalyzerResult(
                    case=1,
                    case_label="Pitch and Hit (single angle change)",
                    pitch_point=self.pitch_point,
                    impact_point=impact,
                    detail=(
                        f"Pitch at {self.pitch_point}. Single angle change at "
                        f"{impact} (dist={dist:.1f}px) used as impact."
                    ),
                )

        if has_pitch and num_angle_changes == 0:
            log.info("├── Edge: pitch=%s, no angle changes", self.pitch_point)
            return AnalyzerResult(
                case=1,
                case_label="Pitch only (no angle changes)",
                pitch_point=self.pitch_point,
                detail=(
                    f"Pitch at {self.pitch_point}. No angle changes detected "
                    f"→ no bat impact."
                ),
            )

        # Absolute fallback (should be unreachable).
        return AnalyzerResult(
            case=0,
            case_label="Indeterminate",
            detail="Could not classify the delivery.",
        )

    # ── External Pitch Injection ────────────────────────────────────────────────

    def set_pitch_point(self, point: tuple[int, int]) -> None:
        """Accept a pitch point detected externally (e.g. by the processor)."""
        if self.pitch_point is None:
            self.pitch_point = (int(point[0]), int(point[1]))
            log.info("├── Analyzer: pitch set externally at %s", self.pitch_point)

    # ── Reset ───────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all state for the next delivery."""
        self.bat_bbox_balls.clear()
        self.major_angle_changes.clear()
        self.pitch_point = None
        self._history.clear()

    # ── Private Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _pixel_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @property
    def tracking_complete(self) -> bool:
        """True once 2 major angle changes have been recorded."""
        return len(self.major_angle_changes) >= 2

    def __repr__(self) -> str:
        return (
            f"BallTrackerAnalyzer("
            f"pitch={self.pitch_point}, "
            f"angle_changes={self.major_angle_changes}, "
            f"bat_balls={len(self.bat_bbox_balls)})"
        )
