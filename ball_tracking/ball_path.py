import math
from ball_tracking.config import DELIVERY_SAG_FACTOR, BOUNCE_SAG_FACTOR, DEFAULT_N_POINTS


# ── Internal helpers ──────────────────────────────────────────────────────────

def _control_point(
    p0: tuple[int, int],
    p2: tuple[int, int],
    sag: float,
) -> tuple[float, float]:
    """Return the quadratic Bezier control point P1 such that the curve
    midpoint (t=0.5) is displaced *sag* pixels from the chord midpoint.

    Derivation:
        B(0.5) = 0.25*P0 + 0.5*P1 + 0.25*P2
        We want B(0.5).y = (P0.y + P2.y)/2 + sag
        Solving → P1.y = 0.5*(P0.y + P2.y) + 2*sag

    Positive sag pushes the arc downward (y increases in image coords).
    Negative sag pushes it upward.
    """
    ctrl_x = (p0[0] + p2[0]) / 2.0
    ctrl_y = 0.5 * (p0[1] + p2[1]) + 2.0 * sag
    return (ctrl_x, ctrl_y)


def _quadratic_bezier_points(
    p0: tuple[int, int],
    p1: tuple[float, float],
    p2: tuple[int, int],
    n: int,
) -> list[tuple[int, int]]:
    """Sample *n+1* evenly-spaced points along the quadratic Bezier P0–P1–P2."""
    pts: list[tuple[int, int]] = []
    for i in range(n + 1):
        t  = i / n
        mt = 1.0 - t
        x  = mt * mt * p0[0] + 2.0 * mt * t * p1[0] + t * t * p2[0]
        y  = mt * mt * p0[1] + 2.0 * mt * t * p1[1] + t * t * p2[1]
        pts.append((int(round(x)), int(round(y))))
    return pts


# ── Public API ────────────────────────────────────────────────────────────────

def compute_delivery_arc(
    first_point: tuple[int, int],
    pitch_point: tuple[int, int],
    n_points: int = DEFAULT_N_POINTS,
    sag_factor: float = DELIVERY_SAG_FACTOR,
) -> list[tuple[int, int]]:
    """Compute a physics-based free-fall arc from the first detected ball
    position to the pitch point.

    Models the ball falling under gravity: the arc sags *downward* relative to
    the straight chord between the two endpoints.
    """
    dist = math.hypot(
        pitch_point[0] - first_point[0],
        pitch_point[1] - first_point[1],
    )
    sag  = -sag_factor * dist         # negative → upward in image (outward arc)
    ctrl = _control_point(first_point, pitch_point, sag)
    return _quadratic_bezier_points(first_point, ctrl, pitch_point, n_points)


def compute_bounce_arc(
    pitch_point: tuple[int, int],
    impact_point: tuple[int, int],
    n_points: int = DEFAULT_N_POINTS,
    sag_factor: float = BOUNCE_SAG_FACTOR,
) -> list[tuple[int, int]]:
    """Compute a physics-based bounce arc from the pitch point to the
    impact / bat point.

    Models the ball launching upward after bouncing off the pitch: the arc
    bulges *upward* relative to the straight chord.
    """
    dist = math.hypot(
        impact_point[0] - pitch_point[0],
        impact_point[1] - pitch_point[1],
    )
    sag  = -sag_factor * dist         # negative → upward in image
    ctrl = _control_point(pitch_point, impact_point, sag)
    return _quadratic_bezier_points(pitch_point, ctrl, impact_point, n_points)


def compute_full_path(
    first_point: tuple[int, int] | None,
    pitch_point: tuple[int, int] | None,
    impact_point: tuple[int, int] | None,
    n_points: int = DEFAULT_N_POINTS,
) -> dict:
    """Compute both path segments and return them as a dict.

    Returns
    -------
    {
        'delivery': list[tuple[int,int]] | None,  # first_point → pitch_point
        'bounce':   list[tuple[int,int]] | None,  # pitch_point → impact_point
    }
    Either value is None when the required endpoints are missing / falsy.
    """
    result: dict = {'delivery': None, 'bounce': None}

    if first_point and pitch_point:
        result['delivery'] = compute_delivery_arc(first_point, pitch_point, n_points)

    if pitch_point and impact_point:
        result['bounce'] = compute_bounce_arc(pitch_point, impact_point, n_points)

    # Fulltoss: no pitch but we have first & impact → single delivery arc
    if not pitch_point and first_point and impact_point:
        result['delivery'] = compute_delivery_arc(first_point, impact_point, n_points)

    return result
