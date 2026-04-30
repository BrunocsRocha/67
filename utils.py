"""
utils.py — Smoothing helpers, drawing helpers, geometric calculations,
           and confidence filtering for the pose-counter project.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config as cfg


# ══════════════════════════════════════════════════════════════
#  Type aliases
# ══════════════════════════════════════════════════════════════
Point = Tuple[float, float]       # (x, y)
Keypoints = Dict[str, Point]      # name → (x, y)


# ══════════════════════════════════════════════════════════════
#  Keypoint confidence filtering
# ══════════════════════════════════════════════════════════════

def filter_keypoints(
    raw: Dict[str, Tuple[float, float, float]],
    min_conf: float = cfg.MIN_KP_CONFIDENCE,
) -> Keypoints:
    """Return only keypoints whose confidence ≥ *min_conf*, dropping the
    confidence value from the tuple."""
    return {
        name: (x, y)
        for name, (x, y, c) in raw.items()
        if c >= min_conf
    }


# ══════════════════════════════════════════════════════════════
#  Exponential moving-average smoother
# ══════════════════════════════════════════════════════════════

class KeypointSmoother:
    """Applies per-keypoint exponential moving average (EMA).

    ``alpha`` controls the weight of *history*:
        smoothed = alpha * previous + (1 - alpha) * current
    Higher alpha → more smoothing (slower response).
    """

    def __init__(self, alpha: float = cfg.SMOOTHING_ALPHA) -> None:
        self.alpha = alpha
        self._state: Dict[str, np.ndarray] = {}

    def smooth(self, keypoints: Keypoints) -> Keypoints:
        result: Keypoints = {}
        for name, (x, y) in keypoints.items():
            cur = np.array([x, y], dtype=np.float64)
            if name in self._state:
                prev = self._state[name]
                smoothed = self.alpha * prev + (1.0 - self.alpha) * cur
            else:
                smoothed = cur
            self._state[name] = smoothed
            result[name] = (float(smoothed[0]), float(smoothed[1]))
        # Evict keys that vanished (lost tracking)
        self._state = {k: v for k, v in self._state.items() if k in keypoints}
        return result

    def reset(self) -> None:
        self._state.clear()


# ══════════════════════════════════════════════════════════════
#  Geometry helpers
# ══════════════════════════════════════════════════════════════

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def unit_vector(a: Point, b: Point) -> Optional[Tuple[float, float]]:
    """Unit vector from *a* → *b*.  Returns ``None`` if points coincide."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return None
    return (dx / mag, dy / mag)


def estimate_torso_height(kps: Keypoints) -> Optional[float]:
    """Rough torso height = shoulder midpoint → hip midpoint.

    Falls back to inter-shoulder distance × 1.5 if hips are not available.
    """
    ls = kps.get("left_shoulder")
    rs = kps.get("right_shoulder")
    if ls is None or rs is None:
        return None
    shoulder_mid = midpoint(ls, rs)

    lh = kps.get("left_hip")
    rh = kps.get("right_hip")
    if lh is not None and rh is not None:
        hip_mid = midpoint(lh, rh)
        return distance(shoulder_mid, hip_mid)

    # Fallback: use shoulder distance * 1.5
    return distance(ls, rs) * 1.5


def shoulder_distance(kps: Keypoints) -> Optional[float]:
    ls = kps.get("left_shoulder")
    rs = kps.get("right_shoulder")
    if ls is None or rs is None:
        return None
    return distance(ls, rs)


def estimate_hand_box(
    wrist: Point,
    elbow: Point,
    body_scale: float,
    scale: float = cfg.HAND_BOX_SCALE,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return (top-left, bottom-right) of a square box centred ahead of the
    wrist, oriented along the forearm direction, scaled relative to body size.
    """
    half = max(int(body_scale * scale / 2), 8)
    uv = unit_vector(elbow, wrist)
    if uv is not None:
        # Shift the box centre a little past the wrist
        cx = wrist[0] + uv[0] * half * 0.5
        cy = wrist[1] + uv[1] * half * 0.5
    else:
        cx, cy = wrist

    tl = (int(cx - half), int(cy - half))
    br = (int(cx + half), int(cy + half))
    return tl, br


# ══════════════════════════════════════════════════════════════
#  Drawing helpers
# ══════════════════════════════════════════════════════════════

_SKELETON_PAIRS: List[Tuple[str, str]] = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_shoulder"),
]


def draw_skeleton(frame: np.ndarray, kps: Keypoints) -> None:
    """Draw arm skeleton lines on *frame*."""
    for a_name, b_name in _SKELETON_PAIRS:
        a = kps.get(a_name)
        b = kps.get(b_name)
        if a is not None and b is not None:
            cv2.line(
                frame,
                _int_pt(a), _int_pt(b),
                cfg.COLOR_SKELETON,
                cfg.SKELETON_THICKNESS,
                cv2.LINE_AA,
            )


def draw_keypoints(frame: np.ndarray, kps: Keypoints) -> None:
    """Draw filled circles at each tracked keypoint."""
    for pt in kps.values():
        cv2.circle(
            frame,
            _int_pt(pt),
            cfg.KEYPOINT_RADIUS,
            cfg.COLOR_KEYPOINT,
            -1,
            cv2.LINE_AA,
        )


def draw_hand_boxes(
    frame: np.ndarray,
    kps: Keypoints,
    body_scale: float,
) -> None:
    """Draw estimated hand boxes for left and right wrists."""
    for side in ("left", "right"):
        wrist = kps.get(f"{side}_wrist")
        elbow = kps.get(f"{side}_elbow")
        if wrist is not None and elbow is not None:
            tl, br = estimate_hand_box(wrist, elbow, body_scale)
            cv2.rectangle(frame, tl, br, cfg.COLOR_HAND_BOX,
                          cfg.HAND_BOX_THICKNESS, cv2.LINE_AA)


def draw_score(
    frame: np.ndarray,
    score: int,
    kps: Keypoints,
) -> None:
    """Draw the score above the person's head / upper body."""
    # Try nose → shoulder midpoint → frame top-centre as fallback
    anchor = kps.get("nose")
    if anchor is None:
        ls = kps.get("left_shoulder")
        rs = kps.get("right_shoulder")
        if ls is not None and rs is not None:
            anchor = midpoint(ls, rs)
    if anchor is None:
        anchor = (frame.shape[1] / 2, 40)

    text = f"Score: {score}"
    pos = (int(anchor[0]) - 60, max(int(anchor[1]) - 40, 30))
    _draw_text_with_bg(frame, text, pos,
                       cfg.FONT_SCALE_SCORE, cfg.COLOR_SCORE_TEXT,
                       cfg.FONT_THICKNESS)


def draw_state_label(
    frame: np.ndarray,
    label: str,
    kps: Keypoints,
) -> None:
    """Show the current state-machine label below the score."""
    anchor = kps.get("nose")
    if anchor is None:
        ls = kps.get("left_shoulder")
        rs = kps.get("right_shoulder")
        if ls is not None and rs is not None:
            anchor = midpoint(ls, rs)
    if anchor is None:
        anchor = (frame.shape[1] / 2, 80)

    pos = (int(anchor[0]) - 90, max(int(anchor[1]) - 10, 55))
    _draw_text_with_bg(frame, label, pos,
                       cfg.FONT_SCALE_STATE, cfg.COLOR_STATE_TEXT, 1)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Show FPS in the top-left corner."""
    text = f"FPS: {fps:.1f}"
    _draw_text_with_bg(frame, text, (10, 28),
                       cfg.FONT_SCALE_FPS, cfg.COLOR_FPS_TEXT, 1)


# ──────────────── internal helpers ────────────────────────────

def _int_pt(p: Point) -> Tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


def _draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw *text* with a semi-transparent dark background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad = 6
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        cfg.COLOR_BG_OVERLAY,
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, org, font, font_scale, color, thickness,
                cv2.LINE_AA)
