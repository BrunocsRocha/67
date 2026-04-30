"""
counter.py — Arm-state classification, deterministic state machine,
             debounce logic, cooldown, and score tracking.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Tuple

import config as cfg
from utils import Keypoints, estimate_torso_height


# ══════════════════════════════════════════════════════════════
#  Individual arm state
# ══════════════════════════════════════════════════════════════

class ArmState(Enum):
    UNKNOWN = auto()
    RAISED = auto()
    LOWERED = auto()


# ══════════════════════════════════════════════════════════════
#  Combined pose state (state-machine nodes)
# ══════════════════════════════════════════════════════════════

class PoseState(Enum):
    UNKNOWN = auto()
    LEFT_UP_RIGHT_DOWN = auto()
    RIGHT_UP_LEFT_DOWN = auto()
    NEUTRAL = auto()            # neither pattern


# ══════════════════════════════════════════════════════════════
#  Arm classifier
# ══════════════════════════════════════════════════════════════

def classify_arm(
    shoulder: Tuple[float, float],
    elbow: Tuple[float, float],
    wrist: Tuple[float, float],
    torso_h: float,
    prev_state: ArmState = ArmState.UNKNOWN,
) -> ArmState:
    """Classify a single arm as RAISED, LOWERED, or UNKNOWN.

    Uses hysteresis to avoid jitter: the threshold to *enter* a state is
    stricter than the threshold to *leave* it.

    All vertical comparisons are normalised by ``torso_h`` so they work
    at any camera distance.
    """
    if torso_h < 1e-3:
        return ArmState.UNKNOWN

    # Positive = wrist is BELOW reference (image-y increases downward)
    wrist_vs_shoulder = (wrist[1] - shoulder[1]) / torso_h
    wrist_vs_elbow = (wrist[1] - elbow[1]) / torso_h

    hyst = cfg.HYSTERESIS_BAND

    # ── RAISED ────────────────────────────────────────────────
    raised_thresh_shoulder = cfg.RAISED_WRIST_ABOVE_SHOULDER
    raised_thresh_elbow = -cfg.RAISED_WRIST_ABOVE_ELBOW   # negative = above

    if prev_state == ArmState.RAISED:
        # easier to stay raised (add hysteresis)
        raised_thresh_shoulder += hyst
        raised_thresh_elbow += hyst

    if wrist_vs_shoulder < raised_thresh_shoulder and wrist_vs_elbow < raised_thresh_elbow:
        return ArmState.RAISED

    # ── LOWERED ───────────────────────────────────────────────
    lowered_thresh = cfg.LOWERED_WRIST_BELOW_SHOULDER

    if prev_state == ArmState.LOWERED:
        lowered_thresh -= hyst  # easier to stay lowered

    if wrist_vs_shoulder > lowered_thresh and wrist_vs_elbow > 0:
        return ArmState.LOWERED

    return ArmState.UNKNOWN


# ══════════════════════════════════════════════════════════════
#  State machine & counter
# ══════════════════════════════════════════════════════════════

class AlternationCounter:
    """Deterministic state machine for arm-alternation counting.

    Lifecycle
    ---------
    1. Each frame, call :meth:`update` with filtered+smoothed keypoints.
    2. The counter classifies left / right arm states.
    3. It maps the pair to a ``PoseState``.
    4. A state must persist for ``MIN_STABLE_FRAMES`` before being accepted.
    5. A valid alternation (A↔B) increments the score.
    6. After scoring, a cooldown of ``COOLDOWN_FRAMES`` blocks further counts.
    """

    def __init__(self) -> None:
        self.score: int = 0

        # Per-arm previous states (for hysteresis)
        self._left_arm: ArmState = ArmState.UNKNOWN
        self._right_arm: ArmState = ArmState.UNKNOWN

        # State machine
        self._confirmed_state: PoseState = PoseState.UNKNOWN
        self._candidate_state: PoseState = PoseState.UNKNOWN
        self._candidate_frames: int = 0
        self._cooldown_remaining: int = 0

    # ──────────────────────────────────────────────────────────
    def update(self, kps: Keypoints) -> Tuple[PoseState, int]:
        """Process one frame.  Returns (current_state, score)."""

        torso_h = estimate_torso_height(kps)
        if torso_h is None or torso_h < 1e-3:
            return self._confirmed_state, self.score

        # Classify each arm
        self._left_arm = self._classify_side("left", kps, torso_h, self._left_arm)
        self._right_arm = self._classify_side("right", kps, torso_h, self._right_arm)

        # Map to combined pose state
        raw_state = self._map_pose_state(self._left_arm, self._right_arm)

        # Debounce
        if raw_state == self._candidate_state:
            self._candidate_frames += 1
        else:
            self._candidate_state = raw_state
            self._candidate_frames = 1

        # Accept candidate when stable
        if (
            self._candidate_frames >= cfg.MIN_STABLE_FRAMES
            and self._candidate_state != self._confirmed_state
        ):
            prev = self._confirmed_state
            self._confirmed_state = self._candidate_state

            # Check for valid alternation & cooldown
            if self._cooldown_remaining <= 0 and self._is_valid_alternation(prev, self._confirmed_state):
                self.score += 1
                self._cooldown_remaining = cfg.COOLDOWN_FRAMES

        # Tick cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        return self._confirmed_state, self.score

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _classify_side(
        side: str,
        kps: Keypoints,
        torso_h: float,
        prev: ArmState,
    ) -> ArmState:
        shoulder = kps.get(f"{side}_shoulder")
        elbow = kps.get(f"{side}_elbow")
        wrist = kps.get(f"{side}_wrist")
        if shoulder is None or elbow is None or wrist is None:
            return ArmState.UNKNOWN
        return classify_arm(shoulder, elbow, wrist, torso_h, prev)

    @staticmethod
    def _map_pose_state(left: ArmState, right: ArmState) -> PoseState:
        if left == ArmState.RAISED and right == ArmState.LOWERED:
            return PoseState.LEFT_UP_RIGHT_DOWN
        if right == ArmState.RAISED and left == ArmState.LOWERED:
            return PoseState.RIGHT_UP_LEFT_DOWN
        if left == ArmState.UNKNOWN or right == ArmState.UNKNOWN:
            return PoseState.UNKNOWN
        return PoseState.NEUTRAL

    @staticmethod
    def _is_valid_alternation(prev: PoseState, cur: PoseState) -> bool:
        valid_pairs = {
            (PoseState.LEFT_UP_RIGHT_DOWN, PoseState.RIGHT_UP_LEFT_DOWN),
            (PoseState.RIGHT_UP_LEFT_DOWN, PoseState.LEFT_UP_RIGHT_DOWN),
        }
        return (prev, cur) in valid_pairs

    # ──────────────────────────────────────────────────────────
    @property
    def state_label(self) -> str:
        _labels = {
            PoseState.UNKNOWN: "UNKNOWN",
            PoseState.LEFT_UP_RIGHT_DOWN: "L-UP  R-DOWN",
            PoseState.RIGHT_UP_LEFT_DOWN: "R-UP  L-DOWN",
            PoseState.NEUTRAL: "NEUTRAL",
        }
        return _labels.get(self._confirmed_state, "???")

    def reset(self) -> None:
        self.score = 0
        self._left_arm = ArmState.UNKNOWN
        self._right_arm = ArmState.UNKNOWN
        self._confirmed_state = PoseState.UNKNOWN
        self._candidate_state = PoseState.UNKNOWN
        self._candidate_frames = 0
        self._cooldown_remaining = 0
