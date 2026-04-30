"""
config.py — Tunable parameters for the YOLO Pose arm-alternation counter.

Adjust these values to change sensitivity, speed, visual style, and hardware usage.
"""

# ─────────────────────────── Camera ───────────────────────────
WEBCAM_INDEX: int = 0          # 0 = default camera
CAPTURE_WIDTH: int = 1280      # requested capture width
CAPTURE_HEIGHT: int = 720      # requested capture height
MIRROR_MODE: bool = True       # flip frame horizontally for user-friendly display

# ─────────────────────────── YOLO Pose ────────────────────────
# Lightweight model for real-time on RTX 3050.
# Swap to "yolov8s-pose.pt" or "yolov8m-pose.pt" for higher accuracy.
YOLO_MODEL: str = "yolov8n-pose.pt"
YOLO_IMGSZ: int = 640          # inference resolution (multiple of 32)
YOLO_CONF: float = 0.40       # detection confidence threshold
YOLO_IOU: float = 0.50        # NMS IoU threshold
YOLO_DEVICE: str = "auto"     # "auto" → CUDA if available, else CPU
                                # set to "cpu" or "cuda:0" to force

# ─────────────────────────── Keypoint filtering ───────────────
MIN_KP_CONFIDENCE: float = 0.45   # ignore keypoints below this score

# ─────────────────────────── Smoothing ────────────────────────
# Exponential moving average factor for keypoint coordinates.
# 0 = no smoothing (raw), 1 = infinite memory (frozen).
SMOOTHING_ALPHA: float = 0.45

# ─────────────────────────── Arm-state classification ─────────
# All vertical thresholds are expressed as fractions of *torso height*
# (shoulder → hip midpoint) so they scale with distance from camera.
# A wrist is "raised" when it is this much ABOVE the shoulder line:
RAISED_WRIST_ABOVE_SHOULDER: float = -0.10   # negative = allowed slightly below
# A wrist is "raised" when it is at least this much ABOVE the elbow:
RAISED_WRIST_ABOVE_ELBOW: float = 0.05
# A wrist is "lowered" when it is this much BELOW the shoulder line:
LOWERED_WRIST_BELOW_SHOULDER: float = 0.15
# Hysteresis band (fraction of torso height) to avoid flickering:
HYSTERESIS_BAND: float = 0.08

# ─────────────────────────── State machine / counter ──────────
MIN_STABLE_FRAMES: int = 4        # frames a state must persist before accepted
COOLDOWN_FRAMES: int = 8          # frames to ignore after a count event

# ─────────────────────────── Hand box estimation ──────────────
# Scale of the hand-box side length relative to *shoulder distance*.
HAND_BOX_SCALE: float = 0.30

# ─────────────────────────── Drawing / colours ────────────────
# BGR tuples
COLOR_KEYPOINT: tuple = (0, 255, 200)      # cyan-green dots
COLOR_SKELETON: tuple = (255, 180, 50)      # warm blue skeleton lines
COLOR_HAND_BOX: tuple = (80, 220, 255)      # amber/yellow hand box
COLOR_SCORE_TEXT: tuple = (255, 255, 255)   # white score text
COLOR_STATE_TEXT: tuple = (200, 200, 200)   # light grey state label
COLOR_FPS_TEXT: tuple = (100, 255, 100)     # green FPS
COLOR_BG_OVERLAY: tuple = (30, 30, 30)      # dark background for text

KEYPOINT_RADIUS: int = 6
SKELETON_THICKNESS: int = 3
HAND_BOX_THICKNESS: int = 2
FONT_SCALE_SCORE: float = 1.2
FONT_SCALE_STATE: float = 0.7
FONT_SCALE_FPS: float = 0.6
FONT_THICKNESS: int = 2

# ─────────────────────────── Display ──────────────────────────
SHOW_FPS: bool = True
WINDOW_NAME: str = "YOLO Pose — Arm Alternation Counter"
