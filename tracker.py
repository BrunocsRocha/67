"""
tracker.py — YOLO Pose model loading, inference, keypoint extraction.
"""

from __future__ import annotations

import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

import config as cfg

_COCO_KP_NAMES: Dict[int, str] = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
}

_REQUIRED_NAMES = {
    "nose", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip",
}


class PoseTracker:
    """Wraps Ultralytics YOLO Pose for single-person upper-body keypoints."""

    def __init__(self) -> None:
        device = self._resolve_device()
        print(f"[tracker] Loading {cfg.YOLO_MODEL} on device={device}")
        self.model = YOLO(cfg.YOLO_MODEL)
        self.device = device

    def process(self, frame: np.ndarray) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """Run inference, return {name: (x,y,conf)} for primary person or None."""
        results = self.model.predict(
            source=frame, imgsz=cfg.YOLO_IMGSZ, conf=cfg.YOLO_CONF,
            iou=cfg.YOLO_IOU, device=self.device, verbose=False, max_det=5,
        )
        if not results or results[0].keypoints is None:
            return None

        kps_data = results[0].keypoints
        boxes = results[0].boxes
        if kps_data.xy is None or len(kps_data.xy) == 0:
            return None

        best_idx = self._select_person(boxes)
        if best_idx is None:
            return None

        xy = kps_data.xy[best_idx].cpu().numpy()
        conf = kps_data.conf[best_idx].cpu().numpy()

        out: Dict[str, Tuple[float, float, float]] = {}
        for idx, name in _COCO_KP_NAMES.items():
            if name in _REQUIRED_NAMES:
                out[name] = (float(xy[idx, 0]), float(xy[idx, 1]), float(conf[idx]))
        return out

    @staticmethod
    def _select_person(boxes) -> Optional[int]:
        if boxes is None or len(boxes) == 0:
            return None
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        max_area = areas.max() if areas.max() > 0 else 1.0
        scores = confs * 0.6 + (areas / max_area) * 0.4
        return int(np.argmax(scores))

    @staticmethod
    def _resolve_device() -> str:
        if cfg.YOLO_DEVICE != "auto":
            return cfg.YOLO_DEVICE
        return "cuda:0" if torch.cuda.is_available() else "cpu"
