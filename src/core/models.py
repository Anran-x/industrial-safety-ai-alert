"""
Frame 数据结构与 YOLO 推理封装。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import gc

import numpy as np


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    track_id: Optional[int] = None

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def aspect(self) -> float:
        return self.w / max(self.h, 1e-6)


@dataclass
class Keypoints:
    # COCO 17 关键点: 0 nose,1 left_eye,2 right_eye,3 left_ear,4 right_ear,
    # 5 left_shoulder,6 right_shoulder,7 left_elbow,8 right_elbow,
    # 9 left_wrist,10 right_wrist,11 left_hip,12 right_hip,
    # 13 left_knee,14 right_knee,15 left_ankle,16 right_ankle
    xy: np.ndarray      # (17, 2)
    conf: np.ndarray    # (17,)

    def vis(self, idx: int, th: float = 0.4) -> bool:
        return self.conf[idx] >= th


@dataclass
class Person:
    box: Box
    kps: Optional[Keypoints] = None


@dataclass
class HeadBox(Box):
    pass


@dataclass
class FrameResult:
    persons: List[Person] = field(default_factory=list)
    heads: List[HeadBox] = field(default_factory=list)
    helmet_count: int = 0
    no_helmet_count: int = 0
    fall: bool = False
    alert_types: List[str] = field(default_factory=list)
    annotated: Optional[np.ndarray] = None