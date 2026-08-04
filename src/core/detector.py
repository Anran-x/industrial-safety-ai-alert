"""
YOLO 检测/姿态推理封装:头盔检测 + 人员姿态估计两条推理管线。
"""
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np

from src.config import (
    HELMET_WEIGHTS, POSE_MODEL, HELMET_CONF, CPU_FALLBACK,
)
from src.core.models import Box, FrameResult, HeadBox, Keypoints, Person

import torch


class SafetyVision:
    """统一封装头盔检测模型与姿态模型,输出结构化的 FrameResult。"""

    def __init__(self, helmet_weights: Optional[str] = str(HELMET_WEIGHTS),
                 pose_model: str = POSE_MODEL,
                 helmet_conf: float = HELMET_CONF,
                 device: Optional[str] = None,
                 helmet_on: bool = True):
        from ultralytics import YOLO
        self.helmet_conf = helmet_conf
        self.device = device or self._pick_device()
        self.helmet_on = helmet_on
        self.helmet_model = None
        if self.helmet_on:
            if not Path(helmet_weights).exists():
                raise FileNotFoundError(f"头盔模型不存在: {helmet_weights} 请先训练")
            self.helmet_model = YOLO(helmet_weights)
        self.pose_model = YOLO(pose_model)

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "0"
        return "cpu"

    def _to_device(self):
        return self.device

    def process_frame(self, frame: np.ndarray, track: bool = True,
                      pose_on: bool = True, helmet_on: bool = True) -> FrameResult:
        """对单帧执行人员姿态估计(倒地/闯入)与头盔检测。"""
        res = FrameResult()
        H, W = frame.shape[:2]

        if pose_on:
            preds = self.pose_model(frame, device=self.device, verbose=False)
            if preds:
                r = preds[0]
                for i, box in enumerate(r.boxes):
                    cls = int(box.cls[0].item())
                    if cls != 0:  # 只处理 person 类
                        continue
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    pbox = Box(x1, y1, x2, y2, conf, 0)
                    kps = None
                    if r.keypoints is not None:
                        xy = r.keypoints.xy[i].cpu().numpy()
                        kc = r.keypoints.conf[i].cpu().numpy()
                        kps = Keypoints(xy=xy, conf=kc)
                    tid = None
                    if track and r.boxes.id is not None:
                        tid = int(r.boxes.id[i].item())
                        pbox.track_id = tid
                    res.persons.append(Person(box=pbox, kps=kps))

        if helmet_on and self.helmet_model is not None:
            hp = self.helmet_model(frame, device=self.device, verbose=False, conf=self.helmet_conf)
            if hp:
                r = hp[0]
                for i, box in enumerate(r.boxes):
                    conf = float(box.conf[0].item())
                    cls = int(box.cls[0].item())
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    hb = HeadBox(x1, y1, x2, y2, conf, cls)
                    if track and r.boxes.id is not None:
                        hb.track_id = int(r.boxes.id[i].item())
                    res.heads.append(hb)
                    if cls == 1:
                        res.helmet_count += 1
                    else:
                        res.no_helmet_count += 1

        return res