"""
危险区域闯入检测:ByteTrack 目标跟踪 + 多边形 ROI 判定。

规则说明:
- ROI 由归一化多边形定义(0~1 坐标,便于适配任意分辨率);
- 人员轨迹中心点进入 ROI 后,连续 N 帧仍在区域内才触发告警(防闪烁);
- 目标离开 ROI 或消失后复位。
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from src.config import INTRUSION_ROI, INTRUSION_CONFIRM_FRAMES
from src.core.models import Person


class IntrusionDetector:
    def __init__(self, roi: Optional[List[Tuple[float, float]]] = None,
                 confirm_frames: int = INTRUSION_CONFIRM_FRAMES):
        self.roi = np.array(roi or INTRUSION_ROI, dtype=np.float32)
        self.confirm_frames = confirm_frames
        self._inside_count: Dict[int, int] = {}
        self._outside_count: Dict[int, int] = {}
        self._alarmed: Dict[int, bool] = {}

    def set_roi(self, roi: List[Tuple[float, float]]):
        self.roi = np.array(roi, dtype=np.float32)

    def _inside(self, x: float, y: float) -> bool:
        # 归一化坐标直接使用 pointPolygonTest
        pt = (float(x), float(y))
        dist = cv2.pointPolygonTest(self.roi, pt, False)
        return dist >= 0

    def update(self, persons: List[Person], frame_shape: Tuple[int, int]) -> List[Tuple[int, str]]:
        """输入人员列表与当前帧 (H, W),返回 [(track_id, 'INTRUSION'), ...]。"""
        alarms: List[Tuple[int, str]] = []
        inside_ids: set = set()
        H, W = frame_shape

        for p in persons:
            tid = p.box.track_id if p.box.track_id is not None else -1
            if tid < 0:
                continue
            x_n, y_n = p.box.cx / max(W, 1), p.box.cy / max(H, 1)
            inside = self._inside(x_n, y_n)
            if inside:
                inside_ids.add(tid)
                self._outside_count[tid] = 0
                self._inside_count[tid] = self._inside_count.get(tid, 0) + 1
                if self._inside_count[tid] >= self.confirm_frames and not self._alarmed.get(tid, False):
                    self._alarmed[tid] = True
                    alarms.append((tid, "INTRUSION"))
            else:
                self._inside_count[tid] = 0
                self._outside_count[tid] = self._outside_count.get(tid, 0) + 1
                if self._outside_count[tid] >= 15:
                    self._alarmed[tid] = False

        stale = [k for k in list(self._alarmed) if k not in inside_ids and self._outside_count.get(k, 0) > 90]
        for k in stale:
            for d in (self._inside_count, self._outside_count, self._alarmed):
                d.pop(k, None)
        return alarms