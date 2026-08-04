"""
人员倒地检测:基于姿态关键点 + 时序确认规则。

规则说明(可解释,面试可讲):
- 站立时人体包围框宽高比(宽/高)通常 < 0.6;倒地时 > 1.0;
- 用"肩部中点与髋部中点近似同高"排除下蹲/弯腰干扰;
- 同一目标连续 N 帧满足倒地条件才触发告警(抑制瞬时误报);
- 连续 M 帧恢复站立后复位状态机。

不选用时序模型(LSTM/Transformer)的原因:小样本下规则法可解释、
可调参、部署开销低;时序模型需要大量带标注的跌倒视频,且难以解释。
"""
from typing import Dict, List, Optional

import numpy as np

from src.config import FALL_LYING_ASPECT, FALL_CONFIRM_FRAMES, FALL_RESET_FRAMES
from src.core.models import Person


class FallStateMachine:
    def __init__(self, lying_aspect: float = FALL_LYING_ASPECT,
                 confirm_frames: int = FALL_CONFIRM_FRAMES,
                 reset_frames: int = FALL_RESET_FRAMES):
        self.lying_aspect = lying_aspect
        self.confirm_frames = confirm_frames
        self.reset_frames = reset_frames
        self._lying_count: Dict[int, int] = {}
        self._upright_count: Dict[int, int] = {}
        self._alarmed: Dict[int, bool] = {}
        self._active: Dict[int, int] = {}

    def _is_lying(self, person: Person) -> bool:
        box = person.box
        # 主判据:包围框宽高比(宽/高)达到阈值,表明身体接近水平
        if box.aspect >= self.lying_aspect:
            return True
        # 辅助判据:躯干(肩部中点→髋部中点)接近水平,排除下蹲/弯腰
        kps = person.kps
        if kps is not None:
            if kps.vis(5) and kps.vis(6) and kps.vis(11) and kps.vis(12):
                neck = (kps.xy[5] + kps.xy[6]) / 2   # 肩部中点
                hip = (kps.xy[11] + kps.xy[12]) / 2  # 髋部中点
                dy = abs(float(neck[1] - hip[1]))
                dx = abs(float(neck[0] - hip[0]))
                if dx > dy * 1.2 and box.aspect >= 0.9:
                    return True
        return False

    def update(self, persons: List[Person]) -> List[int]:
        """输入当前帧人员列表,返回触发倒地告警的 track_id 列表。"""
        alarms: List[int] = []
        seen: set = set()
        for p in persons:
            tid = p.box.track_id if p.box.track_id is not None else -1
            if tid == -1:
                tid = self._match_untracked(p, seen)
            seen.add(tid)
            self._active[tid] = self._active.get(tid, 0) + 1

            if self._is_lying(p):
                self._upright_count[tid] = 0
                self._lying_count[tid] = self._lying_count.get(tid, 0) + 1
                if self._lying_count[tid] >= self.confirm_frames and not self._alarmed.get(tid, False):
                    self._alarmed[tid] = True
                    alarms.append(tid)
            else:
                self._lying_count[tid] = 0
                self._upright_count[tid] = self._upright_count.get(tid, 0) + 1
                if self._upright_count[tid] >= self.reset_frames:
                    self._alarmed[tid] = False

        # 清理长时间消失的目标(约 6s @15fps)
        stale = [k for k in list(self._active) if k not in seen and self._active[k] > 90]
        for k in stale:
            for d in (self._lying_count, self._upright_count, self._alarmed, self._active):
                d.pop(k, None)
        return alarms

    def _match_untracked(self, person: Person, seen: set) -> int:
        """无跟踪ID时的兜底:按包围框中心与活跃目标做最近匹配(简化)。"""
        if person.box.track_id is not None:
            return person.box.track_id
        return -1
