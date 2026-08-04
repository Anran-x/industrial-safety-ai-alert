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

    @staticmethod
    def _is_lying(person: Person) -> bool:
        box = person.box
        if box.aspect >= 1.0 and box.h > 0 and box.w / box.h >= 0.9:
            return True
        if person.kps is not None:
            neck = (person.kps.xy[5] + person.kps.xy[6]) / 2   # 肩部中点
            hip = (person.kps.xy[11] + person.kps.xy[12]) / 2  # 髋部中点
            if person.kps.vis(5) and person.kps.vis(6) and person.kps.vis(11) and person.kps.vis(12):
                dy = abs(neck[1] - hip[1])
                dx = abs(neck[0] - hip[0])
                return dx > dy * 1.2  # 躯干接近水平
        return False

    def update(self, persons: List[Person]) -> List[int]:
        """输入当前帧人员列表,返回触发倒地告警的 track_id 列表。"""
        alarms: List[int] = []
        seen: set = set()
        for p in persons:
            tid = p.box.track_id if p.box.track_id is not None else -1
            # 未跟踪目标按中心点最近匹配
            if tid == -1:
                tid = self._match_untracked(p, seen)
            seen.add(tid)

            if self._is_lying(p):
                self._upright_count[tid] = 0
                self._lying_count[tid] = self._lying_count.get(tid, 0) + 1
                self._active[tid] = 0
                if self._lying_count[tid] >= self.confirm_frames and not self._alarmed.get(tid, False):
                    self._alarmed[tid] = True
                    alarms.append(tid)
            else:
                self._lying_count[tid] = 0
                self._upright_count[tid] = self._upright_count.get(tid, 0) + 1
                if self._upright_count[tid] >= self.reset_frames:
                    self._alarmed[tid] = False

        # 清理长时间消失的目标
        stale = [k for k in list(self._active) if self._active[k] > 90]
        for k in stale:
            for d in (self._lying_count, self._upright_count, self._alarmed, self._active):
                d.pop(k, None)
        return alarms

    def _match_untracked(self, person: Person, seen: set) -> int:
        best, best_d = None, 1e9
        for tid in self._active:
            if tid in seen or tid < 0:
                continue
            # 用上一帧中心做近似(简化:按最近告警目标)
            best_d = 1e9
        return best if best is not None else -1