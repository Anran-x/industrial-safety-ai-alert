"""
人员倒地检测:基于姿态关键点 + 时序确认规则。

规则说明(可解释,可审计):
- 站立时人体包围框宽高比(宽/高)通常 < 0.6;倒地时 > 1.0;
- 用"肩部中点与髋部中点近似同高"排除下蹲/弯腰干扰;
- 同一目标连续 N 帧满足倒地条件才触发告警(抑制瞬时误报);
- 连续 M 帧恢复站立后复位状态机。

对实际部署的鲁棒性设计:
1) 姿态模型在跌倒瞬间常丢帧,重新检出时跟踪 ID 会切换。若按 ID 严格
   清零计数会把实际跌倒漏报,因此:
   - 未跟踪目标按"中心位置就近匹配"续接原目标,而非共用 -1 计数;
   - 新出现的跟踪 ID 若贴近"刚消失"的目标,视为断帧重联,沿用其全部
     状态(计数/门控历史),避免把一次跌倒拆成两段;
   - 躺倒计数采用"衰减"而非"清零":短暂 1~2 帧缺失仍能续上,
     长时间(>= reset_frames)恢复正常姿态才复位告警。
2) 快速倒伏门控(区分跌倒与缓慢坐下/蹲下):
   - 门控A:躺倒必须发生在"刚处于直立窄框(aspect<0.8)"之后不久内;
   - 门控B:近 widen_frames 帧内 aspect 应出现一次明显加宽(max-min >=
     widen_min),排除全程缓慢变宽的坐/趴等日常动作。

不选用时序模型(LSTM/Transformer)的原因:小样本下规则法可解释、
可调参、部署开销低;时序模型需要大量带标注的跌倒视频,且难以解释。
"""
import math
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from src.config import FALL_LYING_ASPECT, FALL_CONFIRM_FRAMES, FALL_RESET_FRAMES
from src.core.models import Person


class FallStateMachine:
    def __init__(self, lying_aspect: float = FALL_LYING_ASPECT,
                 confirm_frames: int = FALL_CONFIRM_FRAMES,
                 reset_frames: int = FALL_RESET_FRAMES,
                 gap_decay: int = 2,
                 match_scale: float = 2.0,
                 fps: float = 25.0,
                 narrow_aspect: float = 0.8,
                 gate_sec: float = 1.8,
                 widen_min: float = 0.5,
                 widen_frames: int = 24):
        self.lying_aspect = lying_aspect
        self.confirm_frames = confirm_frames
        self.reset_frames = reset_frames
        self.gap_decay = gap_decay          # 未躺倒帧的计数衰减量(容忍短暂断检)
        self.match_scale = match_scale      # 未跟踪目标就近匹配的位移容忍(相对框宽)
        self.fps = fps                      # 输入帧率,用于把"快速倒伏"门控换算成秒
        self.narrow_aspect = narrow_aspect  # 低于该宽高比视为"直立/窄框"
        self.gate_sec = gate_sec            # 门控A:距最后直立状态不得超过该秒数
        self.widen_min = widen_min          # 门控B:观察窗口内 aspect 涨幅下限
        self.widen_frames = widen_frames    # 门控B:观察窗口(帧)
        self._lying_count: Dict[int, int] = {}
        self._upright_count: Dict[int, int] = {}
        self._alarmed: Dict[int, bool] = {}
        self._active: Dict[int, int] = {}
        self._last_pos: Dict[int, Tuple[float, float]] = {}
        self._last_seen: Dict[int, int] = {}            # 各目标最近被检测到的状态机帧号
        self._last_narrow: Dict[int, int] = {}          # 最近一次处于"直立窄框"的状态机帧号
        self._aspect_hist: Dict[int, deque] = {}        # 每目标最近 aspect 序列,用于门控B
        self._frame: int = 0

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

    def _match_untracked(self, person: Person, seen: set) -> int:
        """无跟踪 ID 的目标:按中心位置就近匹配最近活跃的目标;未匹配则 -1。"""
        cx, cy = person.box.cx, person.box.cy
        box_w = person.box.w
        best, best_d = None, None
        for k, active in self._active.items():
            if k in seen or not active or k not in self._last_pos:
                continue
            lx, ly = self._last_pos[k]
            d = math.hypot(cx - lx, cy - ly)
            th = self.match_scale * box_w + 20
            if d < th and (best_d is None or d < best_d):
                best, best_d = k, d
        return best if best is not None else -1

    def _find_reacq(self, person: Person, seen: set) -> int:
        """新跟踪 ID 的重联匹配:与"最近刚消失"的活跃目标按中心距离就近续接。"""
        cx, cy = person.box.cx, person.box.cy
        th = self.match_scale * person.box.w + 20
        best, best_d = None, None
        for k, last_seen in self._last_seen.items():
            if k in seen or k not in self._active or k not in self._last_pos:
                continue
            if self._frame - last_seen > 5:   # 仅续接"刚消失"的目标
                continue
            lx, ly = self._last_pos[k]
            d = math.hypot(cx - lx, cy - ly)
            if d < th and (best_d is None or d < best_d):
                best, best_d = k, d
        return best if best is not None else -1

    def _adopt(self, old: int, new: int):
        """把 old 目标的全部状态迁移到 new(断帧重联后的新 ID 沿用历史)。"""
        for d in (self._lying_count, self._upright_count, self._alarmed,
                  self._active, self._last_seen, self._last_narrow, self._aspect_hist):
            if old in d:
                d[new] = d.pop(old)

    def _check_fast_fall(self, tid: int) -> bool:
        """两条门控同时满足才放行:
        1) 躺倒发生在"刚处于直立状态后"的短时间内(距最后窄框 <= gate_sec);
        2) 近 widen_frames 帧内 aspect 出现一次明显加宽(max-min >= widen_min),
           排除全程缓慢变宽的坐下/趴下。
        从未记录过直立状态 / 历史不足时保守放行。"""
        last = self._last_narrow.get(tid)
        if last is not None and (self._frame - last) / self.fps > self.gate_sec:
            return False
        hist = self._aspect_hist.get(tid)
        if hist is None or len(hist) < 8:
            return True
        return max(hist) - min(hist) >= self.widen_min

    def update(self, persons: List[Person]) -> List[int]:
        """输入当前帧人员列表,返回触发倒地告警的 track_id 列表。"""
        alarms: List[int] = []
        seen: set = set()
        self._frame += 1
        for p in persons:
            tid = p.box.track_id
            if tid is None:
                tid = self._match_untracked(p, seen)
            else:
                # 新出现的跟踪 ID:若位置贴近"刚消失"的目标,视为姿态模型断帧
                # 重联,沿用旧目标的全部状态(倒伏计数/门控历史),避免把一次
                # 跌倒拆成两段而漏报,也避免新 ID 因无历史而绕过门控。
                if tid not in self._active:
                    cand = self._find_reacq(p, seen)
                    if cand is not None:
                        self._adopt(cand, tid)
                        tid = cand
            seen.add(tid)
            self._active[tid] = self._active.get(tid, 0) + 1
            self._last_seen[tid] = self._frame
            if tid == -1:
                # 无法关联的新目标(无跟踪 ID 且无历史):不累计,避免把
                # 多目标误并到同一个 -1 计数。
                continue
            self._last_pos[tid] = (p.box.cx, p.box.cy)

            # 记录最近一次"直立窄框"时刻,用于快速倒伏门控:
            # 跌倒=从直立(窄框)到平躺(宽框)发生在极短时间内;
            # 缓慢坐下/蹲下(数十帧内逐帧加宽)不算跌倒。
            if p.box.aspect < self.narrow_aspect:
                self._last_narrow[tid] = self._frame
            hist = self._aspect_hist.setdefault(tid, deque(maxlen=self.widen_frames + 1))
            hist.append(p.box.aspect)

            if self._is_lying(p):
                self._upright_count[tid] = 0
                self._lying_count[tid] = self._lying_count.get(tid, 0) + 1
                if (self._lying_count[tid] >= self.confirm_frames
                        and self._check_fast_fall(tid)
                        and not self._alarmed.get(tid, False)):
                    self._alarmed[tid] = True
                    alarms.append(tid)
            else:
                self._upright_count[tid] = self._upright_count.get(tid, 0) + 1
                # 非躺倒帧:衰减而非清零,容忍检测在跌倒瞬间的 1~2 帧丢失
                if self._lying_count.get(tid, 0) > 0:
                    self._lying_count[tid] = max(0, self._lying_count[tid] - self.gap_decay)
                if self._upright_count[tid] >= self.reset_frames:
                    self._alarmed[tid] = False
                    self._lying_count[tid] = 0
                    self._upright_count[tid] = 0

        # 清理长时间消失的目标(约 6s @15fps)
        stale = [k for k in list(self._active) if k not in seen and self._active[k] > 90]
        for k in stale:
            for d in (self._lying_count, self._upright_count, self._alarmed,
                      self._active, self._last_pos, self._last_seen,
                      self._last_narrow, self._aspect_hist):
                d.pop(k, None)
        return alarms
