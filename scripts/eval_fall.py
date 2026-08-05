"""
在 UR Fall 数据集上评估倒地检测规则。

评估口径:
- 跌倒视频(fall-XX):视频内触发过一次 FALL 告警即记为检出;
- 日常动作视频(adl-XX):累计误报帧数/出现误报的视频数 => 误报率。
返回:检出率(召回率)、误报视频数、误报帧占比。

用法: python scripts/eval_fall.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import UR_FALL_DIR
from src.core import detector as det  # noqa
from src.behavior.fall_detector import FallStateMachine


def process_video(path: str, vision, step: int = 1, sm: FallStateMachine = None):
    """逐帧处理(与应用端逐帧语义一致)。返回 (是否告警, 处理帧数, 最大躺倒证据)。"""
    cap = cv2.VideoCapture(path)
    if sm is None:
        sm = FallStateMachine()
    alarm = []
    max_lying = 0
    total = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue
        res = vision.process_frame(frame, track=True)
        a = sm.update(res.persons)
        if a:
            alarm.extend(a)
        if sm._lying_count:
            max_lying = max(max_lying, max(sm._lying_count.values()))
        total += 1
        idx += 1
    cap.release()
    return bool(alarm), total, max_lying


def main():
    from src.core.detector import SafetyVision

    # 轻量组装:仅姿态模型(倒地判定不依赖头盔模型)
    vision = SafetyVision(helmet_on=False)
    fall_dir = UR_FALL_DIR / "falls"
    adl_dir = UR_FALL_DIR / "adl"

    tp, tn = 0, 0
    fall_total, adl_total = 0, 0
    details = []
    for v in sorted(fall_dir.glob("*.mp4")):
        hit, total, mx = process_video(str(v), vision)
        fall_total += 1
        if hit:
            tp += 1
        details.append(("FALL", v.name, "TP" if hit else "MISS", mx))
        print(f"[FALL] {v.name}: {'检出' if hit else '漏检'}  峰值躺倒计数={mx}")
    for v in sorted(adl_dir.glob("*.mp4")):
        hit, total, mx = process_video(str(v), vision)
        adl_total += 1
        if hit:
            tn += 1
        print(f"[ADL ] {v.name}: {'误报' if hit else '正常'}  峰值躺倒计数={mx}")

    recall = tp / fall_total if fall_total else 0
    print("\n========== 评估结果 ==========")
    print(f"跌倒检出率(召回): {tp}/{fall_total} = {recall:.2%}")
    print(f"ADL 误报视频: {tn}/{adl_total} = {tn/adl_total:.2%} (越低越好)")
    for d in details:
        print("  ", d)


if __name__ == "__main__":
    main()