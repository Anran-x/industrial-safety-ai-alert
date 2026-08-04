"""
CLIP 兜底复核对比实验:在低置信度区间的头盔检测结果上,
对比"仅 YOLO"与"YOLO+CLIP 复核"的误报情况,输出统计报告。

用法: python scripts/eval_clip.py --images <目录> 
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HELMET_YOLO
from src.multimodal.clip_verifier import ClipVerifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, default=str(HELMET_YOLO / "images" / "valid"))
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    from src.core.detector import SafetyVision

    vision = SafetyVision()  # 头盔模型
    verifier = ClipVerifier()

    img_dir = Path(args.images)
    files = sorted(img_dir.glob("*.jpg"))[: args.limit]
    stats = {"low_conf_dets": 0, "clip_flip_to_no_helmet": 0, "clip_flip_to_helmet": 0, "unchanged": 0}
    examples = []
    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        res = vision.process_frame(frame, track=False)
        for hb in res.heads:
            if 0.25 <= hb.conf <= 0.65:  # 低置信度区间
                stats["low_conf_dets"] += 1
                before = hb.cls == 0
                after, p_no = verifier.verify(frame, hb, hb.conf)
                if before != after:
                    if after:
                        stats["clip_flip_to_no_helmet"] += 1
                    else:
                        stats["clip_flip_to_helmet"] += 1
                    examples.append((f.name, round(hb.conf, 3), round(p_no, 3), "->no_helmet" if after else "->helmet"))
                else:
                    stats["unchanged"] += 1

    print("========== CLIP 对比实验 ==========")
    print(f"低置信度检测框总数: {stats['low_conf_dets']}")
    print(f"CLIP 纠正为未戴安全帽(减少漏报): {stats['clip_flip_to_no_helmet']}")
    print(f"CLIP 纠正为已戴安全帽(减少误报): {stats['clip_flip_to_helmet']}")
    print(f"与 YOLO 判断一致: {stats['unchanged']}")
    if stats["low_conf_dets"]:
        flips = stats["clip_flip_to_no_helmet"] + stats["clip_flip_to_helmet"]
        print(f"低置信度区间修正比例: {flips}/{stats['low_conf_dets']} = {flips/stats['low_conf_dets']:.1%}")
    for e in examples[:15]:
        print("  ", e)


if __name__ == "__main__":
    main()