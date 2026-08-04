"""
清洗并整理 SHWD 安全帽数据集为 YOLO 训练格式,生成 data.yaml,
并输出每个类别的图片统计与'类别-颜色'启发式校验(辅助确认 0/1 语义)。

用法: python scripts/prepare_helmet_yolo.py
"""
import shutil
import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HELMET_YOLO, HELMET_DATA_YAML, HELMET_DATA

RAW = HELMET_DATA / "raw" / "Phat_project-3"
CLASS_NAMES = ["helmet", "no_helmet"]  # 待视觉确认


def read_labels(lbl_path: Path) -> list:
    lines = []
    for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            lines.append((int(parts[0]), [float(p) for p in parts[1:5]]))
    return lines


def build_layout():
    for split in ("train", "valid", "test"):
        img_dir = HELMET_YOLO / "images" / split
        lbl_dir = HELMET_YOLO / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        src_img = RAW / split / "images"
        src_lbl = RAW / split / "labels"
        if not src_img.exists():
            continue
        for img in sorted(src_img.glob("*.jpg")):
            lbl = src_lbl / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(img, img_dir / img.name)
                shutil.copy2(lbl, lbl_dir / lbl.name)
            else:
                print(f"[warn] {split}/{img.name} 无标签,跳过")

    data_yaml = HELMET_DATA_YAML
    data_yaml.write_text(
        f"path: {HELMET_YOLO.as_posix()}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n"
        "names:\n"
        "  0: helmet\n"
        "  1: no_helmet\n",
        encoding="utf-8",
    )
    print(f"data.yaml -> {data_yaml}")


def report_stats():
    for split in ("train", "valid", "test"):
        lbl_dir = HELMET_YOLO / "labels" / split
        img_dir = HELMET_YOLO / "images" / split
        if not lbl_dir.exists():
            continue
        cnt = Counter()
        objs = 0
        for lbl in lbl_dir.glob("*.txt"):
            for cls, _ in read_labels(lbl):
                cnt[cls] += 1
                objs += 1
        nimg = len(list(img_dir.glob("*.jpg")))
        nlab = len(list(lbl_dir.glob("*.txt")))
        print(f"[{split}] images={nimg} labels={nlab} objects={objs} "
              f"class0={cnt[0]} class1={cnt[1]}")


def color_check(n_samples: int = 400):
    """每类采样检测框,统计 HSV 均值,辅助判断类别语义。"""
    stats = {0: [], 1: []}
    lbl_dir = HELMET_YOLO / "labels" / "train"
    img_dir = HELMET_YOLO / "images" / "train"
    files = sorted(img_dir.glob("*.jpg"))
    np.random.seed(42)
    chosen = np.random.choice(len(files), min(n_samples, len(files)), replace=False)
    for i in chosen:
        img_path = files[i]
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for cls, (cx, cy, bw, bh) in read_labels(lbl_path):
            if cls not in stats:
                continue
            x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            crop = img[y1:y2, x1:x2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            stats[cls].append(hsv.mean(axis=(0, 1)))
    for cls, arr in stats.items():
        if arr:
            a = np.array(arr)
            print(f"class {cls} mean HSV: H={a[:,0].mean():.1f} S={a[:,1].mean():.1f} V={a[:,2].mean():.1f}")


if __name__ == "__main__":
    build_layout()
    report_stats()
    color_check()