"""
用 CLIP 对两个类别的检测框进行语义校验,确定 class 0/1 哪个是"戴安全帽"。
用法: python scripts/verify_classes.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HELMET_YOLO, CLIP_MODEL

TEXTS = [
    "a worker wearing a safety helmet on head",
    "a worker with bare head without any helmet",
]


def main():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    img_dir = HELMET_YOLO / "images" / "train"
    lbl_dir = HELMET_YOLO / "labels" / "train"
    import random
    random.seed(0)
    files = sorted(img_dir.glob("*.jpg"))
    scores = {0: [], 1: []}
    n = min(150, len(files))
    for f in random.sample(files, n):
        lbl = lbl_dir / (f.stem + ".txt")
        if not lbl.exists():
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        for line in lbl.read_text(errors="ignore").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cls = int(p[0]); cx, cy, bw, bh = map(float, p[1:5])
            x1 = int((cx-bw/2)*w); y1 = int((cy-bh/2)*h)
            x2 = int((cx+bw/2)*w); y2 = int((cy+bh/2)*h)
            y1 = max(0, y1 - int(bh*h*0.3))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2-x1 < 10 or y2-y1 < 10:
                continue
            crop = img[y1:y2, x1:x2]
            inputs = processor(images=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                               text=TEXTS, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**inputs)
            prob = torch.softmax(out.logits_per_image[0], dim=-1).cpu().numpy()
            scores[cls].append(prob)
    print("class0 平均 P(戴帽)=%.2f P(未戴)=%.2f" % tuple(np.mean(scores[0], axis=0)))
    print("class1 平均 P(戴帽)=%.2f P(未戴)=%.2f" % tuple(np.mean(scores[1], axis=0)))
    for c, name in ((0, "class0"), (1, "class1")):
        arr = np.array(scores[c])
        print(f"{name}: {(arr[:,0] > arr[:,1]).mean():.1%} 的样本被判为'戴安全帽'")


if __name__ == "__main__":
    main()