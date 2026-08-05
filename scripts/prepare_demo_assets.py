"""
制作演示素材:
1. data/demo/demo_helmet.mp4 —— SHWD test 图拼成的头盔检测演示(含戴/未戴两类)
2. 确认 UR Fall 倒地演示视频清单
用法: python scripts/prepare_demo_assets.py
"""
import random
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HELMET_YOLO, UR_FALL_DIR, DEMO_VIDEO, DATA_DIR


def _open_writer(path, fps, size):
    for fourcc in ("avc1", "h264", "mp4v"):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if vw.isOpened():
            return vw, fourcc
        vw.release()
    return None, None


def make_helmet_clip(out: Path, n_per_class: int = 6, fps: int = 10):
    random.seed(42)
    img_dir = HELMET_YOLO / "images" / "test"
    lbl_dir = HELMET_YOLO / "labels" / "test"
    files = sorted(img_dir.glob("*.jpg"))
    picks = []
    for cls in (0, 1):
        cands = []
        for f in files:
            lbl = lbl_dir / (f.stem + ".txt")
            if lbl.exists() and any(line.split()[0] == str(cls) for line in lbl.read_text(errors="ignore").splitlines()):
                cands.append(f)
        picks += random.sample(cands, min(n_per_class, len(cands)))
    random.shuffle(picks)

    writer = None
    W = H = None
    for f in picks:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        if W is None:
            W, H = w, h
            writer, codec = _open_writer(out, fps, (W, H))
            if writer is None:
                raise RuntimeError("no video encoder available")
            print(f"  helmet clip 编码: {codec}")
        if h != H or w != W:
            img = cv2.resize(img, (W, H))  # 统一尺寸,否则 mp4v 编码会丢帧
        for _ in range(fps):
            writer.write(img)
    if writer is not None:
        writer.release()
    print(f"helmet demo -> {out} ({len(picks)} images)")


def make_fall_demo_clip(src: Path, out: Path):
    """把 UR Fall 演示片重编码为 H.264,便于浏览器直接预览。"""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"  [skip] 打不开 {src.name}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            writer, codec = _open_writer(out, fps, (frame.shape[1], frame.shape[0]))
            if writer is None:
                print(f"  [skip] 无编码器 {src.name}")
                return
            print(f"  fall demo 编码: {codec}")
        writer.write(frame)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    print(f"fall demo -> {out} ({n} frames)")


def list_fall_demos():
    falls = sorted((UR_FALL_DIR / "falls").glob("*.mp4"))
    print("UR Fall 演示候选(前10):")
    for v in falls[:10]:
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  {v.name} ({n} frames)")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    make_helmet_clip(DEMO_VIDEO.parent / "demo_helmet.mp4")
    make_fall_demo_clip(UR_FALL_DIR / "falls" / "fall-10-cam0.mp4",
                        DEMO_VIDEO.parent / "fall-10-demo.mp4")
    list_fall_demos()