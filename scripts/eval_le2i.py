"""
在 Le2i Fall Detection 帧段上评估倒地检测规则(45° 斜装视角)。

数据来源:Le2i 官方帧镜像(Home/Coffee/Office/Lecture,320x240@25FPS 的连续帧段,
train+val 共 1.1 万帧,按 类别/视频/帧号 组织)。官方完整视频 8.95GB 需学术网络
下载(dataUBFC,CC BY-NC-SA),本脚本不依赖完整视频,用帧段即可复现:
"45° 斜装下几何规则存在参数甜区(1.2/6 帧:Fall 57%,Lie 48%,Likefall 0% 误报)"。

评估口径:
- 每视频取至多 MAX_FRAMES_PER_VIDEO 帧,按帧号排序模拟 25FPS 短片,
  喂完整状态机(与 scripts/eval_fall.py 同语义),触发告警即检出;
- Fall/Lie 为应检出组,Likefall(假倒)/Stand(站立)为误报组;
- 网格扫描 (lying_aspect x confirm_frames),报告各参数组触发率。

用法: python scripts/eval_le2i.py            # 默认网格扫描(自动检测+缓存)
     python scripts/eval_le2i.py --reuse     # 复用已缓存特征,只重扫参数
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR
from src.core.models import Box, Person

LE2I_DIR = DATA_DIR / "fall_detection" / "le2i"
CACHE = PROJECT_ROOT / "runs" / "le2i_feat_cache.json"
POSE_WEIGHTS = PROJECT_ROOT / "models" / "yolo11s-pose.pt"

MAX_FRAMES_PER_VIDEO = 32   # 帧段最长 32 帧(官方帧镜像按帧号连续切段),不截断
MAX_VIDEOS_PER_CLASS = 40
GRID_ASPECT = (1.4, 1.2, 1.0, 0.9)
GRID_CONFIRM = (8, 6, 5)


def detect_video(model, vdir: Path):
    """返回按帧号排序的 [((x1,y1,x2,y2), conf)] 最大人体框(单目标场景取最大框)。"""
    feats = []
    for ip in sorted(vdir.glob("*.jpg"))[:MAX_FRAMES_PER_VIDEO]:
        r = model.predict(str(ip), conf=0.30, imgsz=640, verbose=False)[0]
        best = None
        for b in r.boxes:
            if int(b.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            w, h = x2 - x1, y2 - y1
            if h > 10 and (best is None or w * h > best[0][2] * best[0][3]):
                best = ((x1, y1, x2, y2), float(b.conf[0].item()))
        feats.append(best)
    return feats


def run_state_machine(feats, lying_aspect, confirm_frames):
    """用与生产端一致的 FallStateMachine 回放帧段,触发告警即返回 True。"""
    from src.behavior.fall_detector import FallStateMachine
    sm = FallStateMachine(lying_aspect=lying_aspect,
                          confirm_frames=confirm_frames, fps=25.0)
    for ft in feats:
        persons = []
        if ft is not None:
            (x1, y1, x2, y2), conf = ft
            persons = [Person(box=Box(x1, y1, x2, y2, conf, 0, track_id=1), kps=None)]
        if sm.update(persons):
            return True
    return False


def collect(force: bool):
    if CACHE.exists() and not force:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    from ultralytics import YOLO
    model = YOLO(str(POSE_WEIGHTS), task="pose")
    results = {}
    for split in ("train",):
        for cls in ("Fall", "Lie", "Likefall", "Stand"):
            cls_dir = LE2I_DIR / split / cls
            if not cls_dir.exists():
                continue
            items = []
            for vd in sorted(d for d in cls_dir.iterdir() if d.is_dir())[:MAX_VIDEOS_PER_CLASS]:
                feats = detect_video(model, vd)
                if feats:
                    items.append({"video": vd.name, "frames": len(feats), "feats": feats})
                print(f"[{split}/{cls}] {vd.name} {len(feats)} 帧", flush=True)
            results[f"{split}/{cls}"] = items
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(results), encoding="utf-8")
    print(f"特征缓存 -> {CACHE}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="复用已缓存特征,仅重扫参数")
    args = ap.parse_args()

    results = collect(force=not args.reuse)
    order = ("train/Fall", "train/Lie", "train/Likefall", "train/Stand")
    print("\n===== Le2i 45° 斜装:端到端参数网格(触发率) =====")
    for cf in GRID_CONFIRM:
        for th in GRID_ASPECT:
            cells = []
            for key in order:
                items = results.get(key, [])
                n = len(items)
                hit = sum(1 for it in items if run_state_machine(it["feats"], th, cf))
                short = key.split("/")[1]
                cells.append(f"{short}={hit}/{n}({hit / max(n, 1) * 100:.0f}%)")
            print(f"confirm={cf} aspect>={th:.1f}: " + "  ".join(cells))
    print("\n说明:帧段仅覆盖跌倒核心时段(16-32 帧,无前后站立上下文),属召回下界;")
    print("完整视频复核需官方 FallDataset.zip(学术网络),见 docs/eval_report.md §2.1。")


if __name__ == "__main__":
    main()
