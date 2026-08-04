"""
训练进度监控:打印 ASCII 进度条 + 最近验证指标表,并刷新 results.png 曲线图。

用法: python scripts/show_progress.py
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RUNS_DIR

RUN = RUNS_DIR / "helmet" / "yolo11s_40ep"
LOG = RUNS_DIR / "helmet" / "train_console.log"
CSV = RUN / "results.csv"
PNG = RUN / "results.png"
EP_TOTAL = 40
SECONDS_PER_EPOCH = 6.5 * 60


def parse_progress() -> tuple:
    lines = [l for l in LOG.read_text(errors="replace").splitlines()
             if re.search(r"\d+/\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+\d+\s+640:\s+\d+%", l)]
    if not lines:
        return None
    m = re.search(r"\s(\d+)/(\d+)\s.*?640:\s+(\d+)%", lines[-1])
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def bar(pct: float, width: int = 32) -> str:
    full = int(width * pct / 100)
    return "#" * full + "-" * (width - full)


def print_table():
    if not (CSV.exists() and CSV.stat().st_size > 0):
        print("(验证指标尚未写入,首个 epoch 结束后显示)")
        return
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    print(f"\n{'epoch':>5} {'mAP50':>7} {'mAP50-95':>8} {'P':>6} {'R':>6} {'val_box':>8}")
    for r in rows:
        print(f"{r['epoch']:>5} {float(r['metrics/mAP50(B)']):>7.3f} "
              f"{float(r['metrics/mAP50-95(B)']):>8.3f} {float(r['metrics/precision(B)']):>6.3f} "
              f"{float(r['metrics/recall(B)']):>6.3f} {float(r['val/box_loss']):>8.3f}")
    best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    print(f"最优 mAP50 = {float(best['metrics/mAP50(B)']):.3f} @ epoch {best['epoch']}")


def plot(rows):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    ep = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    keys = [("metrics/precision(B)", "P"), ("metrics/recall(B)", "R"),
            ("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")]
    for k, lbl in keys:
        axes[0][0].plot(ep, [float(r[k]) for r in rows], label=lbl)
    axes[0][0].set_title("验证指标 (P / R / mAP)"); axes[0][0].legend(); axes[0][0].grid(alpha=0.3)
    for k, lbl in [("train/box_loss", "box"), ("train/cls_loss", "cls"), ("train/dfl_loss", "dfl")]:
        axes[0][1].plot(ep, [float(r[k]) for r in rows], label=lbl)
    axes[0][1].set_title("训练损失"); axes[0][1].legend(); axes[0][1].grid(alpha=0.3)
    for k, lbl in [("val/box_loss", "val box"), ("val/cls_loss", "val cls")]:
        axes[1][0].plot(ep, [float(r[k]) for r in rows], label=lbl)
    axes[1][0].set_title("验证损失"); axes[1][0].legend(); axes[1][0].grid(alpha=0.3)
    axes[1][1].plot(ep, [float(r["lr/pg0"]) for r in rows], color="#8c564b")
    axes[1][1].set_title("学习率 lr/pg0"); axes[1][1].grid(alpha=0.3)
    best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    fig.suptitle(f"YOLO11s 安全帽训练进度 (epoch {ep[-1]}/{EP_TOTAL}, 最优 mAP50="
                 f"{float(best['metrics/mAP50(B)']):.3f} @e{best['epoch']})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PNG, dpi=100)
    print(f"\n曲线图已刷新 -> {PNG}")


def main():
    pr = parse_progress()
    if pr:
        ep_now, ep_total, pct = pr
        overall = (ep_now - 1 + pct / 100) / ep_total * 100
        remain = (ep_total - ep_now + (100 - pct) / 100) * SECONDS_PER_EPOCH / 60
        print(f"Epoch {ep_now:>2}/{ep_total}  {bar(pct)} {pct:>3}%")
        print(f"Overall       {bar(overall)} {overall:5.1f}%   预计剩余 ~{remain:.0f} 分钟")
    else:
        print("训练尚未进入循环")
    print_table()
    if CSV.exists() and CSV.stat().st_size > 0:
        rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
        plot(rows)


if __name__ == "__main__":
    main()