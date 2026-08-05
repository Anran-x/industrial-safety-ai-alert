"""
后台 epoch 监视器:每完成一个 epoch,把进度快照写入 epoch_updates.log。
持续运行直到训练进程退出。
用法: python scripts/epoch_watch.py
"""
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RUNS_DIR

LOG = RUNS_DIR / "helmet" / "train_console.log"
CSV = RUNS_DIR / "helmet" / "yolo11s_40ep" / "results.csv"
UPDATE_LOG = RUNS_DIR / "helmet" / "epoch_updates.log"


def last_epoch() -> int:
    if not (CSV.exists() and CSV.stat().st_size > 0):
        return 0
    lines = CSV.read_text(errors="replace").splitlines()
    return int(lines[-1].split(",")[0]) if len(lines) > 1 else 0


def snapshot() -> str:
    m = re.search(r"\s(\d+)/(\d+)\s.*?640:\s+(\d+)%", LOG.read_text(errors="replace"))
    ep_now, pct = (int(m.group(1)), int(m.group(3))) if m else (0, 0)
    lines = CSV.read_text(errors="replace").splitlines() if CSV.exists() else []
    if len(lines) < 2:
        return f"epoch {ep_now}/40 本轮{len(lines)-1}个验证点 0%"
    last = lines[-1].split(",")
    return (f"epoch {ep_now}/40 ({pct}%) | 最新验证 e{last[0]}: "
            f"mAP50={last[7]} mAP50-95={last[8]} P={last[5]} R={last[6]}")

seen = last_epoch()
with open(UPDATE_LOG, "a", encoding="utf-8") as f:
    f.write("=== epoch monitor started ===\n")
while True:
    cur = last_epoch()
    if cur > seen:
        seen = cur
        msg = f"{time.strftime('%H:%M:%S')} [EPOCH {cur} 完成] {snapshot()}"
        with open(UPDATE_LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    if cur >= 40:
        break
    time.sleep(30)