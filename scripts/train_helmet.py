"""
训练安全帽检测模型(YOLO11n)。

基线策略:1050Ti 4GB 显存限制下,先以子集训练出基线模型,
全量数据可后续微调。指标输出到 runs/helmet/result。

用法: python scripts/train_helmet.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HELMET_DATA_YAML, HELMET_WEIGHTS, RUNS_DIR


def main():
    from ultralytics import YOLO
    import torch

    model = YOLO("yolo11n.pt")
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"训练设备: {device}, 显存: {torch.cuda.get_device_name(0) if device=='0' else 'CPU'}")
    results = model.train(
        data=str(HELMET_DATA_YAML),
        epochs=40,
        imgsz=640,
        batch=8,
        device=device,
        workers=4,
        project=str(RUNS_DIR / "helmet"),
        name="yolo11n_50ep",
        exist_ok=True,
        patience=0,
        cache=True,
    )
    best = str(RUNS_DIR / "helmet" / "yolo11n_50ep" / "weights" / "best.pt")
    HELMET_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(best, HELMET_WEIGHTS)
    print(f"best model -> {HELMET_WEIGHTS}")


if __name__ == "__main__":
    main()