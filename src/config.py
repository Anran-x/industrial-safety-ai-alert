import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = PROJECT_ROOT / "runs"

# ---------- 路径 ----------
HELMET_DATA = DATA_DIR / "helmet"
HELMET_YOLO = HELMET_DATA / "yolo"
HELMET_DATA_YAML = HELMET_YOLO / "data.yaml"
HELMET_WEIGHTS = MODEL_DIR / "helmet_yolo11s.pt"
UR_FALL_DIR = DATA_DIR / "fall_detection" / "ur_fall"
DEMO_VIDEO = DATA_DIR / "demo" / "demo_safety.mp4"

# ---------- 检测配置 ----------
YOLO_MODEL = str(MODEL_DIR / "yolo11s.pt")
POSE_MODEL = str(MODEL_DIR / "yolo11s-pose.pt")
CLIP_MODEL = str(MODEL_DIR / "clip-vit-base-patch32")  # transformers 本地权重(已预下载)
HELMET_CONF = 0.35       # 检测置信度下限
HELMET_IMGSZ = 960       # 推理分辨率(640→960:全量test召回94.8%→96.1%,戴帽类混淆减半;小目标受益)
HELMET_ALERT_CONF = 0.45  # 触发告警的置信度
HELMET_CLIP_LOW = 0.35    # 低置信度区间下限(交给 CLIP 复核)
HELMET_CLIP_HIGH = 0.55   # 低置信度区间上限

# ---------- 倒地判定配置 ----------
# 已验证的三视角实测(见 docs/eval_report.md §2):
# - 平视(URFD cam0):aspect=1.4 / confirm=8,检出 24/30(80%),ADL 误报 3/40
# - 正俯视(URFD cam1):几何阈值无甜区,调参后召回上限 ~30%,规则法不适用
# - 45° 斜装(Le2i 帧段):aspect=1.2 / confirm=6 达 57% 且 Likefall 误报 0%
FALL_LYING_ASPECT = 1.4        # 人体框宽高比 > 该值视为"躺倒"
FALL_CONFIRM_FRAMES = 8        # 连续 N 帧判定倒地才告警(抑制瞬时误报)
FALL_RESET_FRAMES = 30         # 离地(站立)连续 M 帧后复位状态

# ---------- 闯入判定配置 ----------
INTRUSION_CONFIRM_FRAMES = 5   # 轨迹中心连续 N 帧在区域内才告警
INTRUSION_ROI = [(0.05, 0.30), (0.95, 0.30), (0.95, 0.75), (0.05, 0.75)]  # 归一化多边形(演示默认,可在 UI 配置)

# ---------- 视频处理 ----------
TARGET_FPS = 15
MAX_FRAME_WIDTH = 1280
CPU_FALLBACK = True            # 无 GPU 时自动用 CPU

# ---------- 告警配置 ----------
ALERT_DB = PROJECT_ROOT / "results" / "alerts.db"
ALERT_COOLDOWN = 5.0           # 同一目标同类告警最短间隔(秒)
ALERT_SCREENSHOT_DIR = RESULTS_DIR / "screenshots"

for _d in (RESULTS_DIR, ALERT_SCREENSHOT_DIR, MODEL_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
