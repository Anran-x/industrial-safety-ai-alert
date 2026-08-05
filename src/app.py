"""
Gradio Web 演示端:上传视频 -> 实时检测 -> 分模块告警面板 + 告警图片墙 + 事件记录。

运行:  python -m src.app
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALERT_SCREENSHOT_DIR, INTRUSION_ROI, TARGET_FPS
from src.core.detector import SafetyVision
from src.behavior.fall_detector import FallStateMachine
from src.behavior.intrusion_detector import IntrusionDetector
from src.multimodal.clip_verifier import ClipVerifier
from src.multimodal.report_gen import AlertEvent, ReportGenerator
from src.storage import AlertStore

import gradio as gr

HELMET_DEMO = PROJECT_ROOT / "data" / "demo" / "demo_helmet.mp4"
FALL_DEMO = PROJECT_ROOT / "data" / "demo" / "fall-10-demo.mp4"
OVERHEAD_DEMO = PROJECT_ROOT / "data" / "demo" / "fall-overhead-demo.mp4"
OUT_VIDEO = PROJECT_ROOT / "results" / "demo_output.mp4"

CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto; }
#page-title {
    background: linear-gradient(135deg, #1a2a4a 0%, #1e3a5f 55%, #2b5f8a 100%);
    color: #fff; padding: 22px 28px; border-radius: 14px; margin-bottom: 6px;
    box-shadow: 0 4px 14px rgba(20,40,80,.25);
}
#page-title h1 { margin: 0; font-size: 26px; letter-spacing: 1px; }
#page-title p { margin: 6px 0 0; opacity: .85; font-size: 14px; }
.module-card { border: 1px solid #e3e8f0; border-radius: 12px; padding: 12px 14px;
    background: #fafbfe; box-shadow: 0 2px 8px rgba(20,40,80,.06); }
.legend-item { display: inline-flex; align-items: center; gap: 6px; margin-right: 16px; }
.legend-box { width: 18px; height: 12px; border-radius: 3px; display: inline-block; }
footer { text-align: center; color: #8892a3; font-size: 12px; margin-top: 10px; }
"""

LEGEND_HTML = """
<div class="module-card" style="margin-top:10px">
<b>画面标注图例</b> &nbsp;
<span class="legend-item"><span class="legend-box" style="background:#e02020"></span>红框=未戴安全帽</span>
<span class="legend-item"><span class="legend-box" style="background:#1faf50"></span>绿框=已戴安全帽</span>
<span class="legend-item"><span class="legend-box" style="background:#2f6bff"></span>蓝框=人员(含跟踪ID)</span>
<span class="legend-item"><span class="legend-box" style="background:#ffc400"></span>黄框=危险区域(ROI,整片固定)</span>
<span class="legend-item" style="color:#d02020;font-weight:600">画面顶部红字=倒地告警</span>
</div>
"""


class DemoApp:
    def __init__(self):
        self.vision = None
        self.fall_sm = FallStateMachine(fps=TARGET_FPS)
        self.intrusion = IntrusionDetector()
        self.clip = None
        self.report = ReportGenerator()
        self.store = AlertStore()
        self.alerts: list = []
        self._run_shots: list = []
        self._last_alert_ts: dict = {}

    def _ensure_models(self):
        if self.vision is None:
            self.vision = SafetyVision()
        if self.clip is None:
            try:
                self.clip = ClipVerifier()
            except Exception as e:
                print(f"[warn] CLIP 加载失败,跳过兜底复核: {e}")
                self.clip = None

    def _draw(self, frame, result, roi_on=True, helmet_on=True, person_on=True):
        img = frame.copy()
        if roi_on:
            pts = np.array([(x * img.shape[1], y * img.shape[0]) for x, y in self.intrusion.roi], np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 215, 255))
            img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
            cv2.polylines(img, [pts], True, (0, 215, 255), 3)
            cv2.putText(img, "DANGER ZONE", (pts[:, 0].min(), max(24, pts[:, 1].min() - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
        if person_on:
            for p in result.persons:
                b = p.box
                cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (255, 128, 0), 2)
                cv2.putText(img, f"P#{b.track_id} {b.conf:.2f}", (int(b.x1), int(b.y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
        if helmet_on:
            for hb in result.heads:
                no_helmet = hb.cls == 0
                color = (40, 40, 235) if no_helmet else (80, 175, 60)
                thick = 3 if no_helmet else 2
                cv2.rectangle(img, (int(hb.x1), int(hb.y1)), (int(hb.x2), int(hb.y2)), color, thick)
                label = f"NO_HELMET {hb.conf:.2f}" if no_helmet else f"HELMET {hb.conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                tx, ty = int(hb.x1), max(18, int(hb.y1) - 8)
                cv2.rectangle(img, (tx, ty - th - 6), (tx + tw, ty + 2), color, -1)
                cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if result.fall:
            banner = "FALL ALERT"
            (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            bx = (img.shape[1] - bw) // 2
            cv2.rectangle(img, (bx - 14, 22), (bx + bw + 14, 22 + bh + 16), (40, 40, 235), -1)
            cv2.putText(img, banner, (bx, 22 + bh + 4), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        return img

    @staticmethod
    def _head_to_person(hb, persons):
        best, best_overlap = None, -1
        nearest, nearest_d = None, None
        hc = ((hb.x1 + hb.x2) / 2, (hb.y1 + hb.y2) / 2)
        for p in persons:
            b = p.box
            x_overlap = min(hb.x2, b.x2) - max(hb.x1, b.x1)
            if x_overlap > 0:
                p_h = b.y2 - b.y1
                if (b.y1 - p_h) <= hb.y1 and hb.y2 <= (b.y2 + p_h):
                    if x_overlap > best_overlap:
                        best_overlap, best = x_overlap, b.track_id
            bc = ((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2)
            d = (hc[0] - bc[0]) ** 2 + (hc[1] - bc[1]) ** 2
            if nearest_d is None or d < nearest_d:
                nearest_d, nearest = d, b.track_id
        return best if best is not None else nearest

    def _fire(self, ev: AlertEvent, frame):
        now = time.time()
        key = (ev.alert_type, ev.track_id)
        if now - self._last_alert_ts.get(key, 0) < 5.0:
            return
        self._last_alert_ts[key] = now
        ALERT_SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        shot = str(ALERT_SCREENSHOT_DIR / f"{ev.timestamp.replace(':', '-')}_{ev.alert_type}.jpg")
        cv2.imwrite(shot, frame)
        ev.screenshot = shot
        self.store.insert(ev)
        self.alerts.append(self.report.build(ev))
        self._run_shots.append(shot)

    @staticmethod
    def _open_writer(path, fps, size):
        for fourcc in ("avc1", "h264", "mp4v"):
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
            if vw.isOpened():
                return vw, fourcc
            vw.release()
        return None, None

    def process_video(self, video_path, roi_str="", helmet_on=True, fall_on=True,
                      intrusion_on=True, use_clip=True, max_frames=0,
                      view_preset="平视(cam0 标定,默认)",
                      progress=gr.Progress()):
        self._ensure_models()
        if view_preset == "45° 斜装(楼道/机柜间)":
            self.fall_sm = FallStateMachine(fps=TARGET_FPS, lying_aspect=1.2,
                                            confirm_frames=6)
        else:
            self.fall_sm = FallStateMachine(fps=TARGET_FPS)
        self.alerts = []
        self._run_shots = []
        if roi_str.strip():
            try:
                pts = [tuple(float(v) for v in p.split(",")) for p in roi_str.split(";")]
                self.intrusion.set_roi(pts)
            except Exception:
                pass
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, "无法打开视频文件,请确认上传的是 mp4/avi/webm 格式。", [], [], 0, 0, 0
        frame_idx = 0
        writer, codec = None, None
        t0 = time.time()
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and frame_idx >= max_frames:
                break
            if writer is None:
                h, w = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
                writer, codec = self._open_writer(OUT_VIDEO, fps, (w, h))
                if writer is None:
                    return None, "视频编码器初始化失败(avc1/h264/mp4v 均不可用)。", [], [], 0, 0, 0
            progress((frame_idx + 1) / (max_frames or total or 1), desc=f"处理中 {frame_idx + 1} 帧")
            res = self.vision.process_frame(frame)
            drawn = self._draw(frame, res, roi_on=intrusion_on,
                               helmet_on=helmet_on, person_on=fall_on or intrusion_on)
            if fall_on:
                fall_ids = self.fall_sm.update(res.persons)
                res.fall = bool(fall_ids)
                drawn = self._draw(frame, res, roi_on=intrusion_on,
                                   helmet_on=helmet_on, person_on=fall_on or intrusion_on)
            intr = self.intrusion.update(res.persons, frame.shape[:2]) if intrusion_on else []
            if helmet_on:
                for hb in res.heads:
                    if hb.cls == 0:
                        use_clip_now = use_clip and self.clip is not None
                        no_helmet = True
                        conf = hb.conf
                        if use_clip_now and 0.3 <= hb.conf <= 0.6:
                            no_helmet, conf = self.clip.verify(frame, hb, hb.conf)
                        if no_helmet:
                            tid = self._head_to_person(hb, res.persons)
                            detail = "YOLO 检出未戴帽"
                            if use_clip_now and 0.3 <= hb.conf <= 0.6:
                                detail = "低置信度,经 CLIP 语义复核确认未戴帽"
                            ev = AlertEvent("NO_HELMET", self.report.now(), conf,
                                            track_id=tid, detail=detail)
                            self._fire(ev, drawn)
            if fall_on:
                for tid in fall_ids:
                    ev = AlertEvent("FALL", self.report.now(), 0.9, track_id=tid,
                                    detail="连续多帧姿态水平判定")
                    self._fire(ev, drawn)
            for tid, _t in intr:
                ev = AlertEvent("INTRUSION", self.report.now(), 0.85, track_id=tid,
                                detail="轨迹中心进入危险区域")
                self._fire(ev, drawn)
            writer.write(drawn)
            frame_idx += 1
        cap.release()
        if writer is not None:
            writer.release()
        if frame_idx == 0:
            return None, "视频没有任何可读帧。", [], [], 0, 0, 0

        n_fall = sum(1 for a in self.alerts if "事件: 人员倒地" in a)
        n_helmet = sum(1 for a in self.alerts if "事件: 未佩戴安全帽" in a)
        n_intr = sum(1 for a in self.alerts if "事件: 危险区域闯入" in a)
        dt = time.time() - t0
        playable = "H.264 编码,浏览器可直接预览" if codec in ("avc1", "h264") \
            else "编码器为 mp4v,浏览器可能无法预览,请下载后用播放器观看"
        info = (f"处理完成:{frame_idx} 帧,耗时 {dt:.1f}s({frame_idx / max(dt, 0.001):.1f} FPS)。"
                f"告警:未戴帽×{n_helmet}、倒地×{n_fall}、闯入×{n_intr}。输出 {OUT_VIDEO.name}({playable})。")
        df = []
        for a in self.alerts:
            lines = a.splitlines()
            info_d = {}
            for ln in lines:
                if ": " in ln:
                    k, v = ln.split(": ", 1)
                    info_d[k] = v
            df.append([
                info_d.get("时间", ""), info_d.get("事件", ""),
                info_d.get("置信度", ""), info_d.get("目标ID", ""),
                Path(info_d.get("证据截图", "")).name if info_d.get("证据截图") else "",
                info_d.get("判定依据", ""),
            ])
        return str(OUT_VIDEO), info, df, list(self._run_shots), n_helmet, n_fall, n_intr

    def all_shots(self):
        if not ALERT_SCREENSHOT_DIR.exists():
            return []
        shots = sorted(ALERT_SCREENSHOT_DIR.glob("*.jpg"), reverse=True)
        return [(str(s), s.name) for s in shots]

    def all_rows(self):
        rows = self.store.query(limit=200)
        return [[r["ts"], r["alert_type"], r["confidence"], r["track_id"],
                 Path(r["screenshot"]).name if r["screenshot"] else "", r["detail"]] for r in rows]


def build_ui():
    app = DemoApp()

    with gr.Blocks(title="工业安全 AI 智能预警系统") as demo:
        with gr.Column(elem_id="page-title"):
            gr.HTML(
                '<h1 style="color:#ffffff;margin:0;font-size:26px;letter-spacing:1px;">'
                '工业安全 AI 智能预警系统</h1>'
                '<p style="color:#cfe0f2;margin:6px 0 0;font-size:14px;">'
                '实时视频智能监控:自动识别「未戴安全帽」「人员倒地」「闯入危险区域」'
                '三类安全隐患,第一时间告警并留存证据,支持事后追溯。</p>'
            )
        gr.Markdown(LEGEND_HTML)
        with gr.Tabs():
            with gr.Tab("① 视频检测"):
                with gr.Row():
                    with gr.Column(scale=2):
                        video = gr.Video(label="① 上传监控视频(mp4/avi/webm)", format=None)
                        with gr.Row():
                            btn_helm_demo = gr.Button("载入:安全帽演示", size="sm")
                            btn_fall_demo = gr.Button("载入:倒地演示(平视)", size="sm")
                            btn_overhead_demo = gr.Button("载入:倒地演示(俯视)", size="sm")
                        with gr.Accordion("高级参数(可按模块独立开关)", open=True):
                            with gr.Row():
                                helmet_on = gr.Checkbox(value=True, label="启用安全帽检测")
                                fall_on = gr.Checkbox(value=True, label="启用倒地检测")
                                intrusion_on = gr.Checkbox(value=True, label="启用闯入检测")
                            view_preset = gr.Radio(
                                ["平视(cam0 标定,默认)", "45° 斜装(楼道/机柜间)"],
                                value="平视(cam0 标定,默认)", label="倒地检测视角预设",
                                info="45° 斜装按 Le2i 帧段验证参数(1.2/6 帧),正俯视不适用规则法")
                            roi = gr.Textbox(value="0.05,0.30;0.95,0.30;0.95,0.75;0.05,0.75",
                                             label="闯入区域 ROI(归一化多边形,x,y;x,y;...;整片视频固定)")
                            with gr.Row():
                                use_clip = gr.Checkbox(value=True, label="启用 CLIP 兜底复核")
                                max_frames = gr.Slider(0, 3000, value=600, step=50,
                                                       label="处理帧数(0=全部)")
                        run_btn = gr.Button("开始检测", variant="primary", size="lg")
                        gr.Markdown(
                            "**判定标准速览** 未戴帽:YOLO 检出 no_helmet;"
                            "倒地:人体框连续 8 帧宽高比>1.4(45° 斜装预设为 6 帧>1.2)且满足快速倒伏门控;"
                            "闯入:人员中心点连续 5 帧进入 ROI 多边形。详见「使用说明」页。")
                    with gr.Column(scale=3):
                        out_video = gr.Video(label="② 检测结果(红=未戴帽 / 绿=已戴帽 / 蓝=人员 / 黄=危险区域)",
                                             format=None)
                        info = gr.Textbox(label="处理信息", interactive=False)
                        with gr.Row():
                            stat_nh = gr.Label(value="未戴帽:—", label="安全帽")
                            stat_fall = gr.Label(value="倒地:—", label="倒地")
                            stat_intr = gr.Label(value="闯入:—", label="闯入")
                        alert_list = gr.Dataframe(
                            headers=["时间", "事件", "置信度", "目标ID", "截图", "判定依据"],
                            label="本轮告警记录", interactive=False)
                btn_helm_demo.click(lambda: str(HELMET_DEMO), None, video)
                btn_fall_demo.click(lambda: str(FALL_DEMO), None, video)
                btn_overhead_demo.click(lambda: str(OVERHEAD_DEMO), None, video)
            with gr.Tab("② 告警中心"):
                gallery = gr.Gallery(label="告警截图证据墙(全部)", columns=4, height=320,
                                     object_fit="cover")
                all_rows = gr.Dataframe(headers=["时间", "类型", "置信度", "目标ID", "截图", "判定依据"],
                                        label="历史告警(SQLite, 最近200条)", interactive=False)
                gr.Button("刷新告警中心", variant="secondary").click(
                    lambda: (app.all_shots(), app.all_rows()), None, [gallery, all_rows])
                demo.load(lambda: (app.all_shots(), app.all_rows()), None, [gallery, all_rows])
            with gr.Tab("③ 使用说明"):
                gr.Markdown(
                    """
### 三个检测模块

| 模块 | 判定标准 | 可调参数 |
|---|---|---|
| 安全帽佩戴 | YOLO11s 检出 no_helmet 类别(置信度≥0.35);低置信度[0.35,0.55]由 CLIP 复核 | 置信度阈值(HELMET_CONF) |
| 人员倒地 | 人体框宽高比 > 1.4 连续 8 帧(躺倒,平视预设);45° 斜装预设 1.2 / 6 帧;离地 30 帧复位;配快速倒伏双门控(1.8s 窗口 + 24 帧宽高比变化 ≥0.5)。**三视角实测**:平视 80%(URFD cam0)、45° 斜装 57%(Le2i 帧段)、正俯视 ~30% 且调参无甜区(URFD cam1),正俯视不适用规则法 | FALL_* 系列参数 / 视角预设 |
| 危险区域闯入 | 人员中心点(归一化)进入 ROI 多边形连续 5 帧告警;离区 15 帧复位,离区 90 帧清理 | ROI 多边形(UI 可直接改) |

### 关于危险区域(实际场景)
- ROI 用**归一化坐标**,与分辨率无关;危险区域(卸料口、配电房等)通常是**固定位置**,
  本项目默认整片视频使用同一 ROI,即"固定背景 + 固定区域"检测方式,贴合实际部署。
- 真实部署时只需按相机画面配置一次多边形即可。

### 运行与硬件
- 全链路 1050Ti 约 65ms/帧(@960,约 15 FPS);纯 CPU 模式可用(降低分辨率)。
- 输出 H.264 编码 mp4(已内置 openh264 编码器),浏览器可直接预览。
- 告警截图保存于 `results/screenshots/`,记录存 `results/alerts.db`,汇总样例见 `results/sample_report.md`。
                    """
                )
                gr.Markdown(
                    "### 演示素材\n- `data/demo/demo_helmet.mp4`:安全帽检测示例(12 张图轮播 120 帧)\n"
                    "- `data/demo/fall-10-demo.mp4`:UR Fall 倒地示例(平视 cam0)\n"
                    "- `data/demo/fall-overhead-demo.mp4`:UR Fall 倒地示例(正俯视 cam1,规则可检出段落)\n"
                    "- 视角预设:「平视」1.4/8 帧(URFD cam0 标定 80%);「45° 斜装」1.2/6 帧(Le2i 帧段 57%,Likefall 误报 0%);正俯视规则法不适用(见 docs/eval_report.md §2)\n\n"
                    "> 原型系统:指标来自公开数据集(SHWD / UR Fall),演示为实验室场景,"
                    "非产线实测,不声明生产级指标。")
        gr.Markdown("<footer>工业安全 AI 智能预警原型系统 · YOLO11s + ByteTrack + CLIP · 原型级,非产品级</footer>")

        run_btn.click(app.process_video,
                      [video, roi, helmet_on, fall_on, intrusion_on, use_clip,
                       max_frames, view_preset],
                      [out_video, info, alert_list, gallery, stat_nh, stat_fall, stat_intr])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, show_error=True,
                      theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=CSS)
