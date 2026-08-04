"""
Gradio Web 演示端:上传/摄像头视频 -> 实时检测 -> 告警面板 + 事件记录。

运行:  python -m src.app
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALERT_SCREENSHOT_DIR, INTRUSION_ROI
from src.core.detector import SafetyVision
from src.behavior.fall_detector import FallStateMachine
from src.behavior.intrusion_detector import IntrusionDetector
from src.multimodal.clip_verifier import ClipVerifier
from src.multimodal.report_gen import AlertEvent, ReportGenerator
from src.storage import AlertStore

import gradio as gr


class DemoApp:
    def __init__(self):
        self.vision = None
        self.fall_sm = FallStateMachine()
        self.intrusion = IntrusionDetector()
        self.clip = None
        self.report = ReportGenerator()
        self.store = AlertStore()
        self.alerts: list = []
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

    def _draw(self, frame, result, roi_on=True):
        img = frame.copy()
        if roi_on:
            pts = np.array([(x * img.shape[1], y * img.shape[0]) for x, y in self.intrusion.roi], np.int32)
            cv2.polylines(img, [pts], True, (0, 200, 255), 2)
        for hb in result.heads:
            color = (0, 0, 255) if hb.cls == 0 else (0, 255, 0)
            cv2.rectangle(img, (int(hb.x1), int(hb.y1)), (int(hb.x2), int(hb.y2)), color, 2)
            label = "NO_HELMET" if hb.cls == 0 else "HELMET"
            cv2.putText(img, f"{label} {hb.conf:.2f}", (int(hb.x1), max(18, int(hb.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for p in result.persons:
            b = p.box
            cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (255, 0, 0), 2)
            cv2.putText(img, f"P#{b.track_id} {b.conf:.2f}", (int(b.x1), int(b.y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if result.fall:
            cv2.putText(img, "FALL ALERT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return img

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

    def process_video(self, video_path, roi_str="", use_clip=True, max_frames=0):
        self._ensure_models()
        if roi_str.strip():
            try:
                pts = [tuple(float(v) for v in p.split(",")) for p in roi_str.split(";")]
                self.intrusion.set_roi(pts)
            except Exception:
                pass
        cap = cv2.VideoCapture(video_path)
        out_frames = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and frame_idx >= max_frames:
                break
            res = self.vision.process_frame(frame)
            fall_ids = self.fall_sm.update(res.persons)
            res.fall = bool(fall_ids)
            intr = self.intrusion.update(res.persons, frame.shape[:2])
            # 头盔告警
            for hb in res.heads:
                if hb.cls == 0:
                    use_clip_now = use_clip and self.clip is not None
                    no_helmet = True
                    conf = hb.conf
                    if use_clip_now and 0.3 <= hb.conf <= 0.6:
                        no_helmet, conf = self.clip.verify(frame, hb, hb.conf)
                    if no_helmet:
                        ev = AlertEvent("NO_HELMET", self.report.now(), conf,
                                        track_id=hb.track_id)
                        self._fire(ev, frame)
            for tid, _t in fall_ids:
                ev = AlertEvent("FALL", self.report.now(), 0.9, track_id=tid,
                                detail=f"连续多帧姿态水平判定")
                self._fire(ev, frame)
            for tid, _t in intr:
                ev = AlertEvent("INTRUSION", self.report.now(), 0.85, track_id=tid,
                                detail="轨迹中心进入危险区域")
                self._fire(ev, frame)

            out_frames.append(self._draw(frame, res))
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"processed {frame_idx} frames", flush=True)
        cap.release()
        if not out_frames:
            return None, [], 0
        h, w = out_frames[0].shape[:2]
        fps = 20
        tmp = PROJECT_ROOT / "results" / "demo_output.mp4"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in out_frames:
            writer.write(f)
        writer.release()
        return str(tmp), self.alerts[-20:], frame_idx


def build_ui():
    app = DemoApp()

    with gr.Blocks(title="工业安全 AI 智能预警系统") as demo:
        gr.Markdown("# 工业安全 AI 智能预警原型系统\n安全帽佩戴 / 人员倒地 / 危险区域闯入 · YOLO11n + ByteTrack + CLIP")
        with gr.Row():
            with gr.Column(scale=2):
                video = gr.Video(label="输入视频", format="mp4")
                run_btn = gr.Button("开始检测", variant="primary")
            with gr.Column(scale=3):
                out_video = gr.Video(label="检测结果")
                alert_list = gr.Dataframe(headers=["告警"], label="告警记录")
                info = gr.Textbox(label="处理信息")
        with gr.Row():
            roi = gr.Textbox(value="0.05,0.30;0.95,0.30;0.95,0.75;0.05,0.75",
                             label="闯入区域 ROI(归一化多边形,格式 x,y;x,y;...)")
            use_clip = gr.Checkbox(value=True, label="启用 CLIP 兜底复核")
            max_frames = gr.Slider(0, 3000, value=600, step=50, label="处理帧数(0=全部)")

        run_btn.click(app.process_video, [video, roi, use_clip, max_frames],
                      [out_video, alert_list, info])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, show_error=True)