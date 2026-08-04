"""
告警事件多模态报告生成(规则模板版)。

设计说明:
- 4GB 显存无法本地运行大语言/视觉模型,为保证确定性采用结构化模板;
- 报告包含:时间、事件类型、位置(区域)、置信度、证据截图、建议动作;
- 预留 VLM 增强接口(云端 API 可按需接入),面试时说明该权衡。
"""
import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class AlertEvent:
    alert_type: str          # NO_HELMET / FALL / INTRUSION
    timestamp: str
    confidence: float
    zone: str = "默认区域"
    track_id: Optional[int] = None
    screenshot: str = ""
    detail: str = ""


SUGGESTIONS = {
    "NO_HELMET": "请立即通知现场人员佩戴安全帽,并核实人员身份与所属班组。",
    "FALL": "人员疑似倒地,请立即安排救援人员前往现场确认,并上报安全管理部门。",
    "INTRUSION": "有人员进入高风险区域,请通过广播或对讲机劝离,并安排巡查确认。",
}


class ReportGenerator:
    def build(self, event: AlertEvent) -> str:
        lines = [
            "【工业安全AI预警报告】",
            f"时间: {event.timestamp}",
            f"事件: {self._type_cn(event.alert_type)}",
            f"置信度: {event.confidence:.2f}",
            f"区域: {event.zone}",
        ]
        if event.track_id is not None:
            lines.append(f"目标ID: #{event.track_id}")
        if event.screenshot:
            lines.append(f"证据截图: {event.screenshot}")
        if event.detail:
            lines.append(f"判定依据: {event.detail}")
        lines.append(f"建议动作: {SUGGESTIONS.get(event.alert_type, '')}")
        return "\n".join(lines)

    @staticmethod
    def _type_cn(t: str) -> str:
        return {
            "NO_HELMET": "未佩戴安全帽",
            "FALL": "人员倒地",
            "INTRUSION": "危险区域闯入",
        }.get(t, t)

    @staticmethod
    def now() -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")