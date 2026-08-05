"""从 alerts.db 取三类告警各一例,生成最终告警报告样例 results/sample_report.md。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ALERT_DB, ALERT_SCREENSHOT_DIR
from src.multimodal.report_gen import AlertEvent, ReportGenerator
from src.storage import AlertStore

OUT = Path(__file__).resolve().parent.parent / "results" / "sample_report.md"


def main():
    store = AlertStore()
    rows = store.query(limit=100000)

    per_type = {}
    for d in rows:
        per_type[d["alert_type"]] = d  # 同类型只留最后一条
    types = ["FALL", "NO_HELMET", "INTRUSION"]

    gen = ReportGenerator()
    parts = [
        "# 工业安全 AI 预警系统 · 告警报告样例",
        "",
        f"> 生成时间:{gen.now()}  |  来源:results/alerts.db(共 {len(rows)} 条告警记录)",
        "",
        "本样例从 SQLite 告警库中提取每类事件最新一例,展示系统产出的完整告警报告",
        "文本与证据截图,截图同时落盘于 results/screenshots/。",
        "",
    ]
    for t in types:
        if t not in per_type:
            parts.append(f"## {t}\n\n(库中暂无此类告警)\n")
            continue
        d = per_type[t]
        ev = AlertEvent(d["alert_type"], d["ts"], d["confidence"], d["zone"] or "默认区域",
                        d["track_id"], d["screenshot"] or "", d["detail"] or "")
        parts += [
            f"## {gen._type_cn(t)}({t})",
            "",
            f"![证据截图](screenshots/{Path(ev.screenshot).name})",
            "",
            "```",
            gen.build(ev),
            "```",
            "",
        ]

    parts.append("---")
    parts.append("## 复现方式")
    parts.append("")
    parts.append("- `python scripts/prepare_demo_assets.py` 生成演示素材")
    parts.append("- 通过 Gradio(`python -m src.app`)或脚本端到端处理视频,告警自动入库")
    parts.append("- 重新生成本样例:`python scripts/gen_sample_report.py`")
    parts.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"已生成 {OUT} ({len(rows)} 条告警 => 每类最新一例)")


if __name__ == "__main__":
    main()