# Industrial Safety AI Alert System

面向工业厂区(固废处理/能源/制造)安全监控场景的 AI 智能预警原型系统。

## 功能

- 安全帽佩戴检测(YOLO11n 目标检测)
- 人员倒地检测(姿态关键点 + 规则判定)
- 危险区域闯入告警(ByteTrack 多目标跟踪 + ROI 区域规则)
- CLIP 零样本兜底复核(降低误报)
- 告警事件多模态报告(截图 + 时间地点 + 文本描述)
- Web 演示端(Gradio)+ SQLite 告警记录

## 目录结构

```
industrial_safety_ai_alert/
├── docs/            # 需求文档、数据说明书、评估报告
├── data/            # 数据集(不入库)
├── scripts/         # 数据下载与整理脚本
├── src/
│   ├── detection/   # 目标检测与跟踪
│   ├── behavior/    # 倒地 / 闯入行为规则
│   ├── multimodal/  # CLIP 兜底、告警报告
│   └── app/         # Gradio 演示端
├── models/          # 训练产物(不入库)
├── runs/            # 训练日志与曲线(不入库)
└── results/         # 评估结果与演示素材(不入库)
```

## 技术栈

Python 3.10 · PyTorch · Ultralytics YOLO11 · ByteTrack · OpenCV · CLIP · Gradio · SQLite

## 快速开始

(开发中,待补充)
