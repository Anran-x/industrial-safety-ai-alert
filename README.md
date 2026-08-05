# Industrial Safety AI Alert System

面向**工业厂区(固废处理 / 能源 / 制造)安全监控**场景的 AI 智能预警原型系统:
**安全帽佩戴检测 + 人员倒地检测 + 危险区域闯入告警**,并叠加 CLIP 零样本兜底复核。

> 技术选型贴近当前工业视觉落地的主流环境:Ultralytics YOLO11 系列(检测/姿态统一
> s 级)+ ByteTrack 跟踪 + 规则化行为判定 + CLIP 图文语义兜底 + SQLite 告警闭环。
> 全部基于真实公开数据集(SHWD / UR Fall)训练与验证,指标可复现。

---

## 为什么做这个(项目动机)

- 固废处理、能源、制造厂区"安全监管靠人盯"是行业普遍痛点:漏看、疲劳、无法追溯;
- 中小型厂商落地时受限于算力(边缘盒子 / 4GB 级 GPU)与成本,需要**轻量 + 可解释 +
  可部署**的检测方案;
- 本项目在 4GB 显存设备上训练/推理,验证了轻量级方案在真实数据集上的可行性。

## 功能

| 模块 | 方案 | 状态 |
|---|---|---|
| 安全帽佩戴检测 | YOLO11s 微调(SHWD, 2 类) | ✔ 完成(test mAP50 97.7%) |
| 人员倒地检测 | YOLO11s-pose 关键点 + 时序规则(多帧确认 + 复位 + 断帧重联) | ✔ 完成(UR Fall 召回 80%) |
| 危险区域闯入 | ByteTrack 多目标跟踪 + 归一化 ROI 多边形判定 | ✔ 完成(演示核验) |
| 低置信度兜底复核 | CLIP 零样本语义复核(图文对齐) | ✔ 完成(低置信区修正 27.3%) |
| 告警报告 | 结构化模板报告(时间/地点/置信度/截图/建议动作),预留 VLM 接口 | ✔ 完成 |
| 告警存档 | SQLite 存储 + CSV 导出 + 截图留证 | ✔ 完成 |
| 演示端 | Gradio Web(上传视频 → 检测 → 告警面板) | ✔ 完成(~23 FPS@1050Ti) |

## 关键设计决策(面试口径)

1. **模型选型为什么是 YOLO11 s 级**:检测/姿态统一 s 级——参数规模与精度平衡,
   是当前安防与工业视觉集成商的主流选型;n→s 在 4GB 显存上代价小、精度收益明显。
2. **为什么用规则法做倒地/闯入而非时序模型**:小样本下规则法可解释、可调参、
   部署开销低;时序模型需大量带标注跌倒视频且难以解释。规则有明确物理意义
   (宽高比 > 阈值 + 躯干水平判定 + 连续帧确认)。
3. **为什么加 CLIP 兜底**:单一检测器的置信度分布对未见场景不可靠;把"是否戴帽"
   转化为 CLIP 图文相似度问题做低置信度区间复核,是检测 → 语义的多模态闭环。
4. **为什么本地不跑大 VLM**:4GB 显存约束,报告采用确定性模板并预留云端 VLM 接口。
5. **数据诚实**:SHWD 类别名存在歧义,本项目用 CLIP 做了独立语义校验
   (class0=未戴帽、class1=戴帽),不轻信第三方命名。详见 `docs/data_sheet.md`。

## 目录结构

```
industrial_safety_ai_alert/
├── docs/            # 需求文档、数据说明书、评估报告
├── data/            # 数据集(不入库):SHWD / UR Fall
├── scripts/         # 数据整理、训练、评估脚本
├── src/
│   ├── core/        # YOLO 推理封装、数据结构
│   ├── behavior/    # 倒地 / 闯入行为规则
│   ├── multimodal/  # CLIP 兜底复核、告警报告
│   ├── storage.py   # SQLite 告警存档
│   └── app.py       # Gradio 演示端
├── models/          # 预训练与微调权重(不入库)
├── runs/            # 训练日志与曲线(不入库)
└── results/         # 评估结果、截图、演示视频(不入库)
```

## 技术栈

Python 3.10 · PyTorch 2.7+cu128 · Ultralytics YOLO11 · ByteTrack · OpenCV · Transformers(CLIP ViT-B/32)· Gradio · SQLite

## 快速开始

```bash
# 1. 安装依赖(GPU 版 torch 见 requirements.txt)
pip install -r requirements.txt

# 2. 准备数据(详见 docs/data_sheet.md)
python scripts/prepare_helmet_yolo.py     # SHWD -> YOLO 格式

# 3. 训练安全帽检测模型(约 4h @1050Ti)
python scripts/train_helmet.py

# 4. 评估
python scripts/eval_fall.py               # 倒地规则:UR Fall 召回/误报
python scripts/eval_clip.py               # CLIP 兜底:低置信度区间修正率

# 5. 演示
python -m src.app                         # http://127.0.0.1:7860
```

## 数据与许可

- SHWD(安全帽佩戴数据集)、UR Fall(跌倒数据集)均为公开研究数据集;
- 本项目仅用于研究与技术演示,详细口径与已知局限见 `docs/data_sheet.md`。

## Roadmap

- [x] 需求文档 / 数据说明书
- [x] 数据收集与 YOLO 格式整理(SHWD 7k+ 图,UR Fall 70 视频)
- [x] 检测 / 行为 / 多模态 / 存储 / 演示端代码
- [ ] 安全帽模型训练与评估(进行中)
- [ ] 倒地 / 闯入 / CLIP 兜底评估报告
- [ ] 演示视频与面试素材整理
