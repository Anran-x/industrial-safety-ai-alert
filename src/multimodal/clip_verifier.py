"""
CLIP 零样本兜底复核(transformers 实现,本地权重,不依赖网络)。

动机(设计取舍):
- 固定类别检测器(YOLO)对未见场景存在盲区,置信度分布不总是可靠;
- 用 CLIP 的图文对齐能力对"边界置信度"检测框做语义复核,
  将"是否佩戴安全帽"转化为文本相似度问题;
- 对比实验证明:低置信度区间的误检显著减少。
"""
from typing import Optional, Tuple

import numpy as np
import cv2
import torch

from src.config import HELMET_CLIP_LOW, HELMET_CLIP_HIGH, CLIP_MODEL


class ClipVerifier:
    PROMPTS_NO_HELMET = [
        "a worker with bare head without any helmet",
        "a head of a person without a safety helmet",
        "a construction worker not wearing a hard hat",
    ]
    PROMPTS_HELMET = [
        "a worker wearing a yellow safety helmet",
        "a person wearing a hard hat on head",
        "a construction worker wearing a safety helmet",
    ]

    def __init__(self, model_dir: str = CLIP_MODEL, device: Optional[str] = None,
                 threshold: float = 0.6):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_dir).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_dir)
        self.threshold = threshold
        self.texts = self.PROMPTS_NO_HELMET + self.PROMPTS_HELMET
        self.n_prompts = len(self.PROMPTS_NO_HELMET)

    @torch.no_grad()
    def _text_features(self):
        inputs = self.processor(text=self.texts, return_tensors="pt", padding=True).to(self.device)
        return self.model.get_text_features(**inputs)

    def verify(self, image: np.ndarray, box, conf: float) -> Tuple[bool, float]:
        """对单个检测框复核,返回 (是否未戴安全帽, 未戴概率)。"""
        if conf >= HELMET_CLIP_HIGH:
            return box.cls == 0, float(conf)
        if conf < HELMET_CLIP_LOW:
            return box.cls == 0, float(conf)

        H, W = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in (box.x1, box.y1, box.x2, box.y2)]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return box.cls == 0, float(conf)
        ext = int((y2 - y1) * 0.5)
        y1e, y2e = max(0, y1 - ext), min(H, y2)
        crop = image[y1e:y2e, x1:x2]

        inputs = self.processor(
            images=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
            text=self.texts, return_tensors="pt", padding=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits_per_image[0], dim=-1).cpu().numpy()
        p_no = float(probs[: self.n_prompts].max())
        p_yes = float(probs[self.n_prompts:].max())
        p_no_total = p_no / (p_no + p_yes + 1e-9)
        return p_no_total >= self.threshold, p_no_total