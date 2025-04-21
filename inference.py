#!/usr/bin/env python3
"""
Hybrid YOLO inference (single image or directory) with configurable thresholds.

Usage:
  .venv\Scripts\python.exe inference.py path/to/image_or_dir
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import cv2
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# ──────────────────────────────── CONSTANTS ────────────────────────────────
DEFAULT_MODEL_PATH = "weights/best.pt"
CLASS_NAMES = ["happy", "sad", "dead"]
WINDOW_SIZE = 512
STRIDE = 384

# ──────────────────────────── geometry helpers ─────────────────────────────
def _area(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _iou(a: List[float], b: List[float]) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0.0

def _ioa(inner: List[float], outer: List[float]) -> float:
    xa1, ya1, xa2, ya2 = inner
    xb1, yb1, xb2, yb2 = outer
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    area_inner = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    return inter / area_inner if area_inner else 0.0

def _to_norm(boxes: List[List[float]], w: int, h: int) -> List[List[float]]:
    return [[x1 / w, y1 / h, x2 / w, y2 / h] for x1, y1, x2, y2 in boxes]

# ───────────────────────────── post‑processing ─────────────────────────────
def _postprocess(
    predictions: List[Dict[str, Any]], w: int, h: int,
    iou_nms: float, iou_wbf: float, ioa_nested: float
) -> List[Dict[str, Any]]:
    if not predictions:
        return []

    boxes = torch.tensor([p["bbox"] for p in predictions], dtype=torch.float32)
    scores = torch.tensor([p["confidence"] for p in predictions])
    labels = torch.tensor([p["class_id"] for p in predictions])
    keep = batched_nms(boxes, scores, labels, iou_nms)
    preds = [predictions[i] for i in keep]

    if len(preds) > 1:
        b_norm = [_to_norm([p["bbox"]], w, h)[0] for p in preds]
        s_lst = [p["confidence"] for p in preds]
        l_lst = [p["class_id"] for p in preds]
        b_f, s_f, l_f = weighted_boxes_fusion([b_norm], [s_lst], [l_lst],
                                               iou_thr=iou_wbf, skip_box_thr=0.0)
        preds = [
            {"bbox": [x1 * w, y1 * h, x2 * w, y2 * h],
             "confidence": float(sc),
             "class_id": int(lb)}
            for (x1, y1, x2, y2), sc, lb in zip(b_f, s_f, l_f)
        ]

    final: List[Dict[str, Any]] = []
    for p in sorted(preds, key=lambda x: x["confidence"], reverse=True):
        keep_box = True
        for q in list(final):
            area_p = _area(p["bbox"])
            area_q = _area(q["bbox"])
            inner_bbox, outer_bbox = (p["bbox"], q["bbox"]) if area_p <= area_q else (q["bbox"], p["bbox"])
            if _ioa(inner_bbox, outer_bbox) >= ioa_nested:
                if inner_bbox == p["bbox"]:
                    keep_box = False
                    break
                else:
                    final.remove(q)
        if keep_box:
            final.append(p)
    return final

# ───────────────────────── hybrid inference core ───────────────────────────
def run_hybrid_inference(
    img_path: Path, model: YOLO,
    conf_thr: float, iou_nms: float, iou_wbf: float, ioa_nested: float
) -> List[Dict[str, Any]]:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    raw_preds: List[Dict[str, Any]] = []

    for box in model.predict(img_rgb, verbose=False)[0].boxes:
        conf = float(box.conf)
        if conf < conf_thr:
            continue
        raw_preds.append({"bbox": box.xyxy[0].cpu().numpy().tolist(),
                          "confidence": conf,
                          "class_id": int(box.cls)})

    for y in range(0, H - WINDOW_SIZE + 1, STRIDE):
        for x in range(0, W - WINDOW_SIZE + 1, STRIDE):
            crop = img_rgb[y : y + WINDOW_SIZE, x : x + WINDOW_SIZE]
            for box in model.predict(crop, verbose=False)[0].boxes:
                conf = float(box.conf)
                if conf < conf_thr:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                raw_preds.append({"bbox": [x1 + x, y1 + y, x2 + x, y2 + y],
                                  "confidence": conf,
                                  "class_id": int(box.cls)})
    processed = _postprocess(raw_preds, W, H, iou_nms, iou_wbf, ioa_nested)
    for det in processed:
        det["bbox"] = [round(v, 6) for v in _to_norm([det["bbox"]], W, H)[0]]
        det["label"] = CLASS_NAMES[det["class_id"]]
    return processed

# ────────────────────────────── CLI entrypoint ──────────────────────────────
def list_images(path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.is_file() and path.suffix.lower() in exts:
        return [path]
    if path.is_dir():
        files: List[Path] = []
        for ext in exts:
            files.extend(path.glob(f"*{ext}"))
        return sorted(files)
    raise ValueError("Source must be an image or directory of images")


def main():
    parser = argparse.ArgumentParser(description="Sticker detector inference CLI")
    parser.add_argument("source", help="Path to image or directory")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="YOLO model weights (.pt)")
    parser.add_argument("--conf-thr", type=float, default=0.50,
                        help="Confidence threshold for detection")
    parser.add_argument("--iou-nms", type=float, default=0.30,
                        help="IoU threshold for NMS")
    parser.add_argument("--iou-wbf", type=float, default=0.60,
                        help="IoU threshold for weighted box fusion")
    parser.add_argument("--ioa-nested", type=float, default=0.75,
                        help="IoA threshold for nested pruning")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="Device for inference: 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    source_path = Path(args.source)
    model = YOLO(args.model)
    model.to(device)

    images = list_images(source_path)
    if not images:
        return

    for img in images:
        detections = run_hybrid_inference(
            img, model,
            args.conf_thr, args.iou_nms, args.iou_wbf, args.ioa_nested
        )
        print(json.dumps({"image": str(img), "detections": detections}, ensure_ascii=False))


if __name__ == "__main__":
    main()
