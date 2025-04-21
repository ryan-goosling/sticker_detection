#!/usr/bin/env python3
"""
test.py — Run hybrid inference on a directory of images, save annotated results, and print detections.

Usage:
    .\.venv\Scripts\python.exe test.py <test-dir> <result-dir>
"""

from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import torch
import time
from ultralytics import YOLO
import inference  # inference.py must be in the same folder

# Default thresholds matching inference.py
default_conf = 0.50
default_iou_nms = 0.3
default_iou_wbf = 0.6
default_ioa_nested = 0.75

# Class name mappings
CLASS_COLORS = {
    0: (255, 0, 0),   # happy - blue
    1: (0, 255, 0),   # sad   - green
    2: (0, 0, 255),   # dead  - red
}

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def list_images(dir_path: Path) -> list[Path]:
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in EXTS and p.is_file()])


def main():
    parser = argparse.ArgumentParser(description="Run test inference and save results")
    parser.add_argument('test_dir', help='Directory with test images')
    parser.add_argument('result_dir', help='Directory to save annotated images')
    parser.add_argument('--model', default=inference.DEFAULT_MODEL_PATH,
                        help='Path to YOLO .pt model')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                        help="Device for inference: 'cuda' or 'cpu'")
    args = parser.parse_args()

    # Device selection
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    test_path = Path(args.test_dir).resolve()
    out_path = Path(args.result_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(args.model)
    model.to(device)

    images = list_images(test_path)
    if not images:
        return

    for img_path in images:
        detections = inference.run_hybrid_inference(
            img_path, model,
            default_conf, default_iou_nms,
            default_iou_wbf, default_ioa_nested
        )

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        for det in detections:
            x1_n, y1_n, x2_n, y2_n = det['bbox']
            x1 = int(x1_n * w); y1 = int(y1_n * h)
            x2 = int(x2_n * w); y2 = int(y2_n * h)
            label = det['label']
            print(f"{img_path.name} ({x1},{y1})-({x2},{y2}) {label}")
            color = CLASS_COLORS.get(det['class_id'], (255, 255, 255))
            text = f"{label} {det['confidence']:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        save_path = out_path / img_path.name
        cv2.imwrite(str(save_path), img)


if __name__ == '__main__':
    main()
