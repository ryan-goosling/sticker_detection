#!/usr/bin/env python3
"""
grid_search.py — grid-search for inference thresholds with embedded ranges.
───────────────────────────────────────────────────────────────────────────────
Usage:
    python grid_search.py data.yaml [--model MODEL]

This script defines internal ranges for thresholds and runs grid-search over all combinations,
prints metrics per combination and selects best by F1 score.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO
import itertools
import yaml

import inference  # inference.py must define run_hybrid_inference
run_hybrid = inference.run_hybrid_inference
CLASS_NAMES = inference.CLASS_NAMES
DEFAULT_MODEL = inference.DEFAULT_MODEL_PATH

# ───────────────────────────── helpers ───────────────────────────────────────

def load_gt(gt_path: Path) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    gt: List[Tuple[int, Tuple[float, float, float, float]]] = []
    if not gt_path.exists():
        return gt
    for line in gt_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = map(float, parts)
        gt.append((int(cls), (cx, cy, bw, bh)))
    return gt

# ───────────────────────────── metrics ───────────────────────────────────────
class Metrics:
    def __init__(self, iou_thr: float):
        self.iou_thr = iou_thr
        self.records: Dict[int, List[Tuple[float, int]]] = {}
        self.gt_count: Dict[int, int] = {}

    def add_image(
        self,
        preds: List[Dict[str, Any]],
        gts: List[Tuple[int, Tuple[float, float, float, float]]],
        w: int,
        h: int
    ):
        # collect ground truth boxes per class
        by_cls: Dict[int, List[List[int]]] = {}
        for cls, box_n in gts:
            self.gt_count[cls] = self.gt_count.get(cls, 0) + 1
            cx, cy, bw, bh = box_n
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            by_cls.setdefault(cls, []).append([x1, y1, x2, y2])
        matched: Dict[int, List[bool]] = {c: [False]*len(lst) for c, lst in by_cls.items()}

        # evaluate predictions sorted by confidence
        for conf, cls, bbox in sorted(
            [(d['confidence'], d['class_id'], d['bbox']) for d in preds],
            key=lambda x: x[0], reverse=True
        ):
            # convert normalized to pixel
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            tp = 0
            if cls in by_cls:
                best_iou = 0.0
                best_idx = -1
                for idx, gt_box in enumerate(by_cls[cls]):
                    if matched[cls][idx]:
                        continue
                    # compute IoU
                    xa1 = max(x1, gt_box[0]); ya1 = max(y1, gt_box[1])
                    xa2 = min(x2, gt_box[2]); ya2 = min(y2, gt_box[3])
                    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    union = area1 + area2 - inter
                    iou_val = inter / union if union else 0.0
                    if iou_val > best_iou:
                        best_iou, best_idx = iou_val, idx
                if best_iou >= self.iou_thr:
                    tp = 1
                    matched[cls][best_idx] = True
            self.records.setdefault(cls, []).append((conf, tp))

    def compute(self) -> Tuple[float, float, float]:
        aps, ps, rs = [], [], []
        for cls, rec in self.records.items():
            total = self.gt_count.get(cls, 0)
            if total == 0:
                continue
            rec_sorted = sorted(rec, key=lambda x: x[0], reverse=True)
            tps = np.array([tp for _, tp in rec_sorted])
            fps = 1 - tps
            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(fps)
            recs = cum_tp / (total + 1e-9)
            precs = cum_tp / (cum_tp + cum_fp + 1e-9)
            # AP calculation
            mrec = np.concatenate(([0], recs, [1]))
            mpre = np.concatenate(([0], precs, [0]))
            for i in range(len(mpre)-1, 0, -1):
                mpre[i-1] = max(mpre[i-1], mpre[i])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])
            aps.append(ap)
            ps.append(precs[-1] if precs.size else 0)
            rs.append(recs[-1] if recs.size else 0)
        mAP = float(np.mean(aps)) if aps else 0.0
        precision = float(np.mean(ps)) if ps else 0.0
        recall = float(np.mean(rs)) if rs else 0.0
        return mAP, precision, recall


# ────────────────────── per-image P/R and F1 helpers ─────────────────────────

def pr_image(
    preds: List[Dict[str, Any]],
    gts: List[Tuple[int, Tuple[float, float, float, float]]],
    w: int, h: int, iou_thr: float
) -> Tuple[float, float]:
    """
    Compute precision and recall for one image:
    TP = matched preds, FP = other preds, FN = unmatched GT.
    """
    # convert GT to absolute boxes
    gt_boxes: List[Tuple[int,int,int,int]] = []
    for cls, (cx, cy, bw, bh) in gts:
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        gt_boxes.append((cls, (x1, y1, x2, y2)))
    matched_gt = [False] * len(gt_boxes)
    tp = 0
    # evaluate predictions
    for det in preds:
        cls = det['class_id']
        x1 = int(det['bbox'][0] * w); y1 = int(det['bbox'][1] * h)
        x2 = int(det['bbox'][2] * w); y2 = int(det['bbox'][3] * h)
        best_iou = 0.0; best_idx = -1
        for idx, (gcls, (gx1, gy1, gx2, gy2)) in enumerate(gt_boxes):
            if matched_gt[idx] or gcls != cls: continue
            inter_w = max(0, min(x2, gx2) - max(x1, gx1))
            inter_h = max(0, min(y2, gy2) - max(y1, gy1))
            inter = inter_w * inter_h
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (gx2 - gx1) * (gy2 - gy1)
            union = area1 + area2 - inter
            iou_val = inter / union if union else 0.0
            if iou_val > best_iou:
                best_iou, best_idx = iou_val, idx
        if best_iou >= iou_thr:
            tp += 1
            matched_gt[best_idx] = True
    fp = len(preds) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    return precision, recall

class MetricF1:
    """Accumulate per-image P/R and compute mean F1 score."""
    def __init__(self):
        self.ps: List[float] = []
        self.rs: List[float] = []
    def add(self, p: float, r: float) -> None:
        self.ps.append(p); self.rs.append(r)
    def compute(self) -> float:
        f1s = [2*p*r/(p+r+1e-9) for p, r in zip(self.ps, self.rs)]
        return float(np.mean(f1s)) if f1s else 0.0

# ──────────────────────────── main grid-search ───────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-yaml', default='data.yaml', help='Path to data.yaml file (default: data.yaml)')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for mAP@0.5')
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.data_yaml).read_text())
    img_dir = Path(cfg['test'])
    label_dir = img_dir.parents[1] / 'labels' / img_dir.name

    # define threshold ranges in code
    conf_list    = [0.3, 0.4, 0.5, 0.6]
    iou_nms_list = [0.2, 0.3, 0.4]
    iou_wbf_list = [0.5, 0.6, 0.7]
    ioa_list     = [0.7, 0.8, 0.9]

    results = []
    # iterate combinations
    for conf_thr, iou_nms, iou_wbf, ioa_nested in itertools.product(
        conf_list, iou_nms_list, iou_wbf_list, ioa_list
    ):
        thresh = {
            'conf': conf_thr,
            'nms': iou_nms,
            'wbf': iou_wbf,
            'nested': ioa_nested
        }
        # run inference and metrics
        # collect all preds & GT first
        metric50 = Metrics(iou_thr=args.iou)
        metricF1 = MetricF1()  # helper to calculate F1
        records = []
        for img_path in sorted(img_dir.glob('*.*')):
            gt = load_gt(label_dir / f"{img_path.stem}.txt")
            dets = run_hybrid(
                img_path, YOLO(args.model),
                thresh['conf'], thresh['nms'], thresh['wbf'], thresh['nested']
            )
            img = cv2.imread(str(img_path)); h, w = img.shape[:2]
            metric50.add_image(dets, gt, w, h)
            # compute per-image precision/recall for F1
            p, r = pr_image(dets, gt, w, h, args.iou)
            metricF1.add(p, r)
        m50, p50, r50 = metric50.compute()
        f1 = metricF1.compute()
        results.append({
            'conf': conf_thr, 'nms': iou_nms,
            'wbf': iou_wbf, 'nested': ioa_nested,
            'mAP50': round(m50,3), 'prec': round(p50,3), 'rec': round(r50,3),
            'f1': round(f1,3)
        })
        print(f"conf={conf_thr}, nms={iou_nms}, wbf={iou_wbf}, nested={ioa_nested}  "
              f"-> mAP50={m50:.3f}, P={p50:.3f}, R={r50:.3f}, F1={f1:.3f}")

    # find best by F1
    best = max(results, key=lambda x: x['f1'])
    print("\nBest parameters by F1 score:")
    print(json.dumps(best, indent=2))

if __name__ == '__main__':
    main()
