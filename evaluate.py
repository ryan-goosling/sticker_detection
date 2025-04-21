#!/usr/bin/env python3

from ultralytics import YOLO
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data.yaml"
    model_path = project_root / "weights/best.pt"

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    model = YOLO(str(model_path))
    results = model.val(data=str(data_path), split='test')

    print("\n=== Evaluation Results on Test Set ===")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")
    print(f"mAP50:     {results.box.map50:.4f}")
    print(f"mAP50-95:  {results.box.map:.4f}")

if __name__ == "__main__":
    main()
