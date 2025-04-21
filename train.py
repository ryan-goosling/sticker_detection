from ultralytics import YOLO
from pathlib import Path

project_root = Path(__file__).resolve().parent
data_path = project_root / "data.yaml"

model = YOLO("yolov8n.pt")

model.train(
    data=str(data_path),
    epochs=200,
    imgsz=512,
    batch=16,
    name="yolo-stickers",
    patience=20,
    save=True,
    save_period=10,
    val=True,
    verbose=True,
    workers=4
)
