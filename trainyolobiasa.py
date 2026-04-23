from ultralytics import YOLO

# ✅ Load pretrained YOLO11n
model = YOLO("yolo11n.pt")

# ✅ Latih model dengan augmentasi "buah tampak kecil / jauh"
model.train(
    data="dataset.yaml",
    epochs=60,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    optimizer="Adam",
    lr0=0.001,
    pretrained=False,
    project="runs",
    name="yolo11n_baseline_ripeness_farview",
)
