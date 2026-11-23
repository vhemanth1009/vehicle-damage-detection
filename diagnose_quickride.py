
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent

MODELS = {
    "quickride_2class": (
        PROJECT / "results" / "quickride_finetune"
        / "quickride_2class_pilot_v1" / "weights" / "best.pt"
    ),
    "quickride_v8": (
        PROJECT / "results" / "stage1_yolo11n_combined_v8"
        / "stage1_yolo11n_combined_v8_20ep_960-3"
        / "weights" / "best.pt"
    ),
}

# Replace this with the full path of ONE Quick Ride image.
IMAGE_PATH = Path(
    r"D:\vehicle-damage-detection\quickride_test_image.jpg"
)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}\n"
        "Update IMAGE_PATH to the location of your test image."
    )

for model_name, model_path in MODELS.items():
    print("\n" + "=" * 60)
    print("MODEL:", model_name)

    if not model_path.exists():
        print("Checkpoint not found:", model_path)
        continue

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(IMAGE_PATH),
        conf=0.001,
        imgsz=1280,
        max_det=100,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("No detections, even at 0.1% confidence.")
    else:
        boxes = sorted(
            result.boxes,
            key=lambda box: float(box.conf[0]),
            reverse=True
        )

        print("Number of candidate detections:", len(boxes))

        for box in boxes[:20]:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0]) * 100

            print(
                f"Class: {model.names[class_id]:15s} "
                f"Confidence: {confidence:6.2f}%"
            )

    output_path = PROJECT / "diagnostics" / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    annotated = result.plot()
    from pathlib import Path as P
    import cv2

    output_image = output_path / "annotated.jpg"
    cv2.imwrite(str(output_image), annotated)

    print("Annotated image saved to:", output_image)

print("\nDiagnostic completed.")
