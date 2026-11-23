
from pathlib import Path
import uuid

import cv2
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

PROJECT_FOLDER = Path(__file__).resolve().parent
UPLOAD_FOLDER = PROJECT_FOLDER / "uploads"

MODEL_PATH = (
    PROJECT_FOLDER
    / "results"
    / "stage1_yolo11n_combined_v8"
    / "stage1_yolo11n_combined_v8_20ep_960-3"
    / "weights"
    / "best.pt"
)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
CONFIDENCE_THRESHOLD = 0.25

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

print("Loading vehicle damage detection model...")
model = YOLO(str(MODEL_PATH))
print("Model loaded successfully.")
print("Damage classes:", model.names)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please select an image."}), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({"error": "No image selected."}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({
            "error": "Upload a JPG, JPEG, PNG, or WEBP image."
        }), 400

    unique_id = uuid.uuid4().hex
    safe_name = secure_filename(uploaded_file.filename)

    input_path = UPLOAD_FOLDER / f"{unique_id}_{safe_name}"
    output_path = UPLOAD_FOLDER / f"{unique_id}_result.jpg"

    try:
        uploaded_file.save(input_path)

        image = cv2.imread(str(input_path))

        if image is None:
            input_path.unlink(missing_ok=True)
            return jsonify({
                "error": "The image could not be read. Try another image."
            }), 400

        results = model.predict(
            source=image,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]
        annotated_image = result.plot()

        saved = cv2.imwrite(str(output_path), annotated_image)

        if not saved:
            raise RuntimeError("Could not save the detection image.")

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = [
                    round(value)
                    for value in box.xyxy[0].tolist()
                ]

                detections.append({
                    "damage": model.names[class_id],
                    "confidence": round(confidence * 100, 2),
                    "box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })

        return jsonify({
            "success": True,
            "message": (
                "Potential damage detected."
                if detections
                else "No damage was detected at this confidence threshold."
            ),
            "count": len(detections),
            "detections": detections,
            "image_url": f"/uploads/{output_path.name}"
        })

    except Exception as error:
        print("Prediction error:", error)
        return jsonify({
            "error": "Prediction failed. Check the terminal for details."
        }), 500


@app.route("/uploads/<filename>")
def uploaded_image(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image is too large. Maximum upload size is 10 MB."
    }), 413


if __name__ == "__main__":
    app.run(debug=True)
