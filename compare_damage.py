
from pathlib import Path
import base64

import cv2
import numpy as np
from flask import Flask, render_template, request
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
IOU_THRESHOLD = 0.30

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

print("Loading vehicle damage detection model...")
model = YOLO(str(MODEL_PATH))
print("Model loaded successfully.")
print("Damage classes:", model.names)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def read_uploaded_image(uploaded_file):
    image_bytes = uploaded_file.read()

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Could not read the image. Please upload a valid image.")

    return image


def detect_damage(image):
    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = round(float(box.conf[0].item()) * 100, 2)

            x1, y1, x2, y2 = [
                round(value) for value in box.xyxy[0].tolist()
            ]

            detections.append({
                "damage": str(model.names[class_id]),
                "confidence": confidence,
                "damage_score": confidence,
                "box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },
                "class_id": class_id
            })

    return detections


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = (
        box_a["x1"], box_a["y1"], box_a["x2"], box_a["y2"]
    )
    bx1, by1, bx2, by2 = (
        box_b["x1"], box_b["y1"], box_b["x2"], box_b["y2"]
    )

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def compare_detections(login_detections, logout_detections):
    candidates = []

    for li, login_detection in enumerate(login_detections):
        for oi, logout_detection in enumerate(logout_detections):
            if login_detection["class_id"] != logout_detection["class_id"]:
                continue

            iou = calculate_iou(
                login_detection["box"],
                logout_detection["box"]
            )

            if iou >= IOU_THRESHOLD:
                candidates.append((iou, li, oi))

    candidates.sort(reverse=True)

    matched_login = set()
    matched_logout = set()

    for _, li, oi in candidates:
        if li not in matched_login and oi not in matched_logout:
            matched_login.add(li)
            matched_logout.add(oi)

    existing = []
    potential_new = []
    logout_statuses = []

    for index, detection in enumerate(logout_detections):
        item = {
            "damage": detection["damage"],
            "confidence": detection["confidence"],
            "damage_score": detection["damage_score"],
            "box": detection["box"]
        }

        if index in matched_logout:
            item["status"] = "Existing damage"
            existing.append(item)
            logout_statuses.append("Existing damage")
        else:
            item["status"] = "Potential new damage"
            potential_new.append(item)
            logout_statuses.append("Potential new damage")

    return existing, potential_new, logout_statuses


def draw_detections(image, detections, image_type, statuses=None):
    annotated = image.copy()

    for index, detection in enumerate(detections):
        box = detection["box"]

        if image_type == "login":
            color = (0, 165, 255)
            status = "Login"
        elif statuses[index] == "Existing damage":
            color = (0, 200, 0)
            status = "Existing"
        else:
            color = (0, 0, 255)
            status = "Potential new"

        label = (
            f"{status}: {detection['damage']} "
            f"{detection['damage_score']:.2f}%"
        )

        cv2.rectangle(
            annotated,
            (box["x1"], box["y1"]),
            (box["x2"], box["y2"]),
            color,
            2
        )

        cv2.putText(
            annotated,
            label,
            (box["x1"], max(25, box["y1"] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    return annotated


def image_to_data_url(image):
    success, encoded = cv2.imencode(".jpg", image)

    if not success:
        raise RuntimeError("Could not encode the annotated image.")

    image_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_base64}"


def clean_detection(detection, status=None):
    item = {
        "damage": detection["damage"],
        "confidence": detection["confidence"],
        "damage_score": detection["damage_score"],
        "box": detection["box"]
    }

    if status is not None:
        item["status"] = status

    return item


def compare_images(login_image, logout_image):
    login_detections = detect_damage(login_image)
    logout_detections = detect_damage(logout_image)

    existing, potential_new, logout_statuses = compare_detections(
        login_detections,
        logout_detections
    )

    login_annotated = draw_detections(
        login_image,
        login_detections,
        image_type="login"
    )

    logout_annotated = draw_detections(
        logout_image,
        logout_detections,
        image_type="logout",
        statuses=logout_statuses
    )

    login_score = max(
        (item["damage_score"] for item in login_detections),
        default=0.0
    )

    logout_score = max(
        (item["damage_score"] for item in logout_detections),
        default=0.0
    )

    if potential_new:
        message = (
            "Potential new damage detected in the logout image. "
            "Review the highlighted areas."
        )
    elif existing:
        message = (
            "No unmatched damage detections were found. "
            "Detected logout damage was matched with login detections."
        )
    else:
        message = "No damage was detected in either image."

    return {
        "login_image_url": image_to_data_url(login_annotated),
        "logout_image_url": image_to_data_url(logout_annotated),

        "login_count": len(login_detections),
        "logout_count": len(logout_detections),
        "existing_count": len(existing),
        "potential_new_count": len(potential_new),

        "login_damage_score": login_score,
        "logout_damage_score": logout_score,

        "login_detections": [
            clean_detection(item) for item in login_detections
        ],
        "logout_detections": [
            clean_detection(item, status)
            for item, status in zip(logout_detections, logout_statuses)
        ],

        "existing_damage": existing,
        "potential_new_damage": potential_new,
        "message": message
    }


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("compare.html")

    login_file = request.files.get("login_image")
    logout_file = request.files.get("logout_image")

    if login_file is None or logout_file is None:
        return render_template(
            "compare.html",
            error="Please upload both login and logout images."
        )

    if not login_file.filename or not logout_file.filename:
        return render_template(
            "compare.html",
            error="Please select both images."
        )

    if (
        not allowed_file(login_file.filename)
        or not allowed_file(logout_file.filename)
    ):
        return render_template(
            "compare.html",
            error="Upload JPG, JPEG, PNG, or WEBP images only."
        )

    try:
        login_image = read_uploaded_image(login_file)
        logout_image = read_uploaded_image(logout_file)

        result = compare_images(login_image, logout_image)

        return render_template("compare.html", result=result)

    except ValueError as error:
        return render_template("compare.html", error=str(error))

    except Exception as error:
        print("Comparison error:", error)
        return render_template(
            "compare.html",
            error="Comparison failed. Check the terminal for details."
        ), 500


@app.errorhandler(413)
def file_too_large(error):
    return (
        "The total upload is too large. Maximum request size is 20 MB.",
        413
    )


if __name__ == "__main__":
    app.run(port=5001, debug=True)
