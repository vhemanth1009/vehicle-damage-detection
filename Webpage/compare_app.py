
from pathlib import Path
import uuid

import cv2
import numpy as np

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_FOLDER.parent
    / "results"
    / "stage1_yolo11n_combined_v8"
    / "stage1_yolo11n_combined_v8_20ep_960-3"
    / "weights"
    / "best.pt"
)

UPLOAD_FOLDER = PROJECT_FOLDER / "uploads"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

CONFIDENCE_THRESHOLD = 0.05
IOU_THRESHOLD = 0.30

UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

model = YOLO(str(MODEL_PATH))


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def calculate_normalized_iou(box1, image1_shape, box2, image2_shape):
    """
    Calculate IoU after converting both bounding boxes
    into relative image coordinates.

    This helps when login and logout images have
    different resolutions.
    """

    height1, width1 = image1_shape[:2]
    height2, width2 = image2_shape[:2]

    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    # Normalize login bounding box
    box1 = [
        x1a / width1,
        y1a / height1,
        x2a / width1,
        y2a / height1
    ]

    # Normalize logout bounding box
    box2 = [
        x1b / width2,
        y1b / height2,
        x2b / width2,
        y2b / height2
    ]

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


def crop_damage(image, box):

    height, width = image.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    return image[y1:y2, x1:x2]


def visual_similarity(crop1, crop2):
    """
    Compare two damage regions using ORB features.
    """

    if crop1.size == 0 or crop2.size == 0:
        return False

    gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=500)

    keypoints1, descriptors1 = orb.detectAndCompute(
        gray1, None
    )

    keypoints2, descriptors2 = orb.detectAndCompute(
        gray2, None
    )

    if descriptors1 is None or descriptors2 is None:
        return False

    if len(keypoints1) < 4 or len(keypoints2) < 4:
        return False

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) == 2:

            first, second = pair

            if first.distance < 0.75 * second.distance:
                good_matches.append(first)

    return len(good_matches) >= 8


# --------------------------------------------------
# 3. YOLO DAMAGE DETECTION
# --------------------------------------------------

def detect_damage(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to read the uploaded image.")

    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=960,
        verbose=False
    )

    detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "damage": model.names[class_id],
                "class_id": class_id,
                "confidence": round(confidence * 100, 2),
                "box": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ]
            })

    annotated_image = results[0].plot()

    output_filename = f"{uuid.uuid4().hex}_result.jpg"
    output_path = UPLOAD_FOLDER / output_filename

    cv2.imwrite(str(output_path), annotated_image)

    return {
        "count": len(detections),
        "detections": detections,
        "image_url": f"/uploads/{output_filename}",
        "original_image": image
    }


# --------------------------------------------------
# 4. LOGIN AND LOGOUT DAMAGE COMPARISON
# --------------------------------------------------

def compare_damage(login_result, logout_result):

    login_detections = login_result["detections"]
    logout_detections = logout_result["detections"]

    login_image = login_result["original_image"]
    logout_image = logout_result["original_image"]

    existing_damage = []
    potential_new_damage = []
    needs_review = []

    # Track which detections have already been matched.
    # This prevents one login detection from matching
    # multiple logout detections.

    matched_login = set()
    matched_logout = set()

    possible_matches = []

    # ----------------------------------------------
    # STEP 1: FIND POSSIBLE ONE-TO-ONE MATCHES
    # ----------------------------------------------

    for login_index, login_detection in enumerate(login_detections):

        for logout_index, logout_detection in enumerate(logout_detections):

            # Only compare detections of the same damage class.
            if login_detection["class_id"] != logout_detection["class_id"]:
                continue

            iou = calculate_normalized_iou(
                login_detection["box"],
                login_image.shape,
                logout_detection["box"],
                logout_image.shape
            )

            login_crop = crop_damage(
                login_image,
                login_detection["box"]
            )

            logout_crop = crop_damage(
                logout_image,
                logout_detection["box"]
            )

            visually_similar = visual_similarity(
                login_crop,
                logout_crop
            )

            # A pair is considered a strong match when
            # either its overlap or visual similarity is sufficient.
            if iou >= IOU_THRESHOLD or visually_similar:

                # Prefer pairs with visual similarity and
                # higher bounding-box overlap.
                match_score = iou + (1 if visually_similar else 0)

                possible_matches.append((
                    match_score,
                    login_index,
                    logout_index
                ))

    # ----------------------------------------------
    # STEP 2: ASSIGN MATCHES WITHOUT REUSING DETECTIONS
    # ----------------------------------------------

    possible_matches.sort(
        key=lambda item: item[0],
        reverse=True
    )

    for score, login_index, logout_index in possible_matches:

        # Skip if either detection has already been matched.
        if login_index in matched_login:
            continue

        if logout_index in matched_logout:
            continue

        matched_login.add(login_index)
        matched_logout.add(logout_index)

        existing_damage.append(
            logout_detections[logout_index]
        )

    # ----------------------------------------------
    # STEP 3: CLASSIFY UNMATCHED LOGOUT DETECTIONS
    # ----------------------------------------------

    for logout_index, logout_detection in enumerate(logout_detections):

        if logout_index in matched_logout:
            continue

        # Find login detections of the same class that
        # have not already been matched.
        remaining_same_class_login = [
            index
            for index, login_detection in enumerate(login_detections)
            if (
                login_detection["class_id"]
                == logout_detection["class_id"]
                and index not in matched_login
            )
        ]

        if remaining_same_class_login:

            # A similar damage type existed at login,
            # but we could not confidently match it.
            needs_review.append(logout_detection)

        else:

            # No available login detection of this class
            # remains. Treat this as potential new damage.
            potential_new_damage.append(logout_detection)

    return {
        "existing_damage": existing_damage,
        "potential_new_damage": potential_new_damage,
        "needs_review": needs_review
    }


# --------------------------------------------------
# 5. MAIN PAGE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def compare():

    if request.method == "GET":
        return render_template("compare.html")

    login_file = request.files.get("login_image")
    logout_file = request.files.get("logout_image")

    if (
        not login_file
        or not logout_file
        or not login_file.filename
        or not logout_file.filename
    ):
        return render_template(
            "compare.html",
            error="Please upload both login and logout images."
        )

    if (
        not allowed_file(login_file.filename)
        or not allowed_file(logout_file.filename)
    ):
        return render_template(
            "compare.html",
            error="Only JPG, JPEG, PNG and WEBP images are allowed."
        )

    login_name = (
        f"{uuid.uuid4().hex}_login_"
        + secure_filename(login_file.filename)
    )

    logout_name = (
        f"{uuid.uuid4().hex}_logout_"
        + secure_filename(logout_file.filename)
    )

    login_path = UPLOAD_FOLDER / login_name
    logout_path = UPLOAD_FOLDER / logout_name

    login_file.save(str(login_path))
    logout_file.save(str(logout_path))

    try:

        login_result = detect_damage(login_path)
        logout_result = detect_damage(logout_path)

        comparison = compare_damage(
            login_result,
            logout_result
        )

        return render_template(
            "compare.html",
            login=login_result,
            logout=logout_result,
            comparison=comparison
        )

    except Exception as error:

        return render_template(
            "compare.html",
            error=f"Prediction failed: {str(error)}"
        )


# --------------------------------------------------
# 6. SERVE ANNOTATED IMAGES
# --------------------------------------------------

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# --------------------------------------------------
# 7. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )
