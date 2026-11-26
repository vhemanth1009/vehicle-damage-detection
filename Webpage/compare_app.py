
import os
from pathlib import Path
import base64
import tempfile
import requests
import ipaddress
import socket

import cv2

from urllib.parse import urlparse
from flask import Flask, render_template, request, jsonify
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

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

CONFIDENCE_THRESHOLD = 0.05
IOU_THRESHOLD = 0.30

MAX_REMOTE_IMAGE_SIZE = 10 * 1024 * 1024

# Approved image hostnames.
# Configure using QUICKRIDE_IMAGE_HOSTS environment variable.

ALLOWED_IMAGE_HOSTS = {
    host.strip().lower()
    for host in os.getenv("QUICKRIDE_IMAGE_HOSTS", "").split(",")
    if host.strip()
}

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

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

    height1, width1 = image1_shape[:2]
    height2, width2 = image2_shape[:2]

    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    box1 = [
        x1a / width1,
        y1a / height1,
        x2a / width1,
        y2a / height1
    ]

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

    if crop1.size == 0 or crop2.size == 0:
        return False

    gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=500)

    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

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


def image_to_base64(image):

    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )

    if not success:
        raise RuntimeError("Unable to encode the annotated image as JPEG.")

    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def make_json_safe_result(result):

    excluded_keys = {
        "original_image",
        "annotated_image",
        "image_base64"
    }

    return {
        key: value
        for key, value in result.items()
        if key not in excluded_keys
    }


# --------------------------------------------------
# 3. DOWNLOAD IMAGE FROM APPROVED HTTPS URL
# --------------------------------------------------

def download_image_from_url(image_url, prefix):

    if not isinstance(image_url, str) or not image_url.strip():
        raise ValueError("A valid image URL is required.")

    image_url = image_url.strip()

    try:
        parsed = urlparse(image_url)
        port = parsed.port
    except ValueError:
        raise ValueError("The supplied image URL is invalid.")

    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Only valid HTTPS image URLs are allowed.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")

    if port not in (None, 443):
        raise ValueError("Only HTTPS port 443 is allowed.")

    hostname = parsed.hostname.lower()

    # Require an explicitly approved hostname.

    if not ALLOWED_IMAGE_HOSTS:
        raise ValueError("No approved image hosts are configured.")

    if hostname not in ALLOWED_IMAGE_HOSTS:
        raise ValueError("This image URL hostname is not approved.")

    # Reject private, local and reserved IP addresses.

    try:

        addresses = socket.getaddrinfo(
            hostname,
            443,
            type=socket.SOCK_STREAM
        )

    except socket.gaierror:
        raise ValueError("The image URL hostname could not be resolved.")

    if not addresses:
        raise ValueError("The image URL hostname has no valid address.")

    for address in addresses:

        ip = ipaddress.ip_address(address[4][0])

        if not ip.is_global:
            raise ValueError("The image URL points to a restricted address.")

    image_path = None
    response = None
    total_size = 0

    try:

        response = requests.get(
            image_url,
            timeout=(5, 30),
            stream=True,
            allow_redirects=False
        )

        response.raise_for_status()

        content_length = response.headers.get("Content-Length")

        if content_length:

            try:

                if int(content_length) > MAX_REMOTE_IMAGE_SIZE:
                    raise ValueError("Image exceeds the 10 MB size limit.")

            except ValueError as exc:

                if "10 MB" in str(exc):
                    raise

                raise ValueError("Invalid image Content-Length header.")

        # Create a temporary file instead of using an uploads folder.

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            prefix=f"{prefix}_",
            delete=False
        ) as output_file:

            image_path = Path(output_file.name)

            for chunk in response.iter_content(chunk_size=8192):

                if not chunk:
                    continue

                total_size += len(chunk)

                if total_size > MAX_REMOTE_IMAGE_SIZE:
                    raise ValueError("Image exceeds the 10 MB size limit.")

                output_file.write(chunk)

        # Validate the downloaded image.

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("The supplied URL did not return a valid image.")

        return image_path

    except Exception:

        if image_path is not None and image_path.exists():
            image_path.unlink()

        raise

    finally:

        if response is not None:
            response.close()


# --------------------------------------------------
# 4. YOLO DAMAGE DETECTION
# --------------------------------------------------

def detect_damage(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to read the supplied image.")

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

    # Convert annotated image directly into Base64.

    annotated_image_base64 = image_to_base64(annotated_image)

    return {
        "count": len(detections),
        "detections": detections,
        "image_base64": annotated_image_base64,
        "original_image": image,
        "annotated_image": annotated_image
    }


# --------------------------------------------------
# 5. LOGIN AND LOGOUT DAMAGE COMPARISON
# --------------------------------------------------

def compare_damage(login_result, logout_result):

    login_detections = login_result["detections"]
    logout_detections = logout_result["detections"]

    login_image = login_result["original_image"]
    logout_image = logout_result["original_image"]

    existing_damage = []
    potential_new_damage = []
    needs_review = []

    matched_login = set()
    matched_logout = set()

    possible_matches = []

    # STEP 1: FIND POSSIBLE MATCHES

    for login_index, login_detection in enumerate(login_detections):

        for logout_index, logout_detection in enumerate(logout_detections):

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

            if iou >= IOU_THRESHOLD or visually_similar:

                match_score = iou + (1 if visually_similar else 0)

                possible_matches.append((
                    match_score,
                    login_index,
                    logout_index
                ))

    # STEP 2: ASSIGN ONE-TO-ONE MATCHES

    possible_matches.sort(
        key=lambda item: item[0],
        reverse=True
    )

    for score, login_index, logout_index in possible_matches:

        if login_index in matched_login:
            continue

        if logout_index in matched_logout:
            continue

        matched_login.add(login_index)
        matched_logout.add(logout_index)

        existing_damage.append(
            logout_detections[logout_index]
        )

    # STEP 3: CLASSIFY UNMATCHED LOGOUT DETECTIONS

    for logout_index, logout_detection in enumerate(logout_detections):

        if logout_index in matched_logout:
            continue

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

            needs_review.append(logout_detection)

        else:

            potential_new_damage.append(logout_detection)

    return {
        "existing_damage": existing_damage,
        "potential_new_damage": potential_new_damage,
        "needs_review": needs_review
    }


# --------------------------------------------------
# 6. MAIN WEBPAGE
# --------------------------------------------------

@app.route("/quickride-damage-detection", methods=["GET", "POST"])
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

    login_path = None
    logout_path = None

    try:

        login_suffix = Path(login_file.filename).suffix.lower()
        logout_suffix = Path(logout_file.filename).suffix.lower()

        # Save login image temporarily.

        with tempfile.NamedTemporaryFile(
            suffix=login_suffix,
            prefix="login_",
            delete=False
        ) as temp_login:

            login_path = Path(temp_login.name)
            login_file.save(str(login_path))

        # Save logout image temporarily.

        with tempfile.NamedTemporaryFile(
            suffix=logout_suffix,
            prefix="logout_",
            delete=False
        ) as temp_logout:

            logout_path = Path(temp_logout.name)
            logout_file.save(str(logout_path))

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

        app.logger.exception("Webpage damage detection failed.")

        return render_template(
            "compare.html",
            error=f"Prediction failed: {str(error)}"
        )

    finally:

        # Delete temporary input images.

        for image_path in (login_path, logout_path):

            if image_path is not None and image_path.exists():
                image_path.unlink()


# --------------------------------------------------
# 7. REST API
# --------------------------------------------------

# Health check endpoint

@app.route("/api/v1/health", methods=["GET"])
def health_check():

    return jsonify({
        "status": "success",
        "message": "QuickRide Vehicle Damage Detection API is running."
    }), 200


# Login and logout image comparison endpoint

@app.route(
    "/api/v1/vehicle-damage-comparisons",
    methods=["POST"]
)
def vehicle_damage_comparison():

    # Accept JSON requests only.

    if request.is_json:
        data = request.get_json()
    elif request.form:
        data = request.form.to_dict()
    else:
        return jsonify({
            "status": "error",
            "message": "Please send input using JSON or x-www-form-urlencoded."
        }), 400

    if not isinstance(data, dict):

        return jsonify({
            "status": "error",
            "message": "Invalid JSON request body."
        }), 400

    login_url = data.get("login_image_url")
    logout_url = data.get("logout_image_url")

    if (
        not isinstance(login_url, str)
        or not isinstance(logout_url, str)
        or not login_url.strip()
        or not logout_url.strip()
    ):

        return jsonify({
            "status": "error",
            "message": "Both login_image_url and logout_image_url are required."
        }), 400

    login_path = None
    logout_path = None

    try:

        # Download both source images temporarily.

        login_path = download_image_from_url(
            login_url,
            "api_login"
        )

        logout_path = download_image_from_url(
            logout_url,
            "api_logout"
        )

        # Run YOLO detection.

        login_result = detect_damage(login_path)
        logout_result = detect_damage(logout_path)

        # Compare login and logout damage.

        comparison = compare_damage(
            login_result,
            logout_result
        )

        # Remove internal NumPy arrays before JSON conversion.

        login_json = make_json_safe_result(login_result)
        logout_json = make_json_safe_result(logout_result)

        # Return annotated images as Base64 strings.

        login_json["annotated_image_base64"] = login_result["image_base64"]
        logout_json["annotated_image_base64"] = logout_result["image_base64"]

        return jsonify({
            "status": "success",
            "message": "Vehicle damage comparison completed.",
            "data": {
                "login_result": login_json,
                "logout_result": logout_json,
                "comparison": comparison
            }
        }), 200

    except ValueError as error:

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 400

    except requests.RequestException:

        app.logger.exception(
            "Unable to download one of the supplied images."
        )

        return jsonify({
            "status": "error",
            "message": "Unable to download one of the supplied images. Check that the URLs are reachable."
        }), 400

    except Exception:

        app.logger.exception(
            "Vehicle damage comparison API failed."
        )

        return jsonify({
            "status": "error",
            "message": "Vehicle damage comparison failed. Check the Flask terminal for details."
        }), 500

    finally:

        # Delete downloaded source images after processing.

        for image_path in (login_path, logout_path):

            if image_path is not None and image_path.exists():
                image_path.unlink()


# --------------------------------------------------
# 8. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )
