
document.addEventListener("DOMContentLoaded", () => {
    const imageInput = document.getElementById("imageInput");
    const dropZone = document.getElementById("dropZone");
    const emptyPreview = document.getElementById("emptyPreview");
    const previewSection = document.getElementById("previewSection");
    const previewImage = document.getElementById("previewImage");
    const fileName = document.getElementById("fileName");
    const detectButton = document.getElementById("detectButton");
    const status = document.getElementById("status");
    const resultSection = document.getElementById("resultSection");

    let selectedFile = null;
    let imageData = null;

    // Show or hide elements using the existing CSS class.
    function showPreview() {
        emptyPreview.classList.add("hidden");
        previewSection.classList.remove("hidden");
    }

    function showPlaceholder() {
        previewSection.classList.add("hidden");
        emptyPreview.classList.remove("hidden");
    }

    // Read the selected image and display it.
    function selectImage(file) {
        if (!file) return;

        if (!file.type.startsWith("image/")) {
            status.textContent = "Please select a valid image file.";
            return;
        }

        selectedFile = file;
        imageData = null;
        detectButton.disabled = true;

        // Keep the placeholder visible until the image loads successfully.
        showPlaceholder();
        status.textContent = "Loading image...";

        const reader = new FileReader();

        reader.onload = function (event) {
            const dataUrl = event.target.result;

            const testImage = new Image();

            testImage.onload = function () {
                imageData = dataUrl;

                previewImage.src = dataUrl;
                previewImage.alt = file.name;

                fileName.textContent = file.name;

                showPreview();

                detectButton.disabled = false;
                status.textContent = "Image ready for damage detection.";

                // Hide previous results when a new image is selected.
                resultSection.classList.add("hidden");
            };

            testImage.onerror = function () {
                selectedFile = null;
                imageData = null;

                showPlaceholder();
                status.textContent =
                    "The image could not be displayed. Please select another.";
            };

            testImage.src = dataUrl;
        };

        reader.onerror = function () {
            selectedFile = null;
            imageData = null;

            showPlaceholder();
            status.textContent = "Unable to read this image file.";
        };

        reader.readAsDataURL(file);
    }

    // Choose an image from the file browser.
    imageInput.addEventListener("change", () => {
        selectImage(imageInput.files[0]);
    });

    // Drag-and-drop support.
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, event => {
            event.preventDefault();
            dropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, event => {
            event.preventDefault();
            dropZone.classList.remove("drag-over");
        });
    });

    dropZone.addEventListener("drop", event => {
        const file = event.dataTransfer.files[0];
        selectImage(file);
    });

    // Send the selected image to the Flask backend.
    detectButton.addEventListener("click", async () => {
        if (!selectedFile || !imageData) {
            status.textContent = "Please select an image first.";
            return;
        }

        const formData = new FormData();
        formData.append("image", selectedFile);

        detectButton.disabled = true;
        detectButton.textContent = "Detecting...";
        status.textContent = "Analyzing image...";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok || data.success === false) {
                throw new Error(
                    data.error || data.message || "Prediction failed."
                );
            }

            displayResults(data);

            status.textContent = "Detection completed.";
        } catch (error) {
            console.error("Prediction error:", error);
            status.textContent = "Detection failed: " + error.message;
        } finally {
            detectButton.disabled = false;
            detectButton.innerHTML = 'Detect Damage <span>→</span>';
        }
    });

    // Display prediction results using the existing HTML IDs.
    function displayResults(data) {
        const resultMessage = document.getElementById("resultMessage");
        const damageCount = document.getElementById("damageCount");
        const resultImage = document.getElementById("resultImage");
        const detectionList = document.getElementById("detectionList");

        resultSection.classList.remove("hidden");

        resultMessage.textContent =
            data.message || data.result || "Analysis complete.";

        const detections = Array.isArray(data.detections)
            ? data.detections
            : [];

        const count = data.count ?? data.damage_count ?? detections.length;
        damageCount.textContent = `${count} detections`;

        detectionList.replaceChildren();

        if (detections.length === 0) {
            const message = document.createElement("div");
            message.className = "no-detections";
            message.textContent =
                "No damage detected by the model in this image.";
            detectionList.appendChild(message);
        } else {
            detections.forEach((item, index) => {
                const card = document.createElement("div");
                card.className = "detection-item";

                const details = document.createElement("div");

                const title = document.createElement("strong");
                title.textContent =
                    String(
                        item.damage ||
                        item.class_name ||
                        item.class ||
                        item.label ||
                        item.name ||
                        "Unknown damage"
                    ).replace(/_/g, " ");

                details.appendChild(title);

                const description = document.createElement("small");
                description.textContent = `Detection ${index + 1}`;
                details.appendChild(description);

                const confidence = document.createElement("span");
                confidence.className = "confidence";

                if (item.confidence !== undefined) {
                    let value = Number(item.confidence);

                    if (Number.isFinite(value)) {
                        if (value <= 1) value *= 100;
                        confidence.textContent = `${value.toFixed(1)}%`;
                    } else {
                        confidence.textContent = "N/A";
                    }
                } else {
                    confidence.textContent = "N/A";
                }

                card.appendChild(details);
                card.appendChild(confidence);
                detectionList.appendChild(card);
            });
        }

        // Use the annotated image URL if the backend provides one.
        const imageUrl = data.image_url || data.result_image;

        if (imageUrl) {
            resultImage.src = imageUrl;
            resultImage.classList.remove("hidden");
        } else {
            resultImage.removeAttribute("src");
            resultImage.classList.add("hidden");
        }
    }
});
