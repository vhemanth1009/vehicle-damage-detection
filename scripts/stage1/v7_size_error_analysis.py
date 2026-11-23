from pathlib import Path
import pandas as pd
from PIL import Image

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\vehicle-damage-detection")

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "stage1"
    / "combined_v7"
)

TEST_IMAGES_DIR = DATASET_DIR / "images" / "test"
TEST_LABELS_DIR = DATASET_DIR / "labels" / "test"

ERROR_CSV = (
    PROJECT_DIR
    / "results"
    / "stage1_yolo11n_combined_v7_10ep_final_test"
    / "error_analysis"
    / "v7_error_analysis.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "stage1_yolo11n_combined_v7_10ep_final_test"
    / "error_analysis"
    / "size_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "glass_shatter",
    4: "lamp_broken",
    5: "tire_flat"
}

# ============================================================
# SIZE THRESHOLDS
# ============================================================

SMALL_THRESHOLD = 0.01
MEDIUM_THRESHOLD = 0.05

# ============================================================
# LOAD ERROR CSV
# ============================================================

print("=" * 70)
print("V7 SIZE-BASED ERROR ANALYSIS")
print("=" * 70)

print()
print("Loading error CSV...")

errors = pd.read_csv(ERROR_CSV)

print(
    "Error records:",
    len(errors)
)

print()

# ============================================================
# CREATE IMAGE -> ERROR TYPE MAPPING
# ============================================================

error_map = {}

for _, row in errors.iterrows():

    image_name = str(row["image"])

    error_type = str(row["error_type"])

    ground_truth = str(
        row["ground_truth_class"]
    )

    key = (
        image_name,
        ground_truth
    )

    # Store multiple error types if present

    if key not in error_map:

        error_map[key] = []

    error_map[key].append(error_type)

# ============================================================
# ANALYZE TEST LABELS
# ============================================================

records = []

image_files = sorted(
    list(TEST_IMAGES_DIR.glob("*.jpg"))
    + list(TEST_IMAGES_DIR.glob("*.jpeg"))
    + list(TEST_IMAGES_DIR.glob("*.png"))
)

print(
    "Test images:",
    len(image_files)
)

print()

processed = 0

for image_path in image_files:

    image_name = image_path.name

    label_path = (
        TEST_LABELS_DIR
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():

        continue

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    try:

        with Image.open(image_path) as img:

            image_width, image_height = img.size

    except Exception:

        continue

    # --------------------------------------------------------
    # Read YOLO labels
    # --------------------------------------------------------

    with open(
        label_path,
        "r"
    ) as f:

        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:

            continue

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        class_name = CLASS_NAMES.get(
            class_id,
            str(class_id)
        )

        # ----------------------------------------------------
        # Relative bounding-box area
        # ----------------------------------------------------

        relative_area = (
            box_width * box_height
        )

        # ----------------------------------------------------
        # Size category
        # ----------------------------------------------------

        if relative_area < SMALL_THRESHOLD:

            size_category = "small"

        elif relative_area < MEDIUM_THRESHOLD:

            size_category = "medium"

        else:

            size_category = "large"

        # ----------------------------------------------------
        # Approximate pixel dimensions
        # ----------------------------------------------------

        pixel_width = (
            box_width * image_width
        )

        pixel_height = (
            box_height * image_height
        )

        # ----------------------------------------------------
        # Determine whether this GT class/image has error
        # ----------------------------------------------------

        key = (
            image_name,
            class_name
        )

        error_types = error_map.get(
            key,
            []
        )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        if error_types:

            for error_type in error_types:

                records.append({

                    "image":
                        image_name,

                    "class":
                        class_name,

                    "size":
                        size_category,

                    "relative_area":
                        relative_area,

                    "pixel_width":
                        pixel_width,

                    "pixel_height":
                        pixel_height,

                    "error_type":
                        error_type

                })

        else:

            records.append({

                "image":
                    image_name,

                "class":
                    class_name,

                "size":
                    size_category,

                "relative_area":
                    relative_area,

                "pixel_width":
                    pixel_width,

                "pixel_height":
                    pixel_height,

                "error_type":
                    "correct_or_not_in_error_csv"

            })

    processed += 1

    if processed % 50 == 0:

        print(
            f"Processed {processed}/{len(image_files)}"
        )

# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)

output_csv = (
    OUTPUT_DIR
    / "v7_size_error_records.csv"
)

df.to_csv(
    output_csv,
    index=False
)

# ============================================================
# ERROR-ONLY DATA
# ============================================================

error_df = df[
    df["error_type"] !=
    "correct_or_not_in_error_csv"
].copy()

# ============================================================
# SIZE SUMMARY
# ============================================================

print()
print("=" * 70)
print("ERRORS BY OBJECT SIZE")
print("=" * 70)

size_summary = (
    error_df
    .groupby("size")
    .size()
    .sort_values(
        ascending=False
    )
)

print(
    size_summary
)

# ============================================================
# CLASS + SIZE
# ============================================================

print()
print("=" * 70)
print("ERRORS BY CLASS AND SIZE")
print("=" * 70)

class_size_summary = pd.crosstab(
    error_df["class"],
    error_df["size"]
)

print(
    class_size_summary
)

# ============================================================
# CLASS + ERROR TYPE + SIZE
# ============================================================

print()
print("=" * 70)
print("ERRORS BY CLASS / SIZE / ERROR TYPE")
print("=" * 70)

detailed_summary = (
    error_df
    .groupby(
        [
            "class",
            "size",
            "error_type"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)

print(
    detailed_summary.to_string(
        index=False
    )
)

# ============================================================
# MISSED DAMAGE ONLY
# ============================================================

print()
print("=" * 70)
print("MISSED DAMAGE BY CLASS AND SIZE")
print("=" * 70)

missed_df = error_df[
    error_df["error_type"]
    == "missed_damage"
]

missed_summary = pd.crosstab(
    missed_df["class"],
    missed_df["size"]
)

print(
    missed_summary
)

# ============================================================
# LOW CONFIDENCE
# ============================================================

print()
print("=" * 70)
print("LOW CONFIDENCE BY CLASS AND SIZE")
print("=" * 70)

low_conf_df = error_df[
    error_df["error_type"]
    == "low_confidence"
]

low_conf_summary = pd.crosstab(
    low_conf_df["class"],
    low_conf_df["size"]
)

print(
    low_conf_summary
)

# ============================================================
# SAVE SUMMARIES
# ============================================================

class_size_summary.to_csv(
    OUTPUT_DIR
    / "v7_errors_class_size.csv"
)

missed_summary.to_csv(
    OUTPUT_DIR
    / "v7_missed_damage_class_size.csv"
)

low_conf_summary.to_csv(
    OUTPUT_DIR
    / "v7_low_confidence_class_size.csv"
)

detailed_summary.to_csv(
    OUTPUT_DIR
    / "v7_detailed_size_error_summary.csv",
    index=False
)

print()
print("=" * 70)
print("SIZE ERROR ANALYSIS COMPLETED")
print("=" * 70)

print()
print("Records:")
print(output_csv)

print()
print("Class + size:")
print(
    OUTPUT_DIR
    / "v7_errors_class_size.csv"
)

print()
print("Missed damage:")
print(
    OUTPUT_DIR
    / "v7_missed_damage_class_size.csv"
)

print()
print("Low confidence:")
print(
    OUTPUT_DIR
    / "v7_low_confidence_class_size.csv"
)

print()
print("Detailed:")
print(
    OUTPUT_DIR
    / "v7_detailed_size_error_summary.csv"
)

print()
print("=" * 70)