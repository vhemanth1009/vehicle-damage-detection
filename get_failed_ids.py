import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\vehicle-damage-detection")

ORIGINAL_CSV = (
    PROJECT_DIR
    / "metadata"
    / "VehicleImageUri.csv"
)

FAILED_CSV = (
    PROJECT_DIR
    / "metadata"
    / "retry_failed_images.csv"
)

OUTPUT_CSV = (
    PROJECT_DIR
    / "metadata"
    / "failed_image_vehicle_ids.csv"
)

# ============================================================
# LOAD FILES
# ============================================================

print("=" * 70)
print("GENERATING IDS FOR FAILED IMAGES")
print("=" * 70)

print("\nLoading original CSV...")

original_df = pd.read_csv(ORIGINAL_CSV)

print("Original rows:", len(original_df))

print("\nLoading failed-image report...")

failed_df = pd.read_csv(FAILED_CSV)

print("Failed image records:", len(failed_df))

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

if "row" not in failed_df.columns:
    print("ERROR: 'row' column not found in retry report.")
    exit()

if "id" not in original_df.columns:
    print("ERROR: 'id' column not found in original CSV.")
    exit()

# ============================================================
# CONVERT ROW NUMBER TO ORIGINAL CSV INDEX
# ============================================================

# retry report uses:
# row 1 = original_df index 0
# row 459 = original_df index 458

failed_df["original_index"] = (
    failed_df["row"] - 1
)

# ============================================================
# GET ORIGINAL IDs
# ============================================================

failed_df["id"] = failed_df["original_index"].apply(
    lambda x:
    original_df.iloc[int(x)]["id"]
    if 0 <= int(x) < len(original_df)
    else None
)

# ============================================================
# GET VEHICLE INFORMATION
# ============================================================

failed_df["vehicleId"] = failed_df["original_index"].apply(
    lambda x:
    original_df.iloc[int(x)]["vehicleId"]
    if 0 <= int(x) < len(original_df)
    else None
)

# ============================================================
# CREATE CLEAN REPORT
# ============================================================

result = failed_df[
    [
        "id",
        "partner_id",
        "vehicleId",
        "row",
        "side",
        "download_type",
        "filename",
        "url",
        "error"
    ]
].copy()

# ============================================================
# REMOVE DUPLICATE IDS
# ============================================================

unique_ids = result["id"].dropna().drop_duplicates()

# ============================================================
# SAVE COMPLETE REPORT
# ============================================================

result.to_csv(
    OUTPUT_CSV,
    index=False
)

# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    "Failed image records :",
    len(result)
)

print(
    "Unique vehicle IDs   :",
    len(unique_ids)
)

print(
    "IDs with missing map :",
    result["id"].isna().sum()
)

print()
print("Complete report:")
print(OUTPUT_CSV)

print()
print("=" * 70)
print("FAILED VEHICLE IDS")
print("=" * 70)

for vehicle_id in unique_ids:
    print(vehicle_id)

print()
print("=" * 70)
print("COMPLETED")
print("=" * 70)