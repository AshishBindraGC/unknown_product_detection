import os
import shutil

# =============================
# PATH CONFIG
# =============================

TXT_FILE = "/mnt/data/ashish/correct_detection_soap.txt"

SOURCE_IMAGES = "/mnt/data/ashish/train/images"
SOURCE_PRED = "/mnt/data/ashish/storage/output/ecpl_prdict_sap/images"
SOURCE_LABELS = "/mnt/data/ashish/storage/output/ecpl_prdict_sap/labels"

OUTPUT_DIR = "output_soap"

# Accepted folder
ACC_IMG = os.path.join(OUTPUT_DIR, "accepted/images")
ACC_PRED = os.path.join(OUTPUT_DIR, "accepted/predictions")
ACC_LABEL = os.path.join(OUTPUT_DIR, "accepted/labels")

# Underdetected folder
UND_IMG = os.path.join(OUTPUT_DIR, "underdetected/images")
UND_PRED = os.path.join(OUTPUT_DIR, "underdetected/predictions")
UND_LABEL = os.path.join(OUTPUT_DIR, "underdetected/labels")

# =============================
# CREATE FOLDERS
# =============================

for path in [ACC_IMG, ACC_PRED, ACC_LABEL, UND_IMG, UND_PRED, UND_LABEL]:
    os.makedirs(path, exist_ok=True)

# =============================
# READ IMAGE LIST
# =============================

accepted_images = set()

with open(TXT_FILE, "r") as f:
    for line in f:
        accepted_images.add(line.strip())

# =============================
# PROCESS ALL IMAGES
# =============================

for img_name in os.listdir(SOURCE_IMAGES):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    label_name = os.path.splitext(img_name)[0] + ".txt"

    img_path = os.path.join(SOURCE_IMAGES, img_name)
    pred_path = os.path.join(SOURCE_PRED, img_name)
    label_path = os.path.join(SOURCE_LABELS, label_name)

    # Decide destination
    if img_name in accepted_images:
        dest_img = ACC_IMG
        dest_pred = ACC_PRED
        dest_label = ACC_LABEL
    else:
        dest_img = UND_IMG
        dest_pred = UND_PRED
        dest_label = UND_LABEL

    # Copy files
    if os.path.exists(img_path):
        shutil.copy(img_path, dest_img)

    if os.path.exists(pred_path):
        shutil.copy(pred_path, dest_pred)

    if os.path.exists(label_path):
        shutil.copy(label_path, dest_label)

print("Dataset split completed")
