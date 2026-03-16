import os
import cv2
import torch
from ultralytics import YOLO

# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = "/mnt/data/ashish/storage/models/best-2.pt"

IMAGE_FOLDER = "/mnt/data/ashish/storage/input/Self_Toothpaste-17/test/images"
GT_LABEL_FOLDER = "/mnt/data/ashish/storage/input/Self_Toothpaste-17/test/labels"
OUTPUT_LABEL_FOLDER = "/mnt/data/ashish/storage/input/Self_Toothpaste-17/output/merged_labels"
VISUAL_OUTPUT_FOLDER = "/mnt/data/ashish/storage/input/Self_Toothpaste-17/output/visualized"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)
os.makedirs(VISUAL_OUTPUT_FOLDER, exist_ok=True)

device = 0 if torch.cuda.is_available() else "cpu"

# ==========================================
# LOAD MODEL
# ==========================================

model = YOLO(MODEL_PATH)
model.to(device)

# ==========================================
# HELPER FUNCTIONS
# ==========================================


def yolo_to_xyxy(cls, xc, yc, w, h):
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter_area

    if union == 0:
        return 0

    return inter_area / union


# ==========================================
# PROCESS IMAGES
# ==========================================

for img_name in os.listdir(IMAGE_FOLDER):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    print(f"Processing: {img_name}")

    img_path = os.path.join(IMAGE_FOLDER, img_name)
    label_name = img_name.rsplit(".", 1)[0] + ".txt"
    gt_label_path = os.path.join(GT_LABEL_FOLDER, label_name)

    image = cv2.imread(img_path)
    if image is None:
        continue

    h, w, _ = image.shape

    # ------------------------------
    # 1️⃣ Run Detection
    # ------------------------------

    results = model(image, conf=CONF_THRESHOLD, device=device)[0]

    pred_boxes = []
    pred_lines = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # Convert to normalized YOLO format
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        pred_lines.append(f"{cls_id} {x_center} {y_center} {width} {height}\n")
        pred_boxes.append(
            (cls_id, yolo_to_xyxy(cls_id, x_center, y_center, width, height))
        )

    # ------------------------------
    # 2️⃣ Read Ground Truth
    # ------------------------------

    gt_lines = []
    gt_boxes = []

    if os.path.exists(gt_label_path):
        with open(gt_label_path, "r") as f:
            gt_lines = f.readlines()

        for line in gt_lines:
            parts = line.strip().split()
            cls = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            gt_boxes.append((cls, yolo_to_xyxy(cls, xc, yc, bw, bh)))

    # ------------------------------
    # 3️⃣ Merge (Remove Duplicates)
    # ------------------------------

    merged_lines = gt_lines.copy()

    for pred_line, (pred_cls, pred_box) in zip(pred_lines, pred_boxes):
        duplicate = False

        for gt_cls, gt_box in gt_boxes:
            if pred_cls != gt_cls:
                continue

            iou = compute_iou(pred_box, gt_box)

            if iou > IOU_THRESHOLD:
                duplicate = True
                break

        if not duplicate:
            merged_lines.append(pred_line)

    # ------------------------------
    # 4️⃣ Save Merged Labels
    # ------------------------------

    output_label_path = os.path.join(OUTPUT_LABEL_FOLDER, label_name)

    with open(output_label_path, "w") as f:
        f.writelines(merged_lines)

    # ------------------------------
    # 5️⃣ Optional Visualization
    # ------------------------------

    for line in merged_lines:
        parts = line.strip().split()
        cls = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(VISUAL_OUTPUT_FOLDER, img_name), image)

print("✅ Detection + Smart Merge Completed")
