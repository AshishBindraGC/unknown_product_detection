import os
import cv2
import torch
from ultralytics import YOLO
from torchvision.ops import nms

# ======================================
# LOAD MODELS (ORDER = PRIORITY)
# First model = Highest Priority
# ======================================

MODEL_PATHS = [
    # 🔥 Highest Priority
    "/mnt/data/ashish/storage/models/Rice_V8X_103C_25AUG25_640B.pt",
    # Medium Priority
    "/mnt/data/ashish/storage/models/Rice_V8X_132C_02OCT25_640B.pt",
    # Lowest Priority
    "/mnt/data/ashish/storage/models/best-2.pt",
]

MODELS = [YOLO(p) for p in MODEL_PATHS]


# ======================================
# CONFIG
# ======================================

CONF_THRESHOLD = 0.25

# Best for FMCG shelves
NMS_IOU = 0.2


# ======================================
# PATHS
# ======================================

input_folder = "/mnt/data/ashish/storage/input/KRBL_rice/valid"

output_folder = "/mnt/data/ashish/storage/output/final_data4"

image_out = os.path.join(output_folder, "images")
label_out = os.path.join(output_folder, "labels")

os.makedirs(image_out, exist_ok=True)
os.makedirs(label_out, exist_ok=True)


# ======================================
# PROCESS IMAGES
# ======================================

num_models = len(MODELS)

image_list = os.listdir(input_folder)

for img_name in image_list:
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    print("Processing:", img_name)

    img_path = os.path.join(input_folder, img_name)

    image = cv2.imread(img_path)

    if image is None:
        continue

    h, w, _ = image.shape

    all_boxes = []
    all_scores = []
    all_labels = []

    # ======================================
    # RUN MODELS
    # ======================================

    for idx, model in enumerate(MODELS):
        # First model highest priority
        priority = num_models - idx

        results = model(image, conf=CONF_THRESHOLD)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # ======================================
            # HYBRID SCORE
            # Confidence first
            # Priority second
            # ======================================

            conf_int = int(conf * 100)

            boosted_score = conf_int * 100 + priority

            all_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])

            all_scores.append(boosted_score)

            all_labels.append(cls_id)

    if len(all_boxes) == 0:
        continue

    # ======================================
    # CROSS MODEL NMS
    # ======================================

    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)

    keep = nms(boxes_tensor, scores_tensor, NMS_IOU)

    final_boxes = boxes_tensor[keep]
    final_labels = [all_labels[i] for i in keep]

    # ======================================
    # SAVE LABELS (YOLO FORMAT)
    # ======================================

    label_path = os.path.join(label_out, img_name.rsplit(".", 1)[0] + ".txt")

    with open(label_path, "w") as f:
        for i in range(len(final_boxes)):
            x1, y1, x2, y2 = final_boxes[i]

            cls_id = final_labels[i]

            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h

            width = (x2 - x1) / w
            height = (y2 - y1) / h

            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

            # Draw box
            cv2.rectangle(
                image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2
            )

    # ======================================
    # SAVE IMAGE
    # ======================================

    save_img_path = os.path.join(image_out, img_name)

    cv2.imwrite(save_img_path, image)

    print("Saved:", save_img_path)


print("\n✅ Dataset Created Successfully")
print("Rule: Higher confidence wins")
print("Tie: Higher priority model wins")
print("First model = Highest priority")
