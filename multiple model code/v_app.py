"""vineet code provided"""

import os
import cv2
import torch
from ultralytics import YOLO
from torchvision.ops import nms

# ============================== # LOAD MODELS # ==============================

model_shop = YOLO("/mnt/data/ashish/storage/models/Rice_V8X_103C_25AUG25_640B.pt")
model_toothpaste = YOLO("/mnt/data/ashish/storage/models/Rice_V8X_132C_02OCT25_640B.pt")
model_others = YOLO("/mnt/data/ashish/storage/models/best-2.pt")

# ============================== # PATHS # ==============================

input_folder = "/mnt/data/ashish/storage/input/KRBL_rice/valid"
output_folder = "/mnt/data/ashish/storage/output/final_data_vineet"
image_out = os.path.join(output_folder, "images")
label_out = os.path.join(output_folder, "labels")

os.makedirs(image_out, exist_ok=True)
os.makedirs(label_out, exist_ok=True)

CONF_THRESHOLD = 0.25
NMS_IOU = 0.01

# ==============================
# PROCESS IMAGES
# ==============================

for img_name in os.listdir(input_folder):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, img_name)
        print("Processing:", img_name)

        image = cv2.imread(img_path)
        h, w, _ = image.shape

        # Run models
        r1 = model_shop(image, conf=CONF_THRESHOLD)[0]
        r2 = model_toothpaste(image, conf=CONF_THRESHOLD)[0]
        r3 = model_others(image, conf=CONF_THRESHOLD)[0]

        all_boxes = []
        all_scores = []
        all_labels = []

        # Helper function
        def collect_boxes(results):
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(conf)
                all_labels.append(cls_id)

        # Collect detections
        collect_boxes(r1)
        collect_boxes(r2)
        collect_boxes(r3)

        if len(all_boxes) == 0:
            continue

        boxes_tensor = torch.tensor(all_boxes)
        scores_tensor = torch.tensor(all_scores)

        # 🔥 Cross Model NMS
        keep = nms(boxes_tensor, scores_tensor, NMS_IOU)

        final_boxes = boxes_tensor[keep]
        final_scores = scores_tensor[keep]
        final_labels = [all_labels[i] for i in keep]

        # ==============================
        # SAVE LABELS (ROBOfLOW FORMAT)
        # ==============================

        label_path = os.path.join(label_out, img_name.rsplit(".", 1)[0] + ".txt")

        with open(label_path, "w") as f:
            for i in range(len(final_boxes)):
                x1, y1, x2, y2 = final_boxes[i]
                cls_id = final_labels[i]

                # Convert to YOLO normalized format
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                # ⭐ Roboflow format (NO CONFIDENCE)
                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

                # Draw final box
                cv2.rectangle(
                    image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2
                )

        # Save image
        save_img_path = os.path.join(image_out, img_name)
        cv2.imwrite(save_img_path, image)

        print("Saved:", save_img_path)

print("✅ Roboflow-ready dataset created")
