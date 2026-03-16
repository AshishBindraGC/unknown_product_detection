import os
import cv2
import torch
from ultralytics import YOLO
from torchvision.ops import nms

# ==============================
# CONFIG

# baby_care
# bar
# chamanprass
# HAIR_OIL
# RB_condom
# RICE
# sampro
# SOAP
# TOOTHPASTE

# ==============================

INPUT_FOLDER = "/mnt/data/ashish/train/images"
OUTPUT_FOLDER = "/mnt/data/ashish/storage/output/ecpl_prdict_sap"

CONF_THRESHOLD = 0.25
NMS_IOU = 0.01

# 🔥 Just add/remove models here
MODELS = [
    # "/mnt/data/ashish/storage/models/new_model.pt",
    # "/mnt/data/ashish/storage/models/Haleon_Toothpaste_V8X_11Nov25_640_B.pt",
    # "/mnt/data/ashish/storage/models/best-2.pt",
    "/mnt/data/ashish/storage/models/yolov11_suk10k_1k.pt"
]

# ==============================
# DEVICE AUTO SELECT
# ==============================

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("Using Device:", DEVICE)

# ==============================
# LOAD MODELS DYNAMICALLY
# ==============================

# MODEL_FOLDER = "/mnt/data/ashish/storage/models"

# MODELS = [
#     os.path.join(MODEL_FOLDER,f)
#     for f in os.listdir(MODEL_FOLDER)
#     if f.endswith(".pt")
# ]

models = []

for m in MODELS:
    print("Loading:", m)
    models.append(YOLO(m))

# ==============================
# OUTPUT PATHS
# ==============================

image_out = os.path.join(OUTPUT_FOLDER, "images")
label_out = os.path.join(OUTPUT_FOLDER, "labels")

os.makedirs(image_out, exist_ok=True)
os.makedirs(label_out, exist_ok=True)

# ==============================
# PROCESS IMAGES
# ==============================

for img_name in os.listdir(INPUT_FOLDER):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_FOLDER, img_name)

    print("\nProcessing:", img_name)

    image = cv2.imread(img_path)

    if image is None:
        print("Image not found:", img_name)
        continue

    h, w, _ = image.shape

    all_boxes = []
    all_scores = []
    all_labels = []

    # ==============================
    # RUN ALL MODELS
    # ==============================

    for model in models:
        results = model.predict(
            image, conf=CONF_THRESHOLD, device=DEVICE, verbose=False
        )[0]

        if results.boxes is None:
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, score, cls_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box

            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(score)
            all_labels.append(int(cls_id))

    # ==============================
    # SKIP IF NO DETECTION
    # ==============================

    if len(all_boxes) == 0:
        continue

    # ==============================
    # CROSS MODEL NMS
    # ==============================

    boxes_tensor = torch.tensor(all_boxes)
    scores_tensor = torch.tensor(all_scores)

    keep = nms(boxes_tensor, scores_tensor, NMS_IOU)

    final_boxes = boxes_tensor[keep]
    final_labels = [all_labels[i] for i in keep]

    # ==============================
    # SAVE LABELS
    # ==============================

    label_path = os.path.join(label_out, img_name.rsplit(".", 1)[0] + ".txt")

    with open(label_path, "w") as f:
        for i in range(len(final_boxes)):
            x1, y1, x2, y2 = final_boxes[i]
            cls_id = final_labels[i]

            # YOLO format
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            f.write(f"{cls_id} {xc} {yc} {bw} {bh}\n")

            # Draw box
            cv2.rectangle(
                image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2
            )

    # Save image
    cv2.imwrite(os.path.join(image_out, img_name), image)

    print("Saved:", img_name)


print("\n✅ Roboflow-ready dataset created")
