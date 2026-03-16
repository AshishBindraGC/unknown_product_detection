import os
import cv2
import torch
from ultralytics import YOLO
from torchvision.ops import box_iou

from src.config import OUTPUT_FOLDER, MODEL_OTHERS_PATH, INPUT_FOLDER
from src.utilty import read_yolo_txt, create_yaml

# data_download,
from src.label_zero import set_lebal_zero
# from src.upload import data_upload

CONF_THRESHOLD = 0.55
IOU_THRESHOLD = 0.1


# Load Model
model_others = YOLO(MODEL_OTHERS_PATH)


# ================= COLLECT BOXES WITH CONF =================


def collect_model_boxes(results):

    boxes = []
    labels = []
    confs = []

    if results.boxes is None:
        return boxes, confs, labels

    xyxy = results.boxes.xyxy.cpu().numpy()
    cls = results.boxes.cls.cpu().numpy()
    conf = results.boxes.conf.cpu().numpy()

    for i in range(len(xyxy)):
        boxes.append(xyxy[i])
        labels.append(int(cls[i]))
        confs.append(float(conf[i]))

    return boxes, confs, labels


# ================= PROCESS =================


def process(input_path, output_path):
    IMAGE_OUT = os.path.join(output_path, "images")
    LABEL_OUT = os.path.join(output_path, "labels")

    IMAGE_INPUT = os.path.join(input_path, "images")
    LABEL_INPUT = os.path.join(input_path, "labels")

    os.makedirs(IMAGE_OUT, exist_ok=True)
    os.makedirs(LABEL_OUT, exist_ok=True)
    for img_name in os.listdir(IMAGE_INPUT):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(IMAGE_INPUT, img_name)
        label_path = os.path.join(LABEL_INPUT, os.path.splitext(img_name)[0] + ".txt")

        print("Processing:", img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]

        # ---------- GT ----------
        gt_boxes, _, gt_labels = read_yolo_txt(label_path, w, h)

        # ---------- YOLO ----------
        results = model_others(image, conf=CONF_THRESHOLD)[0]

        pred_boxes, pred_confs, pred_labels = collect_model_boxes(results)

        final_boxes = []
        final_labels = []
        final_confs = []

        # 1️⃣ Keep GT
        final_boxes.extend(gt_boxes)
        final_labels.extend(gt_labels)
        final_confs.extend([1.0] * len(gt_boxes))  # GT confidence = 1

        # 2️⃣ Add Predictions if not overlapping GT

        if pred_boxes:
            if gt_boxes:
                gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
                pred_tensor = torch.tensor(pred_boxes, dtype=torch.float32)

                ious = box_iou(pred_tensor, gt_tensor)

                for i in range(len(pred_boxes)):
                    if ious[i].max().item() < IOU_THRESHOLD:
                        final_boxes.append(pred_boxes[i])
                        final_labels.append(pred_labels[i])
                        final_confs.append(pred_confs[i])

            else:
                final_boxes.extend(pred_boxes)
                final_labels.extend(pred_labels)
                final_confs.extend(pred_confs)

        if not final_boxes:
            continue

        # ---------- SAVE LABELS ----------

        out_label_path = os.path.join(LABEL_OUT, os.path.splitext(img_name)[0] + ".txt")

        with open(out_label_path, "w", encoding="utf-8") as f:
            for i, box in enumerate(final_boxes):
                x1, y1, x2, y2 = box

                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                conf = final_confs[i]

                f.write(
                    f"{final_labels[i]} {x_center} {y_center} {bw} {bh} {conf:.3f}\n"
                )

        # ---------- DRAW ----------

        # GT → GREEN
        for box in gt_boxes:
            x1, y1, x2, y2 = box

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.putText(
                image,
                "GT",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Predictions → RED

        for i in range(len(gt_boxes), len(final_boxes)):
            x1, y1, x2, y2 = final_boxes[i]
            conf = final_confs[i]

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            cv2.putText(
                image,
                f"YOLO {conf:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        cv2.imwrite(os.path.join(IMAGE_OUT, img_name), image)

        print("Saved:", img_name)

    print("\n✅ Dataset created")


if __name__ == "__main__":
    # Data Download
    # data_download()

    # data copy and merged
    from src.merge_data import merge_dataset
    import shutil

    new_input = INPUT_FOLDER + "/../temp"
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Remove if already exists
    if os.path.exists(new_input):
        shutil.rmtree(new_input)

    # Create fresh folder
    os.makedirs(new_input, exist_ok=True)
    merge_dataset(dataset_root=INPUT_FOLDER, input_path=new_input)
    # data process
    process(new_input, OUTPUT_FOLDER)
    #  set label to zero
    LABEL_OUT = os.path.join(OUTPUT_FOLDER, "labels")

    set_lebal_zero(LABEL_OUT)

    # make data uploadble
    from src.datasets_test import prepare_dataset

    new_input = INPUT_FOLDER + "/../temp"
    dataset_path = OUTPUT_FOLDER + "/../final_dataset"
    prepare_dataset(
        orignal_image_path=new_input,
        output_path=OUTPUT_FOLDER,
        dataset_path=OUTPUT_FOLDER + "/../final_data1",
    )
    # create yaml file
    create_yaml(OUTPUT_FOLDER + "/../final_data1")
    # Uploading data batch-wise
