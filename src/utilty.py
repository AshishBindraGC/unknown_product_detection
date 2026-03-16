import os
from roboflow import Roboflow

yaml_filename = "data.yaml"

# ========================= # CREATE YAML CONTENT # =========================


def create_yaml(dataset_path):
    """
    create yaml
    """
    yaml_content = f"""# Auto-generated YOLOv8 dataset config

path: {dataset_path}

train: images
val: images

nc: 1

names:
- other
    """

    # ========================= # SAVE FILE # =========================

    yaml_path = os.path.join(dataset_path, yaml_filename)

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"✅ YAML file created at: {yaml_path}")


def data_download():
    """data downloading code"""
    # rf = Roboflow(api_key="JgcTrPxYULngJ3en8o9F")
    # project = rf.workspace("toothpaste-ct64e").project("self_toothpaste")
    # version = project.version(17)
    # dataset = version.download("yolov8")

    # from roboflow import Roboflow

    # rf = Roboflow(api_key="FNzIvJXH41bNZtsShiFI")
    # project = rf.workspace("krbl-oil").project("oil-ulubk")
    # version = project.version(3)
    # dataset = version.download("yolov8",location="")

    # !pip install roboflow

    # from roboflow import Roboflow

    # rf = Roboflow(api_key="40jmI9aQ4qKZbFDu27yb")
    # project = rf.workspace("aerhome1brandwise").project("rice-test-ytpm3")
    # version = project.version(1)
    # dataset = version.download("yolov8")


# ============================== FUNCTIONS ==============================


def read_yolo_txt(label_path, img_w, img_h):
    """Read labeled data"""
    boxes, scores, labels = [], [], []

    if not os.path.exists(label_path):
        return boxes, scores, labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            xc *= img_w
            yc *= img_h
            w *= img_w
            h *= img_h

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            boxes.append([x1, y1, x2, y2])
            scores.append(1.0)
            labels.append(cls)

    return boxes, scores, labels


def collect_model_boxes(results):
    """
    Bonding box module..
    """
    boxes, scores, labels = [], [], []

    if results.boxes is None or len(results.boxes) == 0:
        return boxes, scores, labels

    xyxy = results.boxes.xyxy.cpu()
    conf = results.boxes.conf.cpu()
    cls = results.boxes.cls.cpu()

    for j, _ in enumerate(xyxy):
        boxes.append(_.tolist())
        scores.append(float(conf[j]))
        labels.append(int(cls[j]))

    return boxes, scores, labels


if __name__ == "__main__":
    data_download()
