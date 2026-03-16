from ultralytics import YOLO

# Load model
model = YOLO("/mnt/data/ashish/storage/models/best-2.pt")

# Image folder
image_folder = "/mnt/data/ashish/storage/input/Self_Toothpaste-17/test/images"

# Output settings
output_project = "/mnt/data/ashish/storage/output"  # new folder
output_name = "Toothpaste_detection"  # subfolder name

# Run detection + save
results = model(
    image_folder,
    save=True,
    save_txt=True,
    project=output_project,
    name=output_name,
    conf=0.25,
    iou=0.20,
)

# Print results
for r in results:
    print("Image:", r.path)

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        print(f"Detected: {model.names[cls_id]} | Confidence: {conf:.2f}")

print("Done ✅")
