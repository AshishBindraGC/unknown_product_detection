import os
import shutil
# from config import INPUT_FOLDER, OUTPUT_FOLDER


def prepare_dataset(orignal_image_path, output_path, dataset_path):
    """
    orignal_image_path = "/mnt/data/ashish/storage/input/images"
    detected_label_path = "/mnt/data/ashish/storage/output/labels"

    dataset_path = "/mnt/data/ashish/storage/final_dataset"
    """
    orignal_images = os.path.join(orignal_image_path, "images")

    final_images = os.path.join(dataset_path, "images")
    final_labels = os.path.join(dataset_path, "labels")

    os.makedirs(final_images, exist_ok=True)
    os.makedirs(final_labels, exist_ok=True)

    detected_label_path = os.path.join(output_path, "labels")
    labels = os.listdir(detected_label_path)

    total = 0

    for label_file in labels:
        if not label_file.endswith(".txt"):
            continue

        name = os.path.splitext(label_file)[0]

        img_jpg = os.path.join(orignal_images, name + ".jpg")
        img_png = os.path.join(orignal_images, name + ".png")
        img_jpeg = os.path.join(orignal_images, name + ".jpeg")

        label_src = os.path.join(detected_label_path, label_file)
        label_dst = os.path.join(final_labels, label_file)

        # -------- FIND IMAGE --------

        if os.path.exists(img_jpg):
            image_src = img_jpg

        elif os.path.exists(img_png):
            image_src = img_png

        elif os.path.exists(img_jpeg):
            image_src = img_jpeg

        else:
            print("Image not found:", name)
            continue

        image_dst = os.path.join(final_images, os.path.basename(image_src))

        # -------- COPY IMAGE --------

        shutil.copy2(image_src, image_dst)

        # -------- MOVE LABEL --------

        shutil.move(label_src, label_dst)

        total += 1

        if total % 100 == 0:
            print("Processed:", total)

    print("\n✅ Done")
    print("Total Files:", total)


if __name__ == "__main__":
    new_input = INPUT_FOLDER + "/../temp"

    prepare_dataset(
        orignal_image_path=new_input,
        output_path=OUTPUT_FOLDER,
        dataset_path=OUTPUT_FOLDER + "/../final_data",
    )
    # create yaml file
