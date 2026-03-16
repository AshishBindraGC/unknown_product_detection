import os
import shutil


def merge_dataset(
    dataset_root="/mnt/data/ashish/storage/input/OIL-3",
    input_path="/mnt/data/ashish/storage/output/final_dataset",
):
    """
    # ===== INPUT DATASET ROOT =====

    # dataset_root = "/mnt/data/ashish/storage/input/OIL-3"

    # Example:
    # dataset/
    #   train/images
    #   train/labels
    #   test/images
    #   test/labels
    #   valid/images
    #   valid/labels


    # ===== OUTPUT =====

    # input_path = "/mnt/data/ashish/storage/output/final_dataset"
    """

    OUT_IMAGES = os.path.join(input_path, "images")
    OUT_LABELS = os.path.join(input_path, "labels")

    os.makedirs(OUT_IMAGES, exist_ok=True)
    os.makedirs(OUT_LABELS, exist_ok=True)

    # ===== SPLITS =====

    SPLITS = ["train", "test", "valid"]

    grand_total = 0

    for split in SPLITS:
        img_folder = os.path.join(dataset_root, split, "images")
        lbl_folder = os.path.join(dataset_root, split, "labels")

        if not os.path.exists(img_folder):
            print("Missing:", split)
            continue

        images = os.listdir(img_folder)

        print(f"\nProcessing {split}")
        print("Images:", len(images))

        count = 0

        for img_file in images:
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            name = os.path.splitext(img_file)[0]

            img_src = os.path.join(img_folder, img_file)
            lbl_src = os.path.join(lbl_folder, name + ".txt")

            shutil.copy2(img_src, os.path.join(OUT_IMAGES, img_file))

            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, os.path.join(OUT_LABELS, name + ".txt"))

            count += 1
            grand_total += 1

        print(f"{split} copied:", count)

    print("\n✅ Done")
    print("Grand Total:", grand_total)


if __name__ == "__main__":
    merge_dataset()
