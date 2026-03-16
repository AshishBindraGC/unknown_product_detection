import roboflow
import os
import random
import string
import shutil


# ============== CONFIG ==============

ANNOTATION_NAMES = [
    "RICE_A_Ashish_done",
    "RICE_B_Ashish_done",
    "RICE_C_Ashish_done",
    "RICE_D_Ashish_done",
    "RICE_E_Ashish_done",
    "RICE_F_Ashish_done",
    "RICE_G_Ashish_done",
]

BATCH_SIZE = 200


# ============== RANDOM NAME ==============


def random_name():

    return "Batch_" + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=6)
    )


# ============== SPLIT INTO BATCHES ==============


def split_batches(dataset_path):

    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")

    images = sorted(os.listdir(images_path))

    batches = []

    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i : i + BATCH_SIZE]
        batches.append(batch)

    return batches


# ============== CREATE TEMP BATCH FOLDER ==============


def create_batch_folder(dataset_path, batch_files, batch_id):

    temp_folder = f"/tmp/rf_batch_{batch_id}"

    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

    os.makedirs(temp_folder + "/images", exist_ok=True)
    os.makedirs(temp_folder + "/labels", exist_ok=True)

    for file in batch_files:
        img_src = os.path.join(dataset_path, "images", file)
        lbl_src = os.path.join(
            dataset_path, "labels", os.path.splitext(file)[0] + ".txt"
        )

        shutil.copy(img_src, temp_folder + "/images/")

        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, temp_folder + "/labels/")

    return temp_folder


# ============== MAIN UPLOAD FUNCTION ==============


def data_upload(
    api_key="v0yOaHuEZ5QLlx1e9Ng8",
    dataset_path="/mnt/data/ashish/src/rice.v3i.yolov8/valid",
    project_name="project-usmwk",
):

    rf = roboflow.Roboflow(api_key=api_key)

    workspace = rf.workspace()

    # print("Workspace:", workspace)

    # ---------- CREATE PROJECT IF NOT EXISTS ----------

    try:
        project = workspace.project(project_name)

        print("Project exists:", project_name)

    except:
        print("Creating Project:", project_name)

        project = workspace.create_project(
            project_name, project_type="object-detection"
        )

    # ---------- SPLIT BATCHES ----------

    batches = split_batches(dataset_path)

    print("Total batches:", len(batches))

    # ---------- UPLOAD LOOP ----------

    # ---------- UPLOAD LOOP ----------

    for i, batch_files in enumerate(batches):
        print("\nUploading Batch:", i + 1)

        # Pick annotation name
        if i < len(ANNOTATION_NAMES):
            batch_name = ANNOTATION_NAMES[i]
        else:
            batch_name = random_name()

        print("Batch Name:", batch_name)

        # Create temp folder
        batch_folder = create_batch_folder(dataset_path, batch_files, i)

        # ✅ Correct upload method
        workspace.upload_dataset(
            batch_folder,
            project_name=project_name,
            num_workers=10,
            project_license="MIT",
            project_type="object-detection",
            batch_name=batch_name,
            num_retries=3,
            is_prediction=False,
        )

        print("Uploaded:", batch_name)

        shutil.rmtree(batch_folder)

        print("\n✅ Upload Complete")


# ================= RUN =================

if __name__ == "__main__":
    data_upload()
