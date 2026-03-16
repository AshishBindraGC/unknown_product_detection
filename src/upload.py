import roboflow


def data_upload(
    api_key="40jmI9aQ4qKZbFDu27yb",
    path=r"/mnt/data/ashish/storage/output/final",
    project_name="project-usmwk",
):
    """
    function to upload data on roboflow
    """
    rf = roboflow.Roboflow(api_key=api_key)

    workspace = rf.workspace()
    print(workspace)

    # project = rf.workspace().project("soap_20")
    # version = project.version(1)
    # # download the model
    # version.download(model_format="yolov8", location="./downloads")

    # Upload data set to a new/existing project
    workspace.upload_dataset(
        path,  # This is your dataset path
        project_name=project_name,  # This will either create or get a dataset with the given ID
        num_workers=10,
        project_license="MIT",
        project_type="object-detection",
        batch_name=None,
        num_retries=0,
        is_prediction=False,  # optional, set to True if the dataset is not ground truth and needs approval
    )


data_upload()
