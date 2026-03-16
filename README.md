# Unknown Product Detection – Analysis and Resolution

## 1. Problem Statement

During model inference, the system was incorrectly detecting **unknown products**. The number of false detections increased significantly, which affected the reliability of the product detection pipeline.

This issue required investigation to identify the root cause and determine a stable solution.

---

## 2. Root Cause Analysis

After analysis, the issue was traced to two main causes:

### 2.1 Incorrect Training Labels

Some images in the training dataset contained **incorrect or unnecessary annotations**.
In several cases, **text or non-product elements were labeled as objects**, which introduced noise into the training data. As a result, the model learned incorrect patterns and produced false detections.

### 2.2 Very Low Inference Threshold

The inference confidence threshold was set to **0.01** to maximize detections.

While this allowed the model to detect more objects, it also significantly increased **false positives**, leading to incorrect identification of unknown products.

---

## 3. Solution Approaches Evaluated

Multiple approaches were tested to mitigate the issue:

1. Fine-tuning the existing retail model with a cleaned dataset
2. Increasing the confidence threshold during inference
3. Filtering bounding boxes based on aspect ratio
4. Retraining the model using a larger public dataset combined with internal data

---

## 4. Evaluation of Approaches

### 4.1 Fine-Tuning the Existing Retail Model

Fine-tuning the original retail model with a cleaned dataset was tested.

**Observation:**

* The model required **frequent fine-tuning for each product category**.
* Whenever a category was missing or incorrectly detected, additional fine-tuning was needed.

**Conclusion:**
This approach was **not scalable** and difficult to maintain.

---

### 4.2 Increasing the Confidence Threshold

The confidence threshold was increased from **0.01 to 0.55**.

**Observation:**

* Detection accuracy improved.
* False positives decreased.

**Limitation:**

* The number of **unknown product detections decreased significantly**, which was not desirable for the system requirements.

**Conclusion:**
This approach improved precision but **reduced recall for unknown products**, therefore it was not adopted.

---

### 4.3 Bounding Box Filtering Using Aspect Ratio

A filtering mechanism was introduced in the inference pipeline.

**Implementation:**

* Bounding boxes with **very small aspect ratio areas (less than 50 pixels)** were discarded when drawing labels on images.

**Observation:**

* This reduced several incorrect detections.

**Limitation:**

* Certain product categories such as **toothpaste** were also filtered out because they naturally have smaller bounding boxes.

**Conclusion:**
Due to the negative impact on valid detections, this approach was discarded.

---

## 5. Final Solution Implemented

The final approach involved **training a new model using a larger dataset**.

### Dataset Used

* **SKU-110K public dataset:** ~11,000 images
* **Internal dataset:** ~1,900 images

These datasets were combined and used to train a new model.

### Result

* Improved detection performance
* Reduced incorrect unknown product detections
* Better generalization across product categories

---

## 6. Current Deployment Strategy

The newly trained model is currently being used in the detection pipeline.

For future improvements:

* Any additional training will be performed by **fine-tuning this new model**, rather than modifying the original retail model.
* This ensures **better maintainability and scalability** of the system.

---

Below is your **content rewritten in professional, clear, and structured documentation style** suitable for **company documentation, Confluence, or technical reports**. I corrected grammar, clarified sentences, and organized it logically while keeping your technical details.

---

## Auto Labeling & Dataset Upload Pipeline Flow

## 1. Pipeline Overview

The auto-labeling and dataset upload pipeline consists of the following stages:

1. **Dataset Preparation** (Data download and preprocessing)
2. **YOLO Detection** (Two approaches) and **Final Label Creation** (merging labels and removing overlaps)
3. **Label Normalization** (setting all classes to a single class)
4. **Dataset Structuring & YAML File Generation**
5. **Roboflow Batch Upload**

---

## 1. Dataset Preparation

### Process

* Collect or download datasets.
* Data can be downloaded using the **Python SDK** or manually from **Roboflow**.
* Merge multiple datasets into a single temporary folder.
* Convert dataset structure from **train/test/valid** format to **images/labels** format.

### Folder Structure

```bash
temp/
 ├── images/
 └── labels/
```

---

## 2. YOLO Detection

For auto-labeling, two detection approaches are used:

1. **Ground Truth (GT) based detection**
2. **Model-based detection**

---

## 2.1 Ground Truth (GT) Approach

In this approach, datasets downloaded from **Roboflow** contain both **images and their existing labels**.

### Workflow

1. Existing labels (Ground Truth) are first collected into a list using a Python script.
2. The same images are then passed to the **other.pt model** for additional predictions.
3. Detected labels from the model are also added to the list.
4. **Priority is always given to Ground Truth labels.**

If duplicate or overlapping detections occur for the same product item, **YOLO’s built-in Non-Maximum Suppression (NMS)** is applied to remove overlaps.

**NMS configuration**

* IoU threshold: **0.01 – 0.1**

This ensures that duplicate bounding boxes are removed.

---

## 2.2 Model-Based Detection Approach

In this method, multiple product-specific models are used.

For example:

* If the product category is **rice**, we may use:

  * **Rice Model 1**
  * **Rice Model 2**
  * **Other Products Model**

### Workflow

1. The same image is passed to all models.
2. Each model performs inference independently.
3. All detected labels are merged together.
4. **Non-Maximum Suppression (NMS)** is applied to remove overlapping bounding boxes.

### Observation

This method produces **more predictions compared to the Ground Truth approach**, making it useful for discovering additional objects that may not have been labeled previously.

---

## 3. Label Normalization

After detection, multiple classes may exist in the labels.

To simplify training, **all class IDs are normalized to a single class (class = 0)**.

### Implementation

A Python script is used to modify the label files and set all class IDs to **0**.

Example:

```bash
Before:
2 0.45 0.52 0.12 0.18
5 0.32 0.48 0.20 0.30

After:
0 0.45 0.52 0.12 0.18
0 0.32 0.48 0.20 0.30
```

---

## 4. Dataset Structuring & YAML File Generation

After labels are processed:

1. Images are collected from the **original input dataset**.
2. Processed labels are collected from the **modified labels folder**.
3. Images and labels are then structured into the final dataset format required for training.

Example structure:

```bash
dataset/
 ├── images/
 ├── labels/
 └── data.yaml
```

The **YAML file** contains dataset configuration used for YOLO training.

---

## 5. Roboflow Batch Upload

The dataset is uploaded to **Roboflow** using the **Python SDK and API**.

### Benefits of Automated Upload

Manual upload:

* Takes approximately **1 hour**

Automated upload using Python:

* Takes approximately **15 minutes**

### Required Configuration

* **Workspace API Key**
* **Project ID**

### Batch Upload Process

Images are uploaded in batches such as:

```bash
BATCH_A → 200 images
BATCH_B → 200 images
```

Batch size and batch naming can be configured in the script.

---

## Model Training Dataset

For model training, the following datasets were used:

### Public Dataset

* **SKU-110K Dataset**
* Approximately **10,000–11,000 images**

### Internal Dataset

* **~1,900 images collected internally**

Both datasets were combined to create the final training dataset.

---

## Models Trained

Two different YOLO models were trained:

1. **YOLOv8 Model**
2. **YOLOv11 Model**

After evaluation, **YOLOv11 showed better detection performance compared to YOLOv8.**

---

## Data Collection Process

### Auto-labeling Workflow

1. Data was auto-labeled using the detection pipeline.
2. The dataset was then sent for **manual verification** to ensure that all products were correctly labeled.
3. Verified data was uploaded to **Roboflow** and added to the training dataset.

---

## Dataset Labeling Effort

Total labeling effort took approximately **2 weeks**.

### Team Contribution

| Team Member   | Task                                 |
| ------------- | ------------------------------------ |
| Ashish        | Prepared **500 rice dataset images** |
| Vineet        | Prepared **toothpaste dataset**      |
| Aditya        | Assisted with toothpaste dataset     |
| Labeling Team | Initial labeling support             |

### Total Dataset

| Category       | Total Images |
| -------------- | ------------ |
| Rice           | 489          |
| Sanpro         | 239          |
| Toothpaste     | 351          |
| Handwash       | 272          |
| Toilet Cleaner | 527          |
| **Total**      | **1878**     |

Vineet also prepared an **Excel tracking sheet** to monitor:

* Total labeled images
* Remaining images to be labeled

---

### Training Data Preparation

The training dataset preparation process included:

1. Uploading approximately **11 GB of data to Google Drive**
2. Downloading the dataset into **Google Colab**
3. Unzipping the dataset
4. Converting annotations from **CSV format to YOLO `.txt` format**
5. Downloading internal datasets from **Roboflow**
6. Merging public and internal datasets
7. Preparing the final dataset for training

---

# Data Sources

The following sources were used to collect product data:

| S.No. | Source / Client    | Product Category      |
| ----- | ------------------ | --------------------- |
| 1     | KRBL               | Rice                  |
| 2     | KRBL               | Oil                   |
| 3     | KRBL               | Biryani Masala        |
| 4     | Racket             | 28 Product Categories |
| 5     | GCPL / Dabur / AKP | Dabur Products        |
| 6     | Additional         | Toothpaste            |
| 7     | Additional         | Condoms               |
| 8     | Additional         | Soap                  |
| 9     | Additional         | Handwash              |

---
