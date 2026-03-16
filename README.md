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

