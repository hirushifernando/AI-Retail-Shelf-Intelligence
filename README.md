# AI Retail Shelf Intelligence

## AI-Based Retail Product Detection & Shelf Analysis System

An end-to-end Computer Vision system for **retail product detection, product classification, product counting, and shelf inventory analysis** using deep learning and image processing.

The project combines **YOLO11n for object detection**, **PyTorch and TensorFlow/Keras for product classification**, and **OpenCV for image preprocessing and inference**.

---

## Project Overview

Retail shelves can contain multiple products with different appearances, orientations, and packaging. Manually identifying and counting products can be time-consuming.

This project explores an automated Computer Vision pipeline that can:

* Detect products on retail shelf images
* Classify detected products into product categories
* Count detected products
* Perform shelf-level inventory analysis
* Compare PyTorch and TensorFlow/Keras CNN implementations
* Apply OpenCV-based image preprocessing
* Evaluate the complete detection-to-classification pipeline

### End-to-End Pipeline

```text
Retail Shelf Image
        ↓
    YOLO11n
        ↓
Product Detection
        ↓
   Bounding Boxes
        ↓
      Cropping
        ↓
 OpenCV Preprocessing
        ↓
  PyTorch CNN
        ↓
Product Classification
        ↓
Product Counting
        ↓
Shelf Analysis
```

---

## Technologies Used

| Technology         | Purpose                                 |
| ------------------ | --------------------------------------- |
| Python             | Main programming language               |
| PyTorch            | CNN product classification              |
| TensorFlow / Keras | CNN classification comparison           |
| YOLO11n            | Retail product object detection         |
| OpenCV             | Image preprocessing and inference       |
| NumPy              | Numerical processing                    |
| Pandas             | Data processing                         |
| Matplotlib         | Visualization                           |
| scikit-learn       | Evaluation metrics                      |
| Jupyter Notebook   | Experimentation and analysis            |
| Git & Git LFS      | Version control and large model storage |

---

## Dataset

The project uses the **Indonesian Retail Product Dataset**.

The dataset contains six retail product categories:

```text
aqua
chitato
indomie
pepsodent
shampoo
tissue
```

### Dataset Characteristics

* 380 images
* 380 Pascal VOC XML annotations
* 294 training images
* 86 test images
* 6 product classes
* Pascal VOC annotation format
* YOLO-compatible annotations generated during preprocessing

The original dataset is **not included in this repository** because of its size and licensing considerations.

---

# Project Structure

```text
AI-Retail-Shelf-Intelligence/
│
├── .gitignore
├── .gitattributes
├── README.md
├── requirements.txt
│
├── models/
│   ├── retail_cnn_keras.keras
│   ├── retail_cnn_pytorch.pth
│   └── retail_cnn_pytorch_best.pth
│
├── notebooks/
│   ├── 00_environment_test.ipynb
│   ├── 03_yolo_evaluation.ipynb
│   ├── 04_product_cropping.ipynb
│   ├── 05_opencv_preprocessing.ipynb
│   ├── 06_data_augmentation.ipynb
│   ├── 07_pytorch_classification.ipynb
│   ├── 08_keras_classification.ipynb
│   ├── 09_opencv_inference.ipynb
│   └── 10_shelf_analysis.ipynb
│
└── scripts/
    ├── copy_split_annotations.py
    └── voc_to_yolo.py
```

> The dataset, preprocessing outputs, YOLO training runs, and virtual environment are excluded from the repository using `.gitignore`.

---

# Computer Vision Pipeline

## 1. Dataset Inspection

The dataset was first inspected to understand:

* Image distribution
* Product categories
* Annotation structure
* Object counts
* Image dimensions
* Pascal VOC XML structure

---

## 2. Dataset Validation

The annotations were validated to identify:

* Missing annotations
* Invalid bounding boxes
* Unknown classes
* Annotation-image mismatches
* Class distribution

The dataset was found to contain six product categories with balanced representation.

---

## 3. Pascal VOC → YOLO Conversion

The original Pascal VOC XML annotations were converted into YOLO format.

```text
Pascal VOC XML
      ↓
Bounding Box Extraction
      ↓
Coordinate Normalization
      ↓
YOLO Annotation
```

The conversion was implemented using:

```text
scripts/voc_to_yolo.py
```

---

## 4. YOLO11n Object Detection

A lightweight **YOLO11n** model was trained to detect retail products.

### Training Configuration

```text
Model: YOLO11n
Epochs: 50
Image Size: 416
Batch Size: 4
Device: CPU
Workers: 0
Patience: 10
```

### Validation Results

| Metric    | Score |
| --------- | ----: |
| Precision | 91.0% |
| Recall    | 93.7% |
| mAP@50    | 94.5% |
| mAP@50–95 | 66.1% |

### Test Results

| Metric    |     Score |
| --------- | --------: |
| Precision |     84.7% |
| Recall    |     84.4% |
| mAP@50    | **79.3%** |
| mAP@50–95 |     52.0% |

The model performed particularly well on products such as **Aqua, Chitato, and Indomie**, while visually similar or partially occluded products were more challenging.

---

# 5. Product Cropping

Detected bounding boxes were used to extract individual product images.

```text
Shelf Image
    ↓
YOLO Detection
    ↓
Bounding Boxes
    ↓
Individual Product Crops
```

These cropped products were then used for the classification stage.

---

# 6. OpenCV Preprocessing

OpenCV was used to preprocess product crops before classification.

The preprocessing pipeline includes operations such as:

* Image loading
* Resizing
* Color conversion
* Normalization
* Image preparation for CNN inference

---

# 7. Data Augmentation

Data augmentation was applied to improve classification robustness.

The augmentation process helps the CNN handle variations in:

* Product orientation
* Image position
* Scale
* Lighting
* Appearance

---

# 8. PyTorch CNN Classification

A custom CNN was implemented using **PyTorch** as the primary classification model.

The classifier predicts one of the six retail product categories:

```text
Aqua
Chitato
Indomie
Pepsodent
Shampoo
Tissue
```

### Test Results

| Metric        |     Result |
| ------------- | ---------: |
| Test Accuracy | **76.39%** |
| Macro F1      | **76.38%** |
| Weighted F1   | **76.24%** |

### Per-Class F1

| Class     | F1 Score |
| --------- | -------: |
| Aqua      |   83.33% |
| Chitato   |   70.89% |
| Indomie   |   63.89% |
| Pepsodent |   83.58% |
| Shampoo   |   81.16% |
| Tissue    |   75.41% |

---

# 9. TensorFlow / Keras Classification

A second CNN implementation was developed using **TensorFlow/Keras** to compare the two deep learning frameworks.

### Test Results

| Metric        |     Result |
| ------------- | ---------: |
| Test Accuracy | **73.15%** |
| Macro F1      |     73.00% |
| Weighted F1   |     73.00% |

### Framework Comparison

| Model       | Test Accuracy |
| ----------- | ------------: |
| PyTorch CNN |    **76.39%** |
| Keras CNN   |        73.15% |

The PyTorch implementation achieved approximately **3.24 percentage points higher test accuracy** on this dataset.

---

# 10. OpenCV Image Inference

The trained PyTorch classifier was integrated with OpenCV for image-based inference.

Example predictions included:

```text
Aqua       → Aqua        99.92%
Chitato    → Chitato     77.41%
Indomie    → Aqua        90.98%  (Incorrect)
Pepsodent  → Pepsodent   62.36%
Shampoo    → Shampoo     94.64%
Tissue     → Tissue      99.93%
```

This also demonstrates that **high confidence does not always guarantee a correct prediction**.

---

# 11. Shelf Analysis

The final stage combines the Computer Vision components into one pipeline.

```text
Input Shelf Image
       ↓
    YOLO11n
       ↓
Product Detection
       ↓
Product Cropping
       ↓
OpenCV Preprocessing
       ↓
PyTorch Classification
       ↓
Product Category
       ↓
Product Counting
       ↓
Shelf Inventory Analysis
```

### Final Pipeline Evaluation

The complete pipeline was evaluated using **30 test images**, with five images selected from each product class.

| Metric                |     Result |
| --------------------- | ---------: |
| Images Tested         |         30 |
| Products Detected     |         31 |
| Correct Predictions   |         24 |
| Incorrect Predictions |          7 |
| Pipeline Accuracy     | **77.42%** |

### Per-Class Results

| Product   | Correct / Detected | Accuracy |
| --------- | -----------------: | -------: |
| Aqua      |              4 / 5 |   80.00% |
| Chitato   |              3 / 5 |   60.00% |
| Indomie   |              4 / 5 |   80.00% |
| Pepsodent |              4 / 6 |   66.67% |
| Shampoo   |              4 / 5 |   80.00% |
| Tissue    |              5 / 5 |  100.00% |

The final pipeline demonstrates the ability to combine object detection and classification into a practical retail shelf analysis workflow.

---

# Model Performance Summary

| Component           | Model                 |         Main Result |
| ------------------- | --------------------- | ------------------: |
| Object Detection    | YOLO11n               |   **79.30% mAP@50** |
| Classification      | PyTorch CNN           | **76.39% Accuracy** |
| Classification      | Keras CNN             | **73.15% Accuracy** |
| Full Shelf Pipeline | YOLO11n + PyTorch CNN | **77.42% Accuracy** |

---

# Hardware & Environment

The project was developed and tested on a CPU-based laptop environment.

```text
Python: 3.12
CPU: Intel Core i7-1355U
RAM: 16 GB
GPU: Intel Iris Xe
CUDA: Not used
```

The models were trained using **CPU computation**, making this project suitable for experimentation on systems without an NVIDIA GPU.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/hirushifernando/AI-Retail-Shelf-Intelligence.git
cd AI-Retail-Shelf-Intelligence
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

The dataset is not included in the repository.

After obtaining the Indonesian Retail Product Dataset, place the original files inside the appropriate local data directories.

The preprocessing scripts can then be used to prepare the annotations and dataset structure.

---

# Model Files

The trained models are stored using **Git LFS** because of their large file sizes.

Available models:

```text
models/
├── retail_cnn_keras.keras
├── retail_cnn_pytorch.pth
└── retail_cnn_pytorch_best.pth
```

After cloning the repository, Git LFS should retrieve the actual model files.

If Git LFS is not already installed:

```bash
git lfs install
git lfs pull
```

---

# Notebooks

| Notebook                          | Purpose                                        |
| --------------------------------- | ---------------------------------------------- |
| `00_environment_test.ipynb`       | Verify the development environment             |
| `03_yolo_evaluation.ipynb`        | YOLO11n training and evaluation                |
| `04_product_cropping.ipynb`       | Extract product crops                          |
| `05_opencv_preprocessing.ipynb`   | OpenCV preprocessing                           |
| `06_data_augmentation.ipynb`      | Image augmentation                             |
| `07_pytorch_classification.ipynb` | PyTorch CNN training and evaluation            |
| `08_keras_classification.ipynb`   | TensorFlow/Keras CNN comparison                |
| `09_opencv_inference.ipynb`       | Image-based classification inference           |
| `10_shelf_analysis.ipynb`         | Complete shelf detection and analysis pipeline |

---

# Key Learning Outcomes

This project provided practical experience with:

* Computer Vision workflows
* Object detection
* YOLO model training
* CNN image classification
* PyTorch
* TensorFlow/Keras
* OpenCV
* Pascal VOC annotations
* YOLO annotations
* Data preprocessing
* Data augmentation
* Model evaluation
* Confusion matrices and classification metrics
* Model comparison
* Object detection + classification pipelines
* Retail shelf analysis
* Git and Git LFS

---

# Limitations

The project has several limitations:

* The dataset contains only six product categories.
* The dataset is relatively small for deep learning.
* Some products have visually similar packaging.
* Occlusion can affect detection and classification.
* The system was evaluated on a limited test set.
* CPU-only training increases training time.
* The current shelf analysis focuses on still images rather than real-time video.

Therefore, the reported results should be considered experimental results on this dataset rather than production-level performance.

---

# Future Improvements

Possible future improvements include:

* Expanding the number of retail product categories
* Training with a larger and more diverse dataset
* Improving detection of heavily occluded products
* Using stronger classification architectures
* Applying transfer learning
* Improving product tracking across shelf images
* Developing real-time camera-based inference
* Deploying the system as a web application or API
* Adding automated inventory reporting

---

# Project Status

The project successfully demonstrates an end-to-end retail Computer Vision pipeline combining:

**YOLO11n + PyTorch + TensorFlow/Keras + OpenCV**

for product detection, classification, counting, and shelf analysis.

---

## Author

**Hirushi Fernando**

BSc (Hons) Computer Science

