# 🐄 Cattle Skin Disease Prediction using Deep Learning and Self-Supervised Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg)
![Keras](https://img.shields.io/badge/Keras-CNN%20Models-red.svg)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Classification-green.svg)
![Self-Supervised Learning](https://img.shields.io/badge/Self--Supervised-SimCLR-purple.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

---

## 📌 Project Overview

This project is a **cattle skin disease prediction system** built using **computer vision, transfer learning, and self-supervised learning**.

The main objective is to classify cattle skin images into disease categories and support early disease detection. This can help farmers, livestock owners, and veterinary support systems identify possible diseases faster from image data.

The project compares multiple deep learning models, including standard CNN-based transfer learning architectures and a **self-supervised SimCLR fine-tuned model**.

The best-performing model was:

> **SimCLR Fine-Tuned Model**
>
> **Test Accuracy: 95.48%**

---

## 🚀 Key Highlights

- Developed an end-to-end cattle skin disease classification pipeline.
- Used deep learning and computer vision for image-based disease prediction.
- Implemented transfer learning models including **DenseNet121**, **ResNet50**, and **EfficientNetB0**.
- Implemented a **self-supervised SimCLR fine-tuned model** for stronger feature representation.
- Achieved **95.48% test accuracy** using the SimCLR fine-tuned model.
- Evaluated models using Accuracy, Loss, Precision, Recall, F1-score, ROC-AUC, Precision-Recall Curve, and Confusion Matrix.
- Compared all models through a structured final result table.

---

## 🧠 Problem Statement

Cattle diseases can seriously affect livestock health, farming productivity, and the economic stability of farmers.

Traditional disease identification usually depends on expert observation, which can be slow, expensive, or unavailable in rural areas. A computer vision-based disease classification system can help provide faster initial screening by analyzing cattle skin images.

This project aims to build a deep learning-based image classification system that can identify cattle skin disease categories accurately and reliably.

---

## 🎯 Project Objectives

The main objectives of this project are:

- To build a cattle skin disease classification model using image data.
- To compare traditional transfer learning models with a self-supervised learning approach.
- To evaluate model performance using multiple classification metrics.
- To identify the best-performing model for cattle skin disease prediction.
- To demonstrate the usefulness of self-supervised learning in medical/agricultural image classification.

---

## 📂 Dataset

The dataset was downloaded using KaggleHub.

```python
import kagglehub

data_path = kagglehub.dataset_download("devang03mgr/cattle-diseases-datasets")
print("Dataset downloaded to:", data_path)
```

The dataset contains cattle skin images from three classes:

```text
foot-and-mouth
healthy
lumpy
```

These classes represent different cattle skin conditions, including healthy and diseased categories.

---

## 🛠️ Technologies Used

| Category | Tools / Libraries |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Machine Learning Utilities | Scikit-learn |
| Dataset Loading | KaggleHub |
| Model Type | CNN, Transfer Learning, Self-Supervised Learning |
| Main Self-Supervised Method | SimCLR |

---

## 🧪 Methodology

The project follows a complete deep learning workflow.

---

### 1. Dataset Loading

The dataset was downloaded from Kaggle using KaggleHub.  
Image paths and labels were collected and organized into a structured format for preprocessing and model training.

---

### 2. Exploratory Data Analysis

The notebook includes basic EDA to understand the dataset.

EDA steps include:

- Checking available disease classes
- Counting images per class
- Visualizing class distribution
- Displaying sample images from each class

This step helped identify the dataset structure and understand how the cattle skin images were distributed across classes.

---

### 3. Image Preprocessing

Before training, images were processed into a format suitable for CNN-based models.

Preprocessing steps include:

- Reading images using OpenCV
- Converting images into RGB format
- Resizing images to `224 x 224`
- Normalizing pixel values
- Encoding class labels
- Splitting the data into training, validation, and test sets

The use of consistent image size and normalization helped improve training stability.

---

### 4. Model Training

The project trained and compared four major models:

| Model | Learning Approach |
|---|---|
| DenseNet121 | Transfer Learning |
| ResNet50 | Transfer Learning |
| EfficientNetB0 | Transfer Learning |
| SimCLR Fine-Tuned | Self-Supervised Learning + Fine-Tuning |

---

## 🧬 Models Implemented

### DenseNet121

DenseNet121 was used as a transfer learning model with a pretrained CNN backbone.  
It helped establish a strong baseline for image classification.

---

### ResNet50

ResNet50 was used because of its residual learning structure, which helps train deeper neural networks more effectively.

---

### EfficientNetB0

EfficientNetB0 was used as a lightweight transfer learning model.  
It was included to compare whether a smaller and more efficient model could perform competitively on the disease classification task.

---

### SimCLR Fine-Tuned Model

SimCLR was used as the self-supervised learning approach.

SimCLR first learns image representations through contrastive learning. It creates augmented versions of images and trains the model to recognize similar image pairs while separating different image pairs.

After this self-supervised pretraining stage, the learned backbone was fine-tuned for the cattle disease classification task.

This model achieved the best performance in the project.

---

## 🔍 Why Self-Supervised Learning Matters Here

Self-supervised learning is powerful because it allows the model to learn useful visual patterns without depending only on manually labeled examples.

For cattle skin disease images, this is important because disease patterns can appear through:

- Skin texture
- Swelling
- Lesion shape
- Color differences
- Infected regions
- Surface irregularities

The SimCLR model learned stronger image representations and performed better than the standard transfer learning models.

---

## 🏆 Final Model Results

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| **SimCLR Fine-Tuned** | **95.48%** | **0.1496** |
| ResNet50 | 95.28% | 0.1851 |
| DenseNet121 | 94.46% | 0.1822 |
| EfficientNetB0 | 88.30% | 0.3882 |

---

## 🥇 Best Model

```text
Best Model: SimCLR Fine-Tuned
Test Accuracy: 95.48%
Test Loss: 0.1496
Macro Average AUC: 0.9943
Average Precision Macro: 0.9905
```

The **SimCLR Fine-Tuned model** achieved the highest test accuracy and showed the strongest overall performance.

---

## 📊 Classification Report of Best Model

### SimCLR Fine-Tuned Model

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| foot-and-mouth | 1.0000 | 0.9464 | 0.9725 |
| healthy | 0.9583 | 0.9485 | 0.9534 |
| lumpy | 0.9259 | 0.9669 | 0.9459 |

Overall performance:

```text
Accuracy: 95.48%
Macro F1-score: 95.73%
Weighted F1-score: 95.50%
```

---

## 📈 Evaluation Metrics

The following evaluation metrics were used:

- Accuracy
- Test Loss
- Precision
- Recall
- F1-score
- Macro F1-score
- Weighted F1-score
- Confusion Matrix
- ROC-AUC Curve
- Precision-Recall Curve
- Average Precision Score

Using multiple metrics gives a more reliable understanding of model performance instead of depending only on accuracy.

---

## 📊 Visualizations Included

The notebook includes several visual outputs:

- Class distribution plot
- Sample image visualization
- Training accuracy curve
- Validation accuracy curve
- Training loss curve
- Validation loss curve
- Confusion matrix
- ROC-AUC curve
- Precision-Recall curve
- Final model comparison table

These visualizations make the results easier to understand and help compare the models clearly.

---

## 📁 Project Structure

A clean GitHub repository structure can look like this:

```text
Cattle-Skin-Disease-Prediction/
│
├── README.md
├── cattle-skin-disease-prediction.ipynb
├── cattle_disease_all_models_results.csv
│
├── models/
│   ├── best_densenet_cattle.keras
│   ├── best_resnet50_cattle.keras
│   ├── best_effnetb0_cattle.keras
│   ├── best_simclr_fine.keras
│   └── simclr_pretrained_backbone.keras
│
└── outputs/
    ├── class_distribution.png
    ├── sample_images.png
    ├── training_history.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── precision_recall_curve.png
```

If the model files are too large, upload them using **Git LFS** or keep only the notebook, README, result CSV, and output images.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/cattle-skin-disease-prediction.git
cd cattle-skin-disease-prediction
```

Install required dependencies:

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib seaborn scikit-learn kagglehub tqdm
```

---

## ▶️ How to Run the Project

Open the notebook:

```bash
jupyter notebook cattle-skin-disease-prediction.ipynb
```

Or run it in Google Colab.

Then execute the notebook cells in order:

1. Import libraries
2. Download/load dataset
3. Perform EDA
4. Preprocess images
5. Train models
6. Evaluate models
7. Compare final results

---

## 🧾 Sample Code for Dataset Download

```python
import kagglehub

path = kagglehub.dataset_download("devang03mgr/cattle-diseases-datasets")
print("Path to dataset files:", path)
```

---

## 📌 Result Summary

The project showed that self-supervised learning can improve visual feature extraction for disease classification tasks.

Among all tested models, **SimCLR Fine-Tuned** performed the best.

```text
Final Best Model: SimCLR Fine-Tuned
Final Test Accuracy: 95.48%
Final Macro AUC: 0.9943
Final Average Precision Macro: 0.9905
```

---

## 💡 Key Insights

- SimCLR produced the strongest performance because it learned better image representations before supervised fine-tuning.
- ResNet50 also performed very well, but SimCLR achieved slightly higher accuracy and better overall reliability.
- EfficientNetB0 performed the weakest in this experiment, showing that lightweight models are not always the best option for disease image classification.
- Accuracy alone is not enough; F1-score, ROC-AUC, and Precision-Recall metrics give a better performance picture.

---

## ✅ Conclusion

This project successfully developed a cattle skin disease classification system using deep learning and self-supervised learning.

The best model, **SimCLR Fine-Tuned**, achieved **95.48% test accuracy**, showing that self-supervised representation learning can be highly effective for cattle disease image classification.

This type of system can be extended into practical livestock health monitoring tools, mobile-based disease screening apps, or veterinary decision-support systems.

---

## 🔮 Future Improvements

Possible future improvements include:

- Increasing dataset size with more disease categories
- Using external validation data
- Applying Grad-CAM for explainable AI visualization
- Deploying the model as a web or mobile application
- Building a farmer-friendly interface for disease prediction
- Adding treatment recommendation support after classification
- Improving robustness under different lighting and camera conditions

---

## 👨‍💻 Author

**Shafin Mahamud**  
Computer Science Student
Project Developer
Brac University
