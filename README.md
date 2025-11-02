# FaceID-ML: Facial Recognition with PCA and XGBoost

A comprehensive machine learning system for person identification through geometric facial feature extraction, XGBoost classification, and Principal Component Analysis (PCA).

## 📋 Project Overview

This project implements a complete pipeline for facial recognition:
- **Facial feature extraction** using MediaPipe
- **Person identification** using XGBoost classifier
- **Dimensionality reduction** with PCA analysis
- **Model evaluation** and visualization

## 🎯 Features

### 1. Facial Feature Extraction
- Extracts 19 geometric features from facial landmarks
- Uses MediaPipe for robust face detection
- Features focus on bone structure (expression-independent):
  - Face dimensions (width, height, depth)
  - Eye characteristics (spacing, socket dimensions)
  - Nose measurements (length, width, bridge)
  - Facial proportions and ratios

### 2. Machine Learning Classification
- **XGBoost** classifier with GPU acceleration
- **SMOTE** for handling class imbalance
- Hyperparameter tuning with GridSearchCV
- Comprehensive performance metrics

### 3. PCA Analysis
- Principal Component Analysis for dimensionality reduction
- 3D visualizations of data distribution
- Contribution analysis (individuals and variables)
- Correlation sphere visualization

## 🗂️ Project Structure

```
AD_TP6/
├── person_feature_extraction.py    # Extract facial features from images
├── person_training_model.py        # Train XGBoost classifier
├── person_testing_model.py         # Test model on new images
├── acp_person.ipynb               # PCA analysis notebook
├── dataset2/                       # Training dataset (7 persons)
│   ├── Ariel_Sharon/
│   ├── Colin_Powell/
│   ├── Donald_Rumsfeld/
│   ├── George_W_Bush/
│   ├── Gerhard_Schroeder/
│   ├── Hugo_Chavez/
│   └── Tony_Blair/
├── test_images/                    # Images for testing
└── test_results/                   # Annotated output images
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 📖 Usage

### 1. Extract Facial Features

```bash
python person_feature_extraction.py
```

This will:
- Process all images in `dataset2/`
- Extract 19 geometric features per face
- Save features to `person_facial_features.csv`

### 2. Train the Model

```bash
python person_training_model.py
```

This will:
- Load facial features
- Balance dataset using SMOTE
- Train XGBoost classifier with hyperparameter tuning
- Save model files (`person_identification_model.json`, etc.)
- Generate confusion matrix and feature importance plots

### 3. Test on New Images

```bash
python person_testing_model.py
```

This will:
- Load trained model
- Process images from `test_images/`
- Predict person identity with confidence scores
- Save annotated results to `test_results/`

### 4. Run PCA Analysis

Open and run `acp_person.ipynb` in Jupyter:

```bash
jupyter notebook acp_person.ipynb
```

## 📊 Results

The project achieves:
- High classification accuracy on test set
- Meaningful principal components capturing facial variations
- Clear separation of individuals in PCA space

## 🛠️ Technologies Used

- **Python 3.x**
- **MediaPipe** - Facial landmark detection
- **XGBoost** - Gradient boosting classifier
- **scikit-learn** - Machine learning utilities
- **imbalanced-learn** - SMOTE for data balancing
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **OpenCV** - Image processing

## 📝 Dataset

The project uses images of 7 political figures:
- Ariel Sharon
- Colin Powell
- Donald Rumsfeld
- George W. Bush
- Gerhard Schroeder
- Hugo Chavez
- Tony Blair

## 🙏 Acknowledgments

- MediaPipe by Google for facial landmark detection
- XGBoost development team
- scikit-learn community
