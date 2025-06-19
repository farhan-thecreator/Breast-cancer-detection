# Breast-cancer-detection
A machine learning project that detects breast cancer using diagnostic data from the UCI Wisconsin dataset. Trains and evaluates models like Logistic Regression and Random Forest, with metrics like accuracy and ROC AUC. Includes model saving and a custom prediction function for real-world use. Built and tested on Google Colab using Kaggle API.

# 🧠 Breast Cancer Detection using ML

A simple ML project that uses the Breast Cancer Wisconsin (Diagnostic) dataset to classify tumors as **benign** or **malignant** using:
- Logistic Regression
- Random Forest

### 📊 Dataset
- [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

### 🔧 Features
- Preprocessing with StandardScaler
- Training and evaluation with scikit-learn models
- Saved model for future predictions
- Clean modular structure

### 🏁 How to Run
```bash
pip install -r requirements.txt
python src/predict.py
