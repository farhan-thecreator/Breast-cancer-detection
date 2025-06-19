
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


model = joblib.load("models/breast_cancer_rf.pkl")
scaler = joblib.load("models/scaler.pkl")  # Save this if not already

def predict(sample_features):
    sample = np.array(sample_features).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    prob = model.predict_proba(sample_scaled)[0][1]
    label = "Malignant" if prob > 0.5 else "Benign"
    return label, prob
