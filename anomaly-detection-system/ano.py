import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
 
# -----------------------------
# CONFIG
# -----------------------------
MODEL_TYPE = "isolation_forest"  # options: isolation_forest, lof, svm
CONTAMINATION = 0.05

# -----------------------------
# LOAD DATA
# -----------------------------


numeric_data = data.select_dtypes(include=['float64', 'int64'])
numeric_data = numeric_data.fillna(numeric_data.mean())

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# -----------------------------
# MODEL SELECTION
# -----------------------------
if MODEL_TYPE == "isolation_forest":
    model = IsolationForest(contamination=CONTAMINATION, random_state=42)
    preds = model.fit_predict(scaled_data)

elif MODEL_TYPE == "lof":
    model = LocalOutlierFactor(contamination=CONTAMINATION)
    preds = model.fit_predict(scaled_data)

elif MODEL_TYPE == "svm":
    model = OneClassSVM(nu=CONTAMINATION)
    preds = model.fit_predict(scaled_data)

else:
    raise ValueError("Invalid model type")

# -----------------------------
# SAVE RESULTS
# -----------------------------
data["Anomaly"] = preds
data.to_csv("output.csv", index=False)

print("✅ Anomaly detection completed")

# -----------------------------
# MODEL COMPARISON
# -----------------------------
iso = IsolationForest(contamination=CONTAMINATION).fit_predict(scaled_data)
lof = LocalOutlierFactor(contamination=CONTAMINATION).fit_predict(scaled_data)

comparison = pd.DataFrame({
    "IsolationForest": iso,
    "LOF": lof
})

comparison["Agreement"] = comparison["IsolationForest"] == comparison["LOF"]
comparison.to_csv("comparison.csv", index=False)

print("📊 Model comparison saved")

# -----------------------------
# PCA VISUALIZATION
# -----------------------------
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_data)

plt.scatter(reduced[:, 0], reduced[:, 1], c=preds, cmap='coolwarm')
plt.title("PCA Visualization of Anomalies")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# -----------------------------
# REAL-TIME INPUT
# -----------------------------
def predict_new(data_point):
    data_point = np.array(data_point).reshape(1, -1)
    scaled = scaler.transform(data_point)
    
    if MODEL_TYPE == "lof":
        print("⚠️ LOF does not support new predictions easily")
        return
    
    result = model.predict(scaled)
    return "Anomaly" if result[0] == -1 else "Normal"

# Example usage
# print(predict_new([10, 20, 30]))
