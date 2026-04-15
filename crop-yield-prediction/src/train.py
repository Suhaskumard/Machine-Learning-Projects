import pandas as pd
import numpy as np
import pickle
import os   # ✅ added

# ✅ create models folder automatically
os.makedirs("../models", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/crop_production.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop missing values
df = df.dropna()

# Feature engineering
df["Yield"] = df["Production"] / df["Area"]



# Remove invalid rows
df = df[(df["Area"] > 0) & (df["Production"] > 0)]

# Clean text data BEFORE encoding (fix unseen labels)
df["State_Name"] = df["State_Name"].str.strip().str.lower()
df["Crop"] = df["Crop"].str.strip().str.lower()
df["Season"] = df["Season"].str.strip().str.lower()

# Encode categorical variables
le_state = LabelEncoder()
le_crop = LabelEncoder()
le_season = LabelEncoder()

df["State_Name"] = le_state.fit_transform(df["State_Name"])
df["Crop"] = le_crop.fit_transform(df["Crop"])
df["Season"] = le_season.fit_transform(df["Season"])

# Features & target
X = df[["State_Name", "Crop", "Season", "Area"]]
y = df["Yield"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (faster + better)
model = RandomForestRegressor(
    n_estimators=100,   # 🔥 balanced speed + accuracy
    max_depth=10,
    n_jobs=-1,          # 🔥 uses all CPU cores
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
with open("../models/model.pkl", "wb") as f:
    pickle.dump((model, le_state, le_crop, le_season), f)

print("✅ Model saved successfully!")