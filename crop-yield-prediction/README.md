# 🌾 Crop Yield Prediction using Machine Learning

## 📌 Overview
This project predicts crop yield using real-world agricultural data from India. It uses machine learning techniques to analyze crop production patterns.

---

## 🚀 Features
- Real dataset from Kaggle
- Data cleaning & preprocessing
- Feature engineering (Yield calculation)
- Machine Learning model (Random Forest)
- Model evaluation (RMSE, R²)
- Data visualization (EDA)

---

## 📂 Project Structure
```
crop-yield-prediction/
│
├── data/
├── models/
├── src/
├── requirements.txt
└── README.md
```

---

## 📊 Dataset
Source: Kaggle - Crop Production in India

**Download instructions:**
1. Go to https://www.kaggle.com/datasets/abhinand05/crop-production-in-india
2. Download `crop_production.csv`
3. Place in `data/crop_production.csv`

Features:
- State Name
- Crop
- Season
- Area
- Production

---

## 🧠 Model Used
Random Forest Regressor

---

## 📈 Results
- Good prediction accuracy
- R² Score ~ 0.8+
- Low RMSE

---

## ⚙️ Setup Instructions

### 1. Navigate to project
```
cd crop-yield-prediction
```

### 2. Create environment
```
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

---

## ▶️ Run Project

### Train Model
```
cd src
python train.py
```

### Predict
```
python predict.py
```

### EDA (Visualization)
```
python eda.py
```

---

## 🌍 Future Improvements
- Add rainfall dataset
- Integrate weather API
- Deploy using FastAPI
- Add deep learning models

---

## 💼 Use Cases
- Smart agriculture
- Yield prediction
- Government planning

---

## 📜 License
MIT License
