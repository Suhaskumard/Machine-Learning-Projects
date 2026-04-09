# 💳 Credit Card Fraud Detection

## 📌 Overview
This project uses Machine Learning to detect fraudulent credit card transactions.
It handles highly imbalanced data and evaluates performance using relevant metrics.

---

## 📂 Project Structure

```

fraud-detection/
│── fraud_detection.py
│── data_preprocessing.py
│── model_utils.py
│── config.py
│── requirements.txt
│── README.md

```

---

## 📊 Dataset

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Features: V1 to V28 (anonymized)
- Target:
  - 0 → Normal
  - 1 → Fraud

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

1. Download dataset and place `creditcard.csv` in root folder

2. Run:

```bash
python fraud_detection.py
```

---

## 🤖 Model

* Logistic Regression

---

## 📈 Metrics Used

* Accuracy
* Precision
* Recall
* F1 Score

---

## ⚠️ Important Note

Dataset is highly imbalanced → Accuracy is misleading.
Focus on Recall & Precision.

---

## 🔥 Future Improvements

* SMOTE (handling imbalance)
* Random Forest / XGBoost
* Deep Learning (Neural Networks)
* Deploy API (Flask / FastAPI)
* Dashboard (Streamlit)

---

