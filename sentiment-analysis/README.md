# 🎬 Sentiment Analysis using LSTM

This project performs sentiment analysis on movie reviews using Deep Learning (LSTM).

---

## 📂 Dataset Format

train.csv / test.csv must contain:

| review                  | sentiment | 
|------------------------|----------|
| \"Great movie!\"         | 1        |
| \"Worst film ever\"      | 0        |
 
---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Train the Model

```bash
python train.py
```

---

## 🔍 Predict Review

```bash
python predict.py
```

---

## 🧠 Model Architecture

* Embedding Layer
* LSTM Layer (128 units)
* Dense Layer
* Dropout
* Output Layer (Sigmoid)

---

## 📊 Output

* Binary classification:

  * 1 → Positive 😊
  * 0 → Negative 😠

---

## 💡 Features

* Text preprocessing
* Tokenization + padding
* Deep learning with LSTM
* Easy to extend

---

## 🔮 Future Improvements

* Use BERT / Transformers 🤖
* Add attention mechanism
* Hyperparameter tuning
* Deploy as API (Flask/FastAPI)

---

