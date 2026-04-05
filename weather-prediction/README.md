# 🌦️ Weather Prediction Model

An end-to-end **Machine Learning project** that predicts weather conditions (Sunny, Rainy, Cloudy) based on environmental parameters like temperature, humidity, wind speed, and atmospheric pressure.

---

## 🚀 Project Overview

This project uses a **Random Forest Classifier** to analyze weather-related features and predict the corresponding weather condition. It is designed to be simple, extendable, and beginner-friendly while still following good ML practices.

---

## 🎯 Features

* 🌡️ Predicts weather based on input parameters
* ⚡ Fast and efficient ML model (Random Forest)
* 📊 Easy to train and test with custom datasets
* 🔁 Reusable and extendable code structure
* 🧠 Beginner-friendly implementation

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:**

  * pandas
  * numpy
  * scikit-learn
  * joblib

---

## 📂 Project Structure

```
weather-prediction/
│── data.csv                # Dataset
│── train.py               # Model training script
│── predict.py             # Prediction script
│── weather_model.pkl      # Saved model (generated after training)
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/weather-prediction.git
cd weather-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

Run the training script:

```bash
python train.py
```

✔️ This will:

* Load the dataset
* Train the model
* Save it as `weather_model.pkl`

---

## 🔮 Make Predictions

Run:

```bash
python predict.py
```

📌 Example Output:

```
🌤️ Predicted Weather: Sunny
```

---

## 📊 Dataset Details

The dataset contains the following features:

| Feature     | Description                |
| ----------- | -------------------------- |
| Temperature | Measured in °C             |
| Humidity    | Percentage (%)             |
| Wind Speed  | km/h                       |
| Pressure    | Atmospheric pressure (hPa) |
| Weather     | Target label (output)      |

---

## 🔍 How It Works

1. Data is loaded using **pandas**
2. Features and labels are separated
3. Data is split into training and testing sets
4. Model is trained using **Random Forest Algorithm**
5. Model is saved using **joblib**
6. Predictions are made on new input data

---

## 🔥 Future Improvements

* 🌐 Integrate real-time weather APIs
* 📈 Add data visualization dashboard
* 🤖 Improve accuracy using advanced ML/DL models
* 🖥️ Build a frontend (React / HTML / CSS)
* ☁️ Deploy as a web application

---

## 🤝 Contribution

Contributions are welcome!
Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

