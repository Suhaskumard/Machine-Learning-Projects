import joblib
import numpy as np

# Load model
model = joblib.load("weather_model.pkl")

# Example input: [temperature, humidity, wind_speed, pressure]
sample = np.array([[29, 68, 9, 1012]])

prediction = model.predict(sample)

print("🌤️ Predicted Weather:", prediction[0])