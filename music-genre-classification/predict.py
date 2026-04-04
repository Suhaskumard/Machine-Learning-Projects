import librosa
import numpy as np
import joblib

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, mel])

# Load models
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# Predict
file_path = "test.wav"
features = extract_features(file_path)
features = scaler.transform([features])

prediction = model.predict(features)
genre = encoder.inverse_transform(prediction)

print("Predicted Genre:", genre[0])