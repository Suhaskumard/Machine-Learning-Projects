import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 📂 Dataset path (GTZAN format)
DATASET_PATH = "genres"

# 🎵 Extract features from audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    
    return np.hstack([mfcc, chroma, mel])

# 📊 Load dataset
def load_data(dataset_path):
    features = []
    labels = []
    
    for genre in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre)
        
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            
            try:
                data = extract_features(file_path)
                features.append(data)
                labels.append(genre)
            except:
                print(f"Error processing {file_path}")
    
    return np.array(features), np.array(labels)

# 🚀 Main pipeline
def main():
    print("Loading dataset...")
    X, y = load_data(DATASET_PATH)

    print("Encoding labels...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred))

    # 💾 Save model
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "encoder.pkl")

    print("Model saved!")

if __name__ == "__main__":
    main()