import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from data_loader import load_data
from feature_engineering import add_features
from config import *

def evaluate():
    df = load_data()
    df = add_features(df)

    X = df[FEATURES]
    y = df[TARGET]

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    print("Evaluation Results on full dataset:")
    print(f"MAE: ${mae:,.2f}")
    print(f"R² Score: {r2:.4f}")

if __name__ == "__main__":
    evaluate()
