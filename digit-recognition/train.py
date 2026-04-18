from pathlib import Path

from joblib import dump
from sklearn.metrics import accuracy_score, classification_report

from dataset import load_data
from model import build_model


MODEL_PATH = Path(__file__).with_name("digit_model.joblib")


def train():
    X_train, X_test, y_train, y_test, _ = load_data()

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    train()
