# fraud_detection.py

from sklearn.model_selection import train_test_split
from data_preprocessing import load_data, preprocess_data
from model_utils import train_model, evaluate_model
import config


def main():
    print("🔍 Loading Data...")
    data = load_data(config.DATA_PATH)

    print("⚙️ Preprocessing Data...")
    X, Y = preprocess_data(data)

    print("✂️ Splitting Data...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=config.TEST_SIZE,
        stratify=Y,
        random_state=config.RANDOM_STATE
    )

    print("🤖 Training Model...")
    model = train_model(X_train, Y_train, config.MODEL_MAX_ITER)

    print("📊 Evaluating Model...")
    metrics = evaluate_model(model, X_test, Y_test)

    print("\n🔥 Model Performance:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
