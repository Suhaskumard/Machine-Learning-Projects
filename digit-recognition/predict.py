from pathlib import Path

from joblib import load
import numpy as np
from PIL import Image

from dataset import load_data


MODEL_PATH = Path(__file__).with_name("digit_model.joblib")
PLOT_PATH = Path(__file__).with_name("prediction_sample.png")


def predict(sample_index=0):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run train.py first."
        )

    model = load(MODEL_PATH)
    _, X_test, _, y_test, _ = load_data()

    if not 0 <= sample_index < len(X_test):
        raise IndexError(
            f"sample_index must be between 0 and {len(X_test) - 1}, got {sample_index}."
        )

    sample = X_test[sample_index]
    prediction = model.predict(sample.reshape(1, -1))[0]
    actual = y_test[sample_index]

    image_array = np.uint8(sample.reshape(8, 8) * 255)
    image = Image.fromarray(image_array, mode="L").resize((160, 160), Image.NEAREST)
    image.save(PLOT_PATH)

    print(f"Predicted digit: {prediction}")
    print(f"Actual digit: {actual}")
    print(f"Saved preview to: {PLOT_PATH}")


if __name__ == "__main__":
    predict()
