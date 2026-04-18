# Digit Recognition

This project trains a digit recognizer with `scikit-learn` on the built-in handwritten digits dataset and saves the trained model for reuse.

## Project Structure

```text
digit-recognition/
|-- dataset.py
|-- model.py
|-- train.py
|-- predict.py
|-- requirements.txt
`-- README.md
```

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train.py
   ```

3. Run a prediction example:
   ```bash
   python predict.py
   ```

The prediction script saves a preview image as `prediction_sample.png`.
