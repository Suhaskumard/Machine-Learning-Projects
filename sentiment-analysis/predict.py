import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text
import pickle

# Load model
model = load_model("sentiment_model.h5")

# NOTE: In real projects, save tokenizer after training
# Here we assume tokenizer is recreated or saved separately

def predict_review(text, tokenizer, max_len=200):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        print("😊 Positive Review")
    else:
        print("😠 Negative Review")
