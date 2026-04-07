import pandas as pd
from preprocess import load_data, tokenize_data
from model import build_model

# Load data
train_df = load_data('dataset/train.csv')
test_df = load_data('dataset/test.csv')

# Extract features and labels
X_train_text = train_df['review']
y_train = train_df['sentiment']

X_test_text = test_df['review']
y_test = test_df['sentiment']

# Tokenize
X_train, X_test, tokenizer = tokenize_data(X_train_text, X_test_text)

# Build model
model = build_model()

# Train
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=32
)

# Save model
model.save("sentiment_model.h5")

print("✅ Model saved as sentiment_model.h5")
