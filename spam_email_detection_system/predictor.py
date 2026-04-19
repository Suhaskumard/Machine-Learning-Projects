import pickle
from preprocess import clean_text

# Load trained model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_email(text):
    text = clean_text(text)
    prediction = model.predict([text])[0]
    
    return "🚨 Spam" if prediction == 1 else "✅ Not Spam"
