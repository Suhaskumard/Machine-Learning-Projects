import pickle

# Load model
with open("model/classifier.pkl", "rb") as f:
    model = pickle.load(f)

def predict_category(text):
    return model.predict([text])[0]

if __name__ == "__main__":
    print("📄 Resume Classification System")
    print("-" * 40)

    user_input = input("Enter resume text:\n")
    prediction = predict_category(user_input)

    print(f"\n🎯 Predicted Category: {prediction}")
