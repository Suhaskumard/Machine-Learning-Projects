import pickle
import pandas as pd

# ✅ Load model FIRST
with open("models/model.pkl", "rb") as f:
    model, le_state, le_crop, le_season = pickle.load(f)

# ✅ Now safe to use encoders
print("Available states:", list(le_state.classes_)[:10])
print("Available crops:", list(le_crop.classes_)[:10])
print("Available seasons:", list(le_season.classes_)[:10])

# Input (use one from above list EXACTLY)
state = "Karnataka"
crop = "Rice"
season = "Kharif     "
area = 1000

# Clean input
state = state.strip().lower()
crop = crop.strip().lower()
season = season.strip().lower()

# Safe transform
def safe_transform(encoder, value, name):
    if value not in encoder.classes_:
        print(f"❌ Invalid {name}: {value}")
        print(f"Available {name}s:", list(encoder.classes_)[:10])
        exit()
    return encoder.transform([value])[0]

state_enc = safe_transform(le_state, state, "state")
crop_enc = safe_transform(le_crop, crop, "crop")
season_enc = safe_transform(le_season, season, "season")

# Create input DataFrame
features = pd.DataFrame([{
    "State_Name": state_enc,
    "Crop": crop_enc,
    "Season": season_enc,
    "Area": area
}])

# Predict
prediction = model.predict(features)

print(f"🌾 Predicted Yield: {prediction[0]:.2f}")