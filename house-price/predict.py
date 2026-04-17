import numpy as np
import joblib
from config import MODEL_PATH, FEATURES

def predict(area, bedrooms, age, location_score):
    """
    Predict house price given features.
    
    Args:
        area (float): House area in sqft
        bedrooms (int): Number of bedrooms
        age (float): Age of house in years
        location_score (float): Location quality score (1-10)
    
    Returns:
        float: Predicted price
    """
    model = joblib.load(MODEL_PATH)

    data = np.array([[area, bedrooms, age, location_score]])
    prediction = model.predict(data)[0]

    return prediction

if __name__ == "__main__":
    # Example prediction
    result = predict(3200, 4, 2, 9)
    print(f"Predicted house price: ${result:,.2f}")
