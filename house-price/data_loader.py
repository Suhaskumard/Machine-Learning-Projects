import pandas as pd

def load_data():
    # Replace with real dataset
    data = {
        'Area': [1000, 1500, 2000, 2500, 3000, 3500, 4000],
        'Bedrooms': [2, 3, 3, 4, 4, 5, 5],
        'Age': [10, 5, 8, 2, 1, 3, 0],
        'Location_Score': [5, 7, 6, 8, 9, 7, 10],
        'Price': [300000, 450000, 500000, 600000, 650000, 700000, 800000]
    }
    return pd.DataFrame(data)
