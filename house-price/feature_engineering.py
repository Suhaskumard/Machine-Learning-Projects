def add_features(df):
    df = df.copy()
    
    # Advanced feature engineering
    df['Price_per_sqft'] = df['Price'] / df['Area']
    df['Area_per_bedroom'] = df['Area'] / df['Bedrooms']
    df['Age_factor'] = df['Age'] ** 2
    
    return df
