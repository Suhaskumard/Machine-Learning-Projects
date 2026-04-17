import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from config import *
from data_loader import load_data
from feature_engineering import add_features
from utils import setup_logger, log

def train():
    logger = setup_logger()
    logger.info("Loading data...")

    df = load_data()
    df = add_features(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ('model', RandomForestRegressor(random_state=RANDOM_STATE))
    ])

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10]
    }

    logger.info("Training model...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    logger.info(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    logger.info(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    joblib.dump(best_model, MODEL_PATH)
    logger.info("Model saved to house_price_model.pkl!")

if __name__ == "__main__":
    train()
