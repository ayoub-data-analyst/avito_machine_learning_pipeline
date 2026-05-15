import os
import logging
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Logging configuration
logging.basicConfig(
    filename="logs/regression_model.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def regression_model():
    try:
        # Load engineered datasets
        X_train = pd.read_csv("data/engineered_x_train.csv")
        X_test = pd.read_csv("data/engineered_x_test.csv")
        y_train = pd.read_csv("data/engineered_y_train.csv")["price"]
        y_test = pd.read_csv("data/engineered_y_test.csv")["price"]

        logging.info("Engineered datasets loaded successfully.")

        # Initialize models

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        }

        best_model = None
        best_r2 = -999
        best_model_name = ""

        # Train and evaluate each model

        for name, model in models.items():

            # Train model
            model.fit(X_train, y_train)

            # Predict on test set
            predictions = model.predict(X_test)

            # Evaluation metrics
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            logging.info(
                f"{name} | MAE: {mae:.2f} | MSE: {mse:.2f} | R²: {r2:.4f}"
            )

            print(f"\n{name}")
            print(f"MAE: {mae:.2f}")
            print(f"MSE: {mse:.2f}")
            print(f"R²: {r2:.4f}")

            # Select best model
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_model_name = name

        # Save best model

        joblib.dump(best_model, "models/best_regression.pkl")

        logging.info(
            f"Best regression model saved: {best_model_name} with R²={best_r2:.4f}"
        )

        print(f"\nBest Model: {best_model_name}")
        print("Model saved successfully.")

        return best_model

    except Exception as e:
        logging.error(f"Regression modeling failed: {e}")
        raise

if __name__ == "__main__":
    regression_model()