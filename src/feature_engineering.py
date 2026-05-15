import os
import logging
import pandas as pd
import numpy as np

# Create folders
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Logging configuration
logging.basicConfig(
    filename="logs/feature_engineering.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def feature_engineering():
    try:
        # Load preprocessed datasets
        X_train = pd.read_csv("data/x_train.csv")
        X_test = pd.read_csv("data/x_test.csv")

        y_train = pd.read_csv("data/y_train.csv")
        y_test = pd.read_csv("data/y_test.csv")

        y_train_clf = pd.read_csv("data/y_train_classification.csv")
        y_test_clf = pd.read_csv("data/y_test_classification.csv")

        logging.info("Preprocessed datasets loaded successfully.")

        # Log transformation for regression target
        y_train["log_price"] = np.log1p(y_train["price"])
        y_test["log_price"] = np.log1p(y_test["price"])

        logging.info("Log transformation applied to regression targets.")

        # Surface per room ratio
        if "surface" in X_train.columns and "rooms" in X_train.columns:
            X_train["surface_per_room"] = X_train["surface"] / (X_train["rooms"] + 1)
            X_test["surface_per_room"] = X_test["surface"] / (X_test["rooms"] + 1)

        # Bath per room ratio
        if "baths" in X_train.columns and "rooms" in X_train.columns:
            X_train["bath_room_ratio"] = X_train["baths"] / (X_train["rooms"] + 1)
            X_test["bath_room_ratio"] = X_test["baths"] / (X_test["rooms"] + 1)

        # Surface × rooms interaction
        if "surface" in X_train.columns and "rooms" in X_train.columns:
            X_train["surface_rooms_interaction"] = X_train["surface"] * X_train["rooms"]
            X_test["surface_rooms_interaction"] = X_test["surface"] * X_test["rooms"]

        # Surface × baths interaction
        if "surface" in X_train.columns and "baths" in X_train.columns:
            X_train["surface_baths_interaction"] = X_train["surface"] * X_train["baths"]
            X_test["surface_baths_interaction"] = X_test["surface"] * X_test["baths"]

        logging.info("Advanced feature engineering completed successfully.")

        # Save engineered feature datasets
        X_train.to_csv("data/engineered_x_train.csv", index=False)
        X_test.to_csv("data/engineered_x_test.csv", index=False)

        # Save engineered regression targets
        y_train.to_csv("data/engineered_y_train.csv", index=False)
        y_test.to_csv("data/engineered_y_test.csv", index=False)

        # Save classification targets
        y_train_clf.to_csv(
            "data/engineered_y_train_classification.csv",
            index=False
        )

        y_test_clf.to_csv(
            "data/engineered_y_test_classification.csv",
            index=False
        )

        logging.info("All engineered datasets saved successfully.")

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            y_train_clf,
            y_test_clf
        )

    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    feature_engineering()