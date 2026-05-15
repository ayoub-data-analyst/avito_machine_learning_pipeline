import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create folders
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Logging configuration
logging.basicConfig(
    filename="logs/preprocess_data.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def preprocess_data():
    try:
        # Load extracted dataset
        df = pd.read_csv("data/obt_data.csv")

        logging.info(f"Data loaded successfully. Shape: {df.shape}")

        # Drop unnecessary columns
        df.drop(
            columns=["annonce_id", "title", "location", "link"],
            inplace=True
        )

        logging.info(
            f"Columns dropped successfully. Remaining columns: {df.columns.tolist()}"
        )

        # Separate targets
        target_regression = "price"
        target_classification = "price_category"

        X = df.drop(columns=[target_regression, target_classification])

        y_regression = df[target_regression]
        y_classification = df[target_classification]

        # Encode categorical variables
        X = pd.get_dummies(X, drop_first=True)

        logging.info("Categorical variables encoded successfully.")

        # Train/Test Split
        (
            X_train,
            X_test,
            y_train_reg,
            y_test_reg,
            y_train_clf,
            y_test_clf
        ) = train_test_split(
            X,
            y_regression,
            y_classification,
            test_size=0.2,
            random_state=42
        )

        logging.info(
            f"Data split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        )

        # Feature Scaling
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save processed feature datasets
        pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns
        ).to_csv("data/x_train.csv", index=False)

        pd.DataFrame(
            X_test_scaled,
            columns=X_test.columns
        ).to_csv("data/x_test.csv", index=False)

        # Save regression targets
        pd.DataFrame(
            y_train_reg,
            columns=[target_regression]
        ).to_csv("data/y_train.csv", index=False)

        pd.DataFrame(
            y_test_reg,
            columns=[target_regression]
        ).to_csv("data/y_test.csv", index=False)

        # Save classification targets
        pd.DataFrame(
            y_train_clf,
            columns=[target_classification]
        ).to_csv("data/y_train_classification.csv", index=False)

        pd.DataFrame(
            y_test_clf,
            columns=[target_classification]
        ).to_csv("data/y_test_classification.csv", index=False)

        logging.info("Preprocessing completed successfully.")

        return (
            X_train_scaled,
            X_test_scaled,
            y_train_reg,
            y_test_reg,
            y_train_clf,
            y_test_clf
        )

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    preprocess_data()