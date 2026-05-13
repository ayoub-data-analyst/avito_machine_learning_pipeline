import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create folders
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Logging
logging.basicConfig(
    filename="logs/preprocess_data.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def preprocess_data():
    try:
        # Load extracted data
        df = pd.read_csv("data/obt_data.csv")

        logging.info(f"Data loaded successfully. Shape: {df.shape}")

        # Drop unnecessary columns
        df.drop(columns=["annonce_id", "title", "location", "link"], inplace=True)

        logging.info(f"Columns dropped successfully. Remaining columns: {df.columns.tolist()}")

        # Separate target variable
        target = "price"

        X = df.drop(columns=[target])
        y = df[target]
    
        # Encode categorical variables
        X = pd.get_dummies(X, drop_first=True)

        logging.info("Categorical variables encoded successfully.")

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        logging.info(
            f"Data split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        )

        # Scaling numerical features
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save processed datasets
        pd.DataFrame(X_train_scaled).to_csv("data/train.csv", index=False)
        pd.DataFrame(X_test_scaled).to_csv("data/test.csv", index=False)

        pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
        pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)

        logging.info("Preprocessing completed successfully.")

        return X_train_scaled, X_test_scaled, y_train, y_test

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    preprocess_data()