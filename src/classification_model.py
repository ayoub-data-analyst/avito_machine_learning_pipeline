import os
import logging
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create folders
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Logging configuration
logging.basicConfig(
    filename="logs/classification_model.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def classification_model():
    try:
        # Load engineered feature datasets
        X_train = pd.read_csv("data/engineered_x_train.csv")
        X_test = pd.read_csv("data/engineered_x_test.csv")

        # Load engineered classification targets
        y_train = pd.read_csv(
            "data/engineered_y_train_classification.csv"
        )

        y_test = pd.read_csv(
            "data/engineered_y_test_classification.csv"
        )

        logging.info("Engineered classification datasets loaded successfully.")

        # Build classification target
        target_column = "price_category"

        label_encoder = LabelEncoder()

        y_train_encoded = label_encoder.fit_transform(
            y_train[target_column]
        )

        y_test_encoded = label_encoder.transform(
            y_test[target_column]
        )

        logging.info("Target labels encoded successfully.")

        # Initialize models
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),

            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
        }

        best_model = None
        best_f1 = -999
        best_model_name = ""

        # Train and evaluate models
        for name, model in models.items():

            # Train model
            model.fit(X_train, y_train_encoded)

            # Predict on test set
            predictions = model.predict(X_test)

            # Evaluation metrics
            accuracy = accuracy_score(
                y_test_encoded,
                predictions
            )

            precision = precision_score(
                y_test_encoded,
                predictions,
                average="weighted"
            )

            recall = recall_score(
                y_test_encoded,
                predictions,
                average="weighted"
            )

            f1 = f1_score(
                y_test_encoded,
                predictions,
                average="weighted"
            )

            logging.info(
                f"{name} | Accuracy: {accuracy:.4f} | "
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | "
                f"F1-score: {f1:.4f}"
            )

            print(f"\n{name}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-score: {f1:.4f}")

            # Select best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = name

        # Save best classification model
        joblib.dump(
            best_model,
            "models/best_classification.pkl"
        )

        # Save label encoder
        joblib.dump(
            label_encoder,
            "models/label_encoder.pkl"
        )

        logging.info(
            f"Best classification model saved: "
            f"{best_model_name} with F1-score={best_f1:.4f}"
        )

        print(f"\nBest Model: {best_model_name}")
        print("Classification model saved successfully.")

        return best_model

    except Exception as e:
        logging.error(f"Classification modeling failed: {e}")
        raise

if __name__ == "__main__":
    classification_model()