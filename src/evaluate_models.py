import os
import logging
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Create folders
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# Logging configuration
logging.basicConfig(
    filename="logs/evaluate_models.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate_models():
    try:
        # Load datasets
        X_test_reg = pd.read_csv("data/engineered_x_test.csv")
        y_test_reg = pd.read_csv("data/engineered_y_test.csv")["price"]

        X_test_clf = pd.read_csv("data/engineered_x_test.csv")
        y_test_clf = pd.read_csv(
            "data/engineered_y_test_classification.csv"
        )["price_category"]

        # Load trained models
        regression_model = joblib.load(
            "models/best_regression.pkl"
        )

        classification_model = joblib.load(
            "models/best_classification.pkl"
        )

        label_encoder = joblib.load(
            "models/label_encoder.pkl"
        )

        logging.info("Models and datasets loaded successfully.")

        # Regression evaluation
        reg_predictions = regression_model.predict(X_test_reg)

        mae = mean_absolute_error(y_test_reg, reg_predictions)
        mse = mean_squared_error(y_test_reg, reg_predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test_reg, reg_predictions)

        regression_report = f"""
Regression Model Evaluation
---------------------------
MAE: {mae:.2f}
MSE: {mse:.2f}
RMSE: {rmse:.2f}
R²: {r2:.4f}
"""

        with open("outputs/regression_report.txt", "w") as f:
            f.write(regression_report)

        logging.info("Regression evaluation completed.")

        # Classification evaluation
        y_test_encoded = label_encoder.transform(y_test_clf)

        clf_predictions = classification_model.predict(X_test_clf)

        accuracy = accuracy_score(
            y_test_encoded,
            clf_predictions
        )

        precision = precision_score(
            y_test_encoded,
            clf_predictions,
            average="weighted"
        )

        recall = recall_score(
            y_test_encoded,
            clf_predictions,
            average="weighted"
        )

        f1 = f1_score(
            y_test_encoded,
            clf_predictions,
            average="weighted"
        )

        class_report = classification_report(
            y_test_encoded,
            clf_predictions
        )

        confusion = confusion_matrix(
            y_test_encoded,
            clf_predictions
        )

        classification_report_text = f"""
Classification Model Evaluation
-------------------------------
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}

Classification Report:
{class_report}

Confusion Matrix:
{confusion}
"""

        with open("outputs/classification_report.txt", "w") as f:
            f.write(classification_report_text)

        logging.info("Classification evaluation completed.")

        # Feature importance (if Random Forest)
        if hasattr(regression_model, "feature_importances_"):
            feature_importance = pd.DataFrame({
                "feature": X_test_reg.columns,
                "importance": regression_model.feature_importances_
            })

            feature_importance.sort_values(
                by="importance",
                ascending=False,
                inplace=True
            )

            feature_importance.to_csv(
                "outputs/feature_importance.csv",
                index=False
            )

            # Plot feature importance
            feature_importance.head(10).plot(
                x="feature",
                y="importance",
                kind="bar"
            )

            plt.title("Top 10 Feature Importances")
            plt.tight_layout()
            plt.savefig(
                "outputs/plots/feature_importance.png"
            )

            logging.info("Feature importance analysis completed.")

        # Cross-validation (Regression)
        X_train_reg = pd.read_csv("data/engineered_x_train.csv")
        y_train_reg = pd.read_csv(
            "data/engineered_y_train.csv"
        )["price"]

        cv_scores = cross_val_score(
            regression_model,
            X_train_reg,
            y_train_reg,
            cv=5,
            scoring="r2"
        )

        cv_report = f"""
Cross Validation Scores (R²):
{cv_scores}

Mean CV Score:
{cv_scores.mean():.4f}
"""

        with open("outputs/cross_validation_report.txt", "w") as f:
            f.write(cv_report)

        logging.info("Cross-validation completed.")

        print(regression_report)
        print(classification_report_text)
        print(cv_report)

        return {
            "regression_r2": r2,
            "classification_f1": f1,
            "cv_mean_r2": cv_scores.mean()
        }

    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        raise

if __name__ == "__main__":
    evaluate_models()