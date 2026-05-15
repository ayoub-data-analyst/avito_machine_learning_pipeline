import os
import logging

from extract_data import extract_obt
from preprocess_data import preprocess_data
from feature_engineering import feature_engineering
from regression_model import regression_model
from classification_model import classification_model
from evaluate_models import evaluate_models

# Create logs folder
os.makedirs("logs", exist_ok=True)

# Logging configuration
logging.basicConfig(
    filename="logs/main_pipeline.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_pipeline():
    try:
        print("Starting Real Estate ML Pipeline...")
        logging.info("Pipeline execution started.")

        # Step 1: Data Extraction
        print("\nSTEP 1 — Data Extraction")
        extract_obt()
        logging.info("STEP 1 completed successfully.")

        # Step 2: Data Preprocessing
        print("\nSTEP 2 — Data Preprocessing")
        preprocess_data()
        logging.info("STEP 2 completed successfully.")

        # Step 3: Feature Engineering
        print("\nSTEP 3 — Feature Engineering")
        feature_engineering()
        logging.info("STEP 3 completed successfully.")

        # Step 4: Regression Modeling
        print("\nSTEP 4 — Regression Modeling")
        regression_model()
        logging.info("STEP 4 completed successfully.")

        # Step 5: Classification Modeling
        print("\nSTEP 5 — Classification Modeling")
        classification_model()
        logging.info("STEP 5 completed successfully.")

        # Step 6: Evaluation & Optimization
        print("\nSTEP 6 — Model Evaluation")
        results = evaluate_models()
        logging.info("STEP 6 completed successfully.")

        print("\nPipeline completed successfully.")
        print("\nFinal Results:")
        print(results)

        logging.info(f"Pipeline completed successfully. Results: {results}")

        return results

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        print(f"\nPipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()