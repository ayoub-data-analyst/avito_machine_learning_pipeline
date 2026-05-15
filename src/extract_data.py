import os
import logging
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Create folders if not exist
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Logging configuration
logging.basicConfig(
    filename="logs/extract_data.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S"
)

logging.info("Start connection with avito_db...")

# Load environment variables
load_dotenv()


def get_engine():
    try:
        engine = create_engine(
            f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        )

        logging.info("Database connection created successfully.")
        return engine

    except Exception as e:
        logging.error(f"Error creating database connection: {e}")
        raise


def extract_obt():
    try:
        engine = get_engine()

        query = "SELECT * FROM ml_schema.obt_avito_annonce"

        df = pd.read_sql(query, engine)

        df.to_csv("data/obt_data.csv", index=False)

        logging.info(f"OBT extracted successfully. Shape: {df.shape}")

        return df

    except Exception as e:
        logging.error(f"Error extracting OBT data: {e}")
        raise

if __name__ == "__main__":
    extract_obt()