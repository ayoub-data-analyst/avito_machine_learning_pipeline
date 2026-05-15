# Avito Real Estate — Machine Learning Pipeline

> **Part 3 of the Avito End-to-End Data Project**
> Predicts real estate prices and classifies listings by price category using data extracted from [avito-end-to-end-data-pipeline](https://github.com/ayoub-data-analyst/avito-end-to-end-data-pipeline) and visualized in [avito_report_power_bi](https://github.com/ayoub-data-analyst/avito_report_power_bi).

---

## Project Overview

This project builds a complete machine learning pipeline on top of Moroccan real estate listings scraped from **Avito.ma**. The pipeline covers everything from raw data extraction out of PostgreSQL, through preprocessing and feature engineering, to training, evaluating, and saving regression and classification models.

| Task | Target | Best Model | Score |
|---|---|---|---|
| **Regression** | `price` (MAD) | Random Forest | R² = **0.9967** |
| **Classification** | `price_category` (Low / Medium / High) | Random Forest | F1 = **0.9942** |

---

## Project Structure

```
avito_machine_learning_pipeline/
│
├── src/
│   ├── main.py                  # Pipeline orchestrator
│   ├── extract_data.py          # Pulls OBT from PostgreSQL
│   ├── preprocess_data.py       # Cleans, encodes, scales, splits
│   ├── feature_engineering.py  # Creates interaction & ratio features
│   ├── regression_model.py      # Trains & selects best regressor
│   ├── classification_model.py  # Trains & selects best classifier
│   └── evaluate_models.py       # Final evaluation, CV, feature importance
│
├── data/                        # Intermediate CSVs (generated at runtime)
├── models/                      # Saved .pkl model files
├── outputs/
│   ├── regression_report.txt
│   ├── classification_report.txt
│   ├── cross_validation_report.txt
│   ├── feature_importance.csv
│   └── plots/feature_importance.png
│
├── logs/                        # Per-step log files
├── notebooks/ml_analysis.ipynb  # Exploratory analysis
├── .env                         # DB credentials (not committed)
├── .gitignore
└── requirements.txt
```

---

## Pipeline Steps

The pipeline runs sequentially via `src/main.py`:

```
STEP 1 — Data Extraction
STEP 2 — Data Preprocessing
STEP 3 — Feature Engineering
STEP 4 — Regression Modeling
STEP 5 — Classification Modeling
STEP 6 — Model Evaluation
```

### Step 1 · Data Extraction
Connects to a local PostgreSQL database (`avito_db`) and pulls the One Big Table `ml_schema.obt_avito_annonce` into `data/obt_data.csv`.

### Step 2 · Preprocessing
- Drops non-predictive columns: `annonce_id`, `title`, `location`, `link`
- One-hot encodes categorical variables
- Splits into train/test (80/20, `random_state=42`)
- Applies `StandardScaler` on features
- Saves separate CSVs for regression and classification targets

### Step 3 · Feature Engineering
Creates domain-specific features from the real estate context:

| Feature | Formula |
|---|---|
| `surface_per_room` | surface / (rooms + 1) |
| `bath_room_ratio` | baths / (rooms + 1) |
| `surface_rooms_interaction` | surface × rooms |
| `surface_baths_interaction` | surface × baths |

Also applies `log1p` transformation on the regression target (`price`) to reduce skewness.

### Step 4 · Regression Modeling
Trains and compares:
- `LinearRegression`
- `RandomForestRegressor` (100 estimators)

Best model selected by **R²** and saved to `models/best_regression.pkl`.

### Step 5 · Classification Modeling
Trains and compares:
- `LogisticRegression` (balanced class weights)
- `RandomForestClassifier` (100 estimators, balanced class weights)

Best model selected by **weighted F1-score** and saved to `models/best_classification.pkl`. Label encoder saved as `models/label_encoder.pkl`.

### Step 6 · Evaluation
- Full regression metrics: MAE, MSE, RMSE, R²
- Full classification metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
- 5-fold cross-validation on the regression model
- Feature importance chart (top 10 features) saved to `outputs/plots/`

---

## Results

### Regression (Price Prediction)

| Metric | Value |
|---|---|
| MAE | 160,257 MAD |
| RMSE | 2,049,418 MAD |
| R² | **0.9967** |
| CV Mean R² (5-fold) | **0.9950** |

### Classification (Price Category)

| Metric | Value |
|---|---|
| Accuracy | **99.42%** |
| Precision | **99.42%** |
| Recall | **99.42%** |
| F1-score | **99.42%** |

---

## Getting Started

### Prerequisites
- Python 3.12+
- PostgreSQL running locally with `avito_db` populated (see [avito-end-to-end-data-pipeline](https://github.com/ayoub-data-analyst/avito-end-to-end-data-pipeline))

### Installation

```bash
git clone https://github.com/ayoub-data-analyst/avito_machine_learning_pipeline.git
cd avito_machine_learning_pipeline

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file at the project root:

```env
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=avito_db
```

### Run the Pipeline

```bash
cd src
python main.py
```

Each step prints progress to the console and writes detailed logs to the `logs/` folder.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data | pandas, numpy, SQLAlchemy, psycopg2 |
| ML | scikit-learn (RandomForest, LogisticRegression, StandardScaler, GridSearchCV) |
| Serialization | joblib |
| Visualization | matplotlib |
| Environment | python-dotenv |

---

## Related Projects

This project is part of a larger end-to-end data workflow:

| Project | Description |
|---|---|
| [avito-end-to-end-data-pipeline](https://github.com/ayoub-data-analyst/avito-end-to-end-data-pipeline) | Scraping → cleaning → PostgreSQL (star schema + OBT) |
| [avito_report_power_bi](https://github.com/ayoub-data-analyst/avito_report_power_bi) | Power BI dashboard on top of the PostgreSQL data |
| **avito_machine_learning_pipeline** *(this repo)* | ML models for price prediction & classification |

---

## Author

**Ayoub** — [@ayoub-data-analyst](https://github.com/ayoub-data-analyst)