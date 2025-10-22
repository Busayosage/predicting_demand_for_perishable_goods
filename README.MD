Perishable Demand Forecasting — Quick Start

- Goal: forecast weekly `Units_Sold` for perishable goods using tabular features (price, discount, marketing, store/product meta, optional weather), with leakage-safe time splits.
- Repo contains EDA in `predicting_demand_for_perishable_goodipynb.ipynb` and a runnable baseline in `src/baseline_model.py`.

Project Structure

- `data/`
  - `weekly_sales.csv` — required. Expected columns include: `Store_ID`, `Product_ID`, `Units_Sold`, `Week_Number` (e.g., `2023-W05`) or `Week_Start_Date`. Optional: `Price`, `Discount_Percent`, `Marketing_Spend`, `Wastage_Units`.
  - `product_details.csv` — optional. Example: `Product_ID`, `Product_Category`, `Shelf_Life`.
  - `store_info.csv` — optional. Example: `Store_ID`, `Region`, `Store_Size`.
  - `supplier_info.csv` — optional.
  - `weather_data.csv` — optional. Example: `Region`, `Week_Number`/`Week_Start_Date`, weather features.
- `src/baseline_model.py` — feature engineering, time split, baseline model + naive predictor.
- `outputs/validation_predictions.csv` — generated predictions and holdout metrics.

Setup

1) Activate your environment

- Windows (PowerShell):
  - `venv\Scripts\Activate.ps1`
- macOS/Linux:
  - `source venv/bin/activate` (or create a new venv)

2) Install dependencies

- `pip install -r requirements.txt`
- Note: some optional libs (LightGBM/Prophet) may not support the very latest Python versions. If installs fail, consider Python 3.10–3.11.

Run the Baseline

- `python src/baseline_model.py`
- What it does:
  - Parses weekly date from `Week_Start_Date` or `Week_Number`.
  - Merges product/store (and weather by `Region` + week if available).
  - Builds leakage-safe features: lags (1,2,4,8), rolling means/std, calendar fields, numeric drivers (price/discount/marketing).
  - Splits last 8 weeks for validation; trains a RandomForest (if scikit-learn available) and a naive last-week baseline.
  - Prints MAE/RMSE/R2 and writes `outputs/validation_predictions.csv`.

Notebook Notes

- Replace Colab-style paths like `/content/weekly_sales.csv` with project-relative paths, e.g. `pd.read_csv('data/weekly_sales.csv')`.
- Ensure week parsing is robust:
  - Preferred: a true date column `Week_Start_Date`.
  - If using ISO week strings (e.g., `2023-W05`), build a Monday date with: `pd.to_datetime(week_str + '-1', format='%Y-W%W-%w')` and verify.
- Avoid leakage: compute lags/rolling with `.shift()` before `.rolling()`.

Next Steps

- Add rolling-origin backtests (TimeSeriesSplit) for stability.
- Try gradient boosting (LightGBM/XGBoost) and compare to RandomForest.
- Add wastage-aware metrics (e.g., higher weight on short shelf-life SKUs).
- Calibrate forecasts for promotions and inventory constraints.

Python Versions

- Python 3.11 (recommended for broad library support):
  - Create venv: `py -3.11 -m venv venv`
  - Activate: `venv\Scripts\Activate.ps1`
  - Install: `python -m pip install -r requirements-py311.txt`

- Python 3.13 (works for core libs; some ML libs may lag):
  - Install: `python -m pip install -r requirements.txt`
  - If `scikit-learn` fails to build, the baseline still runs the naive model. For full modeling, prefer Python 3.10–3.11.
