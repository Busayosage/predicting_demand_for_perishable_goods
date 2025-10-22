import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Support both root-level CSVs and data/ folder CSVs.
DATA_DIRS = [PROJECT_ROOT / "data", PROJECT_ROOT]

def _find_csv(name: str) -> Optional[Path]:
    for d in DATA_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def _load_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        print(f"[warn] Missing file — skipping.")
        return None
    try:
        df = pd.read_csv(path)
        print(f"[ok] Loaded {path.name} with shape {df.shape}.")
        return df
    except Exception as e:
        print(f"[error] Failed to read {path}: {e}")
        return None


def _parse_week_date(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer existing datetime column if present
    for col in ("Week_Start_Date", "Date", "WeekStart", "week_start_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df["WeekDate"] = df[col]
            break
    else:
        # Fallback: try from 'Week_Number' like '2023-W05'
        if "Week_Number" in df.columns:
            # Try ISO week parse by appending '-1' for Monday
            s = df["Week_Number"].astype(str).str.strip() + "-1"
            dt = pd.to_datetime(s, format="%Y-W%W-%w", errors="coerce")
            # If many NaT, try alternative patterns
            if dt.isna().mean() > 0.5:
                # Try pandas' parser without strict format
                dt = pd.to_datetime(df["Week_Number"], errors="coerce")
            df["WeekDate"] = dt
        else:
            raise ValueError(
                "Could not infer weekly date. Provide 'Week_Start_Date' or 'Week_Number'."
            )

    if df["WeekDate"].isna().any():
        missing = int(df["WeekDate"].isna().sum())
        print(f"[warn] {missing} rows have invalid WeekDate and will be dropped.")
        df = df.loc[df["WeekDate"].notna()].copy()
    df["WeekDate"] = df["WeekDate"].dt.to_period("W").dt.start_time
    return df


def _safe_merge(left: pd.DataFrame, right: Optional[pd.DataFrame], on: str, how: str = "left") -> pd.DataFrame:
    if right is None:
        return left
    if on not in left.columns or on not in right.columns:
        print(f"[warn] Cannot merge on '{on}'; column missing. Skipping merge.")
        return left
    return left.merge(right, on=on, how=how)


def build_features(sales: pd.DataFrame,
                   product: Optional[pd.DataFrame],
                   store: Optional[pd.DataFrame],
                   weather: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = sales.copy()

    # Normalize essential columns
    rename_map = {
        "UnitsSold": "Units_Sold",
        "units_sold": "Units_Sold",
        "store_id": "Store_ID",
        "product_id": "Product_ID",
    }
    df.rename(columns=rename_map, inplace=True)

    # Date handling
    df = _parse_week_date(df)

    # Merge metadata
    df = _safe_merge(df, product, on="Product_ID")
    df = _safe_merge(df, store, on="Store_ID")

    # Optional: merge weather by Region + WeekDate if available
    if weather is not None:
        weather = weather.copy()
        # Try to harmonize date
        if "WeekDate" not in weather.columns:
            try:
                weather = _parse_week_date(weather)
            except Exception:
                pass
        if "Region" in df.columns and "Region" in weather.columns and "WeekDate" in weather.columns:
            df = df.merge(weather, on=["Region", "WeekDate"], how="left", suffixes=("", "_wx"))
        else:
            print("[warn] Weather merge skipped (needs Region and WeekDate).")

    # Sort for lag calculations
    keys = [c for c in ["Store_ID", "Product_ID"] if c in df.columns]
    if not keys:
        keys = ["Product_ID"] if "Product_ID" in df.columns else ["Store_ID"] if "Store_ID" in df.columns else []

    df = df.sort_values(keys + ["WeekDate"]) if keys else df.sort_values(["WeekDate"]) 

    # Target
    if "Units_Sold" not in df.columns:
        raise ValueError("Expected 'Units_Sold' column in weekly_sales.csv")

    # Lags and rolling features (avoid leakage)
    def add_group_lags(g: pd.DataFrame) -> pd.DataFrame:
        for lag in [1, 2, 4, 8]:
            g[f"lag_{lag}"] = g["Units_Sold"].shift(lag)
        g["roll_mean_4"] = g["Units_Sold"].shift(1).rolling(4).mean()
        g["roll_std_4"] = g["Units_Sold"].shift(1).rolling(4).std()
        g["roll_mean_8"] = g["Units_Sold"].shift(1).rolling(8).mean()
        return g

    if keys:
        try:
            df = df.groupby(keys, group_keys=False).apply(add_group_lags, include_groups=False)
        except TypeError:
            df = df.groupby(keys, group_keys=False).apply(add_group_lags)
    else:
        df = add_group_lags(df)

    # Calendar features
    df["weekofyear"] = df["WeekDate"].dt.isocalendar().week.astype(int)
    df["month"] = df["WeekDate"].dt.month
    df["quarter"] = df["WeekDate"].dt.quarter

    # Price/discount if present
    for col in ["Price", "Discount_Percent", "Marketing_Spend"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows where we cannot compute lags
    min_lag_rows = 8
    before = len(df)
    df = df.dropna(subset=[c for c in df.columns if c.startswith("lag_")])
    after = len(df)
    if before != after:
        print(f"[info] Dropped {before - after} rows without sufficient lag history.")

    return df


def make_feature_matrix(df: pd.DataFrame, target: str = "Units_Sold"):
    y = df[target].astype(float)
    # Exclude identifier and leakage columns
    drop_cols: List[str] = [
        target,
        "WeekDate",
        "Wastage_Units",
    ] + [c for c in ["Store_ID", "Product_ID", "Region"] if c in df.columns]

    # Drop raw week string/date identifiers to avoid one-hot drift
    drop_cols += [c for c in ["Week_Number", "Week_Start_Date", "Date", "WeekStart"] if c in df.columns]

    # Select basic numeric features
    X_num = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # One-hot reasonable low-cardinality categoricals
    cat_cols = [c for c in X_num.select_dtypes(include=["object", "category"]).columns if X_num[c].nunique() <= 50]
    X = pd.get_dummies(X_num, columns=cat_cols, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y


def temporal_split(df: pd.DataFrame, n_val_weeks: int = 8):
    weeks = np.sort(df["WeekDate"].unique())
    if len(weeks) <= n_val_weeks:
        raise ValueError("Not enough weeks to create a validation split.")
    cutoff = weeks[-n_val_weeks]
    train = df[df["WeekDate"] < cutoff]
    valid = df[df["WeekDate"] >= cutoff]
    print(f"[split] Train weeks: {train['WeekDate'].min().date()} -> {train['WeekDate'].max().date()} ({train.shape[0]} rows)")
    print(f"[split] Valid weeks: {valid['WeekDate'].min().date()} -> {valid['WeekDate'].max().date()} ({valid.shape[0]} rows)")
    return train, valid


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # R2 without sklearn
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def naive_last_week(valid: pd.DataFrame, keys: List[str]) -> np.ndarray:
    # For each key, use lag_1 if available; else group median; else global median
    if "lag_1" in valid.columns:
        pred = valid["lag_1"].copy()
    else:
        pred = pd.Series(np.nan, index=valid.index)

    if pred.isna().any():
        if keys and all(k in valid.columns for k in keys):
            med = valid.groupby(keys)["Units_Sold"].transform("median")
            pred = pred.fillna(med)
        pred = pred.fillna(valid["Units_Sold"].median())
    return pred.to_numpy()


def fit_model_rf(X_train: pd.DataFrame, y_train: pd.Series):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as e:
        print(f"[warn] sklearn not available ({e}). Using naive baseline only.")
        return None

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def main():
    # Load
    weekly_sales = _load_csv(_find_csv("weekly_sales.csv"))
    if weekly_sales is None:
        print("[fatal] weekly_sales.csv is required.")
        return
    product = _load_csv(_find_csv("product_details.csv"))  # optional
    store = _load_csv(_find_csv("store_info.csv"))  # optional
    weather = _load_csv(_find_csv("weather_data.csv"))  # optional

    # Build features
    df = build_features(weekly_sales, product, store, weather)

    # Split
    train_df, valid_df = temporal_split(df, n_val_weeks=8)
    keys = [k for k in ["Store_ID", "Product_ID"] if k in df.columns]

    # Feature matrix
    X_train, y_train = make_feature_matrix(train_df)
    X_valid, y_valid = make_feature_matrix(valid_df)

    # Align validation features to training columns to prevent feature-name mismatch
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0.0)

    # Train
    model = fit_model_rf(X_train, y_train)

    # Predict
    preds = {}
    preds["naive"] = naive_last_week(valid_df, keys)

    if model is not None:
        preds["rf"] = model.predict(X_valid)

    # Evaluate
    print("\n[results] Validation metrics:")
    for name, yhat in preds.items():
        m = metrics(y_valid, yhat)
        print(f"- {name:>6}: MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} R2={m['R2']:.3f}")

    # Persist predictions
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = valid_df[[c for c in ["Store_ID", "Product_ID", "WeekDate", "Units_Sold"] if c in valid_df.columns]].copy()
    for name, yhat in preds.items():
        out[f"pred_{name}"] = yhat
    out_path = out_dir / "validation_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"[ok] Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
