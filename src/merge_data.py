from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
        print(f"[ok] Loaded {path.name} {df.shape}")
        return df
    except Exception as e:
        print(f"[error] Failed to read {path}: {e}")
        return None


def _parse_week_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("Week_Start_Date", "Date", "WeekStart", "week_start_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df["WeekDate"] = df[col]
            break
    else:
        if "Week_Number" in df.columns:
            s = df["Week_Number"].astype(str).str.strip() + "-1"
            dt = pd.to_datetime(s, format="%Y-W%W-%w", errors="coerce")
            if dt.isna().mean() > 0.5:
                dt = pd.to_datetime(df["Week_Number"], errors="coerce")
            df["WeekDate"] = dt
        else:
            # If nothing workable, leave as-is.
            df["WeekDate"] = pd.NaT

    if df["WeekDate"].notna().any():
        df["WeekDate"] = df["WeekDate"].dt.to_period("W").dt.start_time
    return df


def _safe_merge(left: pd.DataFrame, right: Optional[pd.DataFrame], on, how: str = "left") -> pd.DataFrame:
    if right is None:
        return left
    if isinstance(on, (list, tuple)):
        missing = [c for c in on if c not in left.columns or c not in right.columns]
        if missing:
            print(f"[warn] Cannot merge on {on}; missing {missing}. Skipping.")
            return left
    else:
        if on not in left.columns or on not in right.columns:
            print(f"[warn] Cannot merge on '{on}'; column missing. Skipping.")
            return left
    return left.merge(right, on=on, how=how)


def main():
    sales = _load_csv(_find_csv("weekly_sales.csv"))
    if sales is None:
        print("[fatal] weekly_sales.csv is required (put it under data/).")
        return 1
    product = _load_csv(_find_csv("product_details.csv"))
    store = _load_csv(_find_csv("store_info.csv"))
    supplier = _load_csv(_find_csv("supplier_info.csv"))
    weather = _load_csv(_find_csv("weather_data.csv"))

    # Normalize expected names
    sales = sales.rename(
        columns={
            "units_sold": "Units_Sold",
            "store_id": "Store_ID",
            "product_id": "Product_ID",
            "supplier_id": "Supplier_ID",
        }
    )

    sales = _parse_week_date(sales)
    if weather is not None:
        weather = _parse_week_date(weather)

    merged = sales.copy()
    merged = _safe_merge(merged, product, on="Product_ID")
    merged = _safe_merge(merged, store, on="Store_ID")

    # Try supplier merge: prefer Supplier_ID if present, else try Product_ID if supplier file has it
    if supplier is not None:
        if "Supplier_ID" in merged.columns and "Supplier_ID" in supplier.columns:
            merged = _safe_merge(merged, supplier, on="Supplier_ID")
        elif "Product_ID" in merged.columns and "Product_ID" in supplier.columns:
            merged = _safe_merge(merged, supplier, on="Product_ID")
        else:
            print("[warn] Supplier merge skipped (no matching key).")

    # Weather merge on Region + WeekDate if available
    if weather is not None:
        if "Region" in merged.columns and "Region" in weather.columns and "WeekDate" in weather.columns:
            merged = _safe_merge(merged, weather, on=["Region", "WeekDate"])
        else:
            print("[warn] Weather merge skipped (needs Region and WeekDate).")

    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "merged_dataset.csv"
    merged.to_csv(out_path, index=False)
    print(f"[ok] Wrote {out_path} with shape {merged.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

