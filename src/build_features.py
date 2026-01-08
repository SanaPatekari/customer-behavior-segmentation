import os
import pandas as pd
import numpy as np


RAW_PATH = "C:/Project S/CustomerSegemetation/Data/credit_card_transactions.csv"
OUT_DIR = "C:/Project S/CustomerSegemetation/Data/processed"
OUT_PATH = os.path.join(OUT_DIR, "customer_features.csv")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (np.sin(dlat / 2.0) ** 2) + (np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2))
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df = df.dropna(subset=["trans_dt"])

    df["amt"] = pd.to_numeric(df["amt"], errors="coerce")
    df = df.dropna(subset=["amt"])

    df["hour"] = df["trans_dt"].dt.hour
    df["dayofweek"] = df["trans_dt"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 22)).astype(int)

    df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    customer_col = "cc_num"

    agg = df.groupby(customer_col).agg(
        txn_count=("amt", "count"),
        total_spend=("amt", "sum"),
        avg_amt=("amt", "mean"),
        std_amt=("amt", "std"),
        max_amt=("amt", "max"),
        unique_merchants=("merchant", "nunique"),
        unique_categories=("category", "nunique"),
        weekend_share=("is_weekend", "mean"),
        night_share=("is_night", "mean"),
        avg_distance_km=("distance_km", "mean"),
        max_distance_km=("distance_km", "max"),
        avg_city_pop=("city_pop", "mean"),
        first_txn=("trans_dt", "min"),
        last_txn=("trans_dt", "max"),
    ).reset_index()

    agg["std_amt"] = agg["std_amt"].fillna(0.0)
    agg["avg_city_pop"] = agg["avg_city_pop"].fillna(agg["avg_city_pop"].median())

    agg["active_days"] = (agg["last_txn"] - agg["first_txn"]).dt.days.clip(lower=0)
    agg["txn_per_day"] = agg["txn_count"] / (agg["active_days"] + 1)
    agg["spend_per_day"] = agg["total_spend"] / (agg["active_days"] + 1)

    agg = agg.drop(columns=["first_txn", "last_txn"])

    return agg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    features = build_customer_features(df)

    features.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")
    print(f"Customers: {len(features):,}")
    print("Columns:", list(features.columns))


if __name__ == "__main__":
    main()
