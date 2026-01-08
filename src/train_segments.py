import os
import pandas as pd
from joblib import dump

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


IN_PATH = "C:/Project S/CustomerSegemetation/Data/processed/customer_features.csv"
OUT_DIR = "C:/Project S/CustomerSegemetation/Data/processed"
OUT_SEG = os.path.join(OUT_DIR, "customer_segments.csv")
OUT_MODEL = os.path.join(OUT_DIR, "segmentation_model.joblib")


def choose_k(X, k_min=2, k_max=8, random_state=42):
    best_k = None
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(IN_PATH)

    customer_col = "cc_num"
    feature_cols = [c for c in df.columns if c != customer_col]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    best_k, best_score = choose_k(X, 2, 8)
    print(f"Chosen k: {best_k}, silhouette: {best_score:.3f}")

    model = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    segment = model.fit_predict(X)

    out = df[[customer_col]].copy()
    out["segment"] = segment
    out.to_csv(OUT_SEG, index=False)

    dump(
        {
            "customer_col": customer_col,
            "feature_cols": feature_cols,
            "scaler": scaler,
            "model": model,
        },
        OUT_MODEL,
    )

    print(f"Saved: {OUT_SEG}")
    print(f"Saved: {OUT_MODEL}")


if __name__ == "__main__":
    main()
