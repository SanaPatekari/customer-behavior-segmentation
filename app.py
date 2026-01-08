import pandas as pd
import streamlit as st

from src.plots import fig_segment_distribution, fig_top_categories, fig_monthly_spend


RAW_PATH = "C:/Project S/CustomerSegemetation/Data/credit_card_transactions.csv"
FEAT_PATH = "C:/Project S/CustomerSegemetation/Data/processed/customer_features.csv"
SEG_PATH = "C:/Project S/CustomerSegemetation/Data/processed/customer_segments.csv"


st.set_page_config(page_title="Customer Segmentation Explorer", layout="wide")
st.title("Customer Segmentation Explorer")


@st.cache_data
def load_data():
    raw = pd.read_csv(RAW_PATH)
    feats = pd.read_csv(FEAT_PATH)
    seg = pd.read_csv(SEG_PATH)
    return raw, feats, seg


raw, feats, seg = load_data()

raw["trans_dt"] = pd.to_datetime(raw["trans_date_trans_time"], errors="coerce")
raw = raw.dropna(subset=["trans_dt"])
raw["month"] = raw["trans_dt"].dt.to_period("M").astype(str)

seg_map = seg.set_index("cc_num")["segment"].to_dict()
raw["segment"] = raw["cc_num"].map(seg_map)

min_date = raw["trans_dt"].min().date()
max_date = raw["trans_dt"].max().date()

tab1, tab2, tab3 = st.tabs(["Portfolio overview", "Segment insights", "Customer explorer"])


with tab1:
    st.subheader("Portfolio overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{seg['cc_num'].nunique():,}")
    c2.metric("Transactions", f"{len(raw):,}")
    c3.metric("Segments", f"{seg['segment'].nunique():,}")
    c4.metric("Date range", f"{min_date} to {max_date}")

    segment_counts = seg["segment"].value_counts().sort_index()
    st.pyplot(fig_segment_distribution(segment_counts))

    st.subheader("Segment summary")
    merged = feats.merge(seg, on="cc_num", how="inner")

    summary = merged.groupby("segment").agg(
        customers=("cc_num", "nunique"),
        avg_txn_count=("txn_count", "mean"),
        avg_total_spend=("total_spend", "mean"),
        avg_amt=("avg_amt", "mean"),
        avg_volatility=("std_amt", "mean"),
        avg_unique_categories=("unique_categories", "mean"),
        avg_unique_merchants=("unique_merchants", "mean"),
        avg_weekend_share=("weekend_share", "mean"),
        avg_night_share=("night_share", "mean"),
        avg_distance_km=("avg_distance_km", "mean"),
    ).reset_index()

    st.dataframe(summary.round(3))


with tab2:
    st.subheader("Segment insights")

    merged = feats.merge(seg, on="cc_num", how="inner")
    chosen_segment = st.selectbox("Select segment", options=sorted(seg["segment"].unique().tolist()))

    seg_df = merged[merged["segment"] == chosen_segment].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers in segment", f"{seg_df['cc_num'].nunique():,}")
    c2.metric("Avg txn count", f"{seg_df['txn_count'].mean():.2f}")
    c3.metric("Avg total spend", f"{seg_df['total_spend'].mean():.2f}")
    c4.metric("Avg amount volatility", f"{seg_df['std_amt'].mean():.2f}")

    st.subheader("Typical behavior (based on averages)")
    st.write(
        "This view helps interpret what makes this segment different."
    )

    show_cols = [
        "txn_count",
        "total_spend",
        "avg_amt",
        "std_amt",
        "unique_merchants",
        "unique_categories",
        "weekend_share",
        "night_share",
        "avg_distance_km",
        "max_distance_km",
        "txn_per_day",
        "spend_per_day",
    ]
    st.dataframe(seg_df[show_cols].describe().loc[["mean", "50%", "std"]].round(3))


with tab3:
    st.subheader("Customer explorer")

    segment_filter = st.multiselect(
        "Filter customers by segment",
        options=sorted(seg["segment"].unique().tolist()),
        default=sorted(seg["segment"].unique().tolist()),
    )

    eligible_customers = seg[seg["segment"].isin(segment_filter)]["cc_num"].tolist()

    selected_customer = st.selectbox("Select customer (cc_num)", options=eligible_customers)

    cust_seg = int(seg.loc[seg["cc_num"] == selected_customer, "segment"].iloc[0])
    cust_feats = feats[feats["cc_num"] == selected_customer]

    if cust_feats.empty:
        st.warning("No features found for this customer. Build features first.")
        st.stop()

    row = cust_feats.iloc[0].to_dict()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Segment", str(cust_seg))
    m2.metric("Transactions", f"{int(row['txn_count']):,}")
    m3.metric("Total spend", f"{row['total_spend']:.2f}")
    m4.metric("Avg amount", f"{row['avg_amt']:.2f}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Amt volatility (std)", f"{row['std_amt']:.2f}")
    m6.metric("Unique merchants", f"{int(row['unique_merchants']):,}")
    m7.metric("Unique categories", f"{int(row['unique_categories']):,}")
    m8.metric("Txn per day", f"{row['txn_per_day']:.3f}")

    m9, m10, m11, m12 = st.columns(4)
    m9.metric("Weekend share", f"{row['weekend_share']:.2%}")
    m10.metric("Night share", f"{row['night_share']:.2%}")
    m11.metric("Avg distance (km)", f"{row['avg_distance_km']:.2f}")
    m12.metric("Max distance (km)", f"{row['max_distance_km']:.2f}")

    cust_raw = raw[raw["cc_num"] == selected_customer].copy()

    left, right = st.columns(2)

    with left:
        monthly_spend = cust_raw.groupby("month")["amt"].sum().sort_index()
        st.pyplot(fig_monthly_spend(monthly_spend))

    with right:
        top_cats = cust_raw["category"].value_counts()
        st.pyplot(fig_top_categories(top_cats, top_n=10))

    st.subheader("Sample transactions (safe columns only)")
    safe_cols = ["trans_date_trans_time", "merchant", "category", "amt", "segment"]
    st.dataframe(cust_raw[safe_cols].sort_values("trans_date_trans_time", ascending=False).head(25))
