"""
Streamlit Dashboard untuk Supply Chain ML Forecasting & Optimizer

Dashboard ini menampilkan:
- Ringkasan metrik evaluasi (WAPE, MASE)
- Visualisasi forecast vs actual
- Tabel evaluasi per SKU-location
- Detail forecast untuk SKU-location tertentu
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Config
st.set_page_config(
    page_title="Supply Chain ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
PRED_PATH = Path("data/processed/predictions.csv")
FEAT_PATH = Path("data/processed/weekly_features.parquet")
STATS_PATH = Path("models/artifacts/demand_stats.json")


@st.cache_data
def load_predictions():
    """Load predictions.csv jika ada."""
    if PRED_PATH.exists():
        return pd.read_csv(PRED_PATH)
    return None


@st.cache_data
def load_features():
    """Load weekly features jika ada."""
    if FEAT_PATH.exists():
        return pd.read_parquet(FEAT_PATH)
    return None


@st.cache_data
def load_stats():
    """Load demand stats jika ada."""
    if STATS_PATH.exists():
        return json.loads(STATS_PATH.read_text())
    return None


def main():
    st.title("📊 Supply Chain ML Forecasting Dashboard")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        show_details = st.checkbox("Tampilkan Detail per SKU-Location", value=False)
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # Load data
    pred_df = load_predictions()
    feat_df = load_features()
    stats = load_stats()

    if pred_df is None:
        st.warning(
            "⚠️ File `data/processed/predictions.csv` belum ada. "
            "Jalankan dulu: `python -m src.forecasting.evaluate`"
        )
        st.info("💡 Atau jalankan pipeline lengkap:\n```bash\npython etl/generate_dummy.py\npython etl/build_features.py\npython -m src.forecasting.train\npython -m src.forecasting.evaluate\n```")
        return

    # --- Summary Metrics ---
    st.header("📈 Summary Metrics")

    col1, col2, col3, col4 = st.columns(4)

    # Global WAPE
    naive_wape = ((pred_df["y_true"] - pred_df["y_pred_naive"]).abs().sum() / 
                  pred_df["y_true"].abs().sum()) if pred_df["y_true"].abs().sum() > 0 else 0
    seasonal_wape = ((pred_df["y_true"] - pred_df["y_pred_seasonal"]).abs().sum() / 
                     pred_df["y_true"].abs().sum()) if pred_df["y_true"].abs().sum() > 0 else 0
    
    model_wape = None
    if "y_pred_model" in pred_df.columns:
        model_wape = ((pred_df["y_true"] - pred_df["y_pred_model"]).abs().sum() / 
                      pred_df["y_true"].abs().sum()) if pred_df["y_true"].abs().sum() > 0 else 0

    with col1:
        st.metric("Naive WAPE", f"{naive_wape:.4f}", delta=None)
    with col2:
        st.metric("Seasonal WAPE", f"{seasonal_wape:.4f}", 
                 delta=f"{(seasonal_wape - naive_wape):.4f}" if seasonal_wape != naive_wape else None)
    with col3:
        if model_wape is not None:
            improvement = naive_wape - model_wape
            st.metric("Model WAPE", f"{model_wape:.4f}", 
                     delta=f"{improvement:.4f} (better)" if improvement > 0 else None,
                     delta_color="inverse")
        else:
            st.metric("Model WAPE", "N/A", help="Model belum di-train")
    with col4:
        st.metric("Test Rows", len(pred_df))
        st.metric("Unique SKU-Locations", pred_df[["store_id", "product_id"]].drop_duplicates().shape[0])

    st.markdown("---")

    # --- Forecast Comparison Chart ---
    st.header("📉 Forecast vs Actual (Sample)")

    # Pilih beberapa SKU-location untuk ditampilkan
    unique_pairs = pred_df[["store_id", "product_id"]].drop_duplicates()
    
    col_left, col_right = st.columns([3, 1])
    with col_left:
        selected_pair = st.selectbox(
            "Pilih SKU-Location:",
            options=[f"{row['store_id']} | {row['product_id']}" 
                    for _, row in unique_pairs.head(20).iterrows()],
            index=0
        )
    
    if selected_pair:
        sid, pid = selected_pair.split(" | ")
        pair_data = pred_df[(pred_df["store_id"] == sid) & (pred_df["product_id"] == pid)].copy()
        pair_data = pair_data.sort_values(["year", "week"])

        # Buat chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(len(pair_data))),
            y=pair_data["y_true"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8)
        ))

        fig.add_trace(go.Scatter(
            x=list(range(len(pair_data))),
            y=pair_data["y_pred_naive"],
            mode="lines+markers",
            name="Naive Forecast",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            marker=dict(size=6)
        ))

        fig.add_trace(go.Scatter(
            x=list(range(len(pair_data))),
            y=pair_data["y_pred_seasonal"],
            mode="lines+markers",
            name="Seasonal Forecast",
            line=dict(color="#2ca02c", width=2, dash="dot"),
            marker=dict(size=6)
        ))

        if "y_pred_model" in pair_data.columns:
            fig.add_trace(go.Scatter(
                x=list(range(len(pair_data))),
                y=pair_data["y_pred_model"],
                mode="lines+markers",
                name="Model Forecast",
                line=dict(color="#d62728", width=2),
                marker=dict(size=6)
            ))

        fig.update_layout(
            title=f"Forecast Comparison: {sid} | {pid}",
            xaxis_title="Week Index (Test Period)",
            yaxis_title="Units Sold",
            hovermode="x unified",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Error Distribution ---
    st.header("📊 Error Distribution")

    col_err1, col_err2 = st.columns(2)

    with col_err1:
        pred_df["naive_error"] = (pred_df["y_true"] - pred_df["y_pred_naive"]).abs()
        fig_err_naive = px.histogram(
            pred_df, 
            x="naive_error",
            nbins=30,
            title="Naive Forecast Error Distribution",
            labels={"naive_error": "Absolute Error", "count": "Frequency"}
        )
        st.plotly_chart(fig_err_naive, use_container_width=True)

    with col_err2:
        if "y_pred_model" in pred_df.columns:
            pred_df["model_error"] = (pred_df["y_true"] - pred_df["y_pred_model"]).abs()
            fig_err_model = px.histogram(
                pred_df,
                x="model_error",
                nbins=30,
                title="Model Forecast Error Distribution",
                labels={"model_error": "Absolute Error", "count": "Frequency"},
                color_discrete_sequence=["#d62728"]
            )
            st.plotly_chart(fig_err_model, use_container_width=True)
        else:
            st.info("Model forecast belum tersedia untuk error analysis.")

    st.markdown("---")

    # --- Per SKU-Location Metrics (if requested) ---
    if show_details:
        st.header("🔍 Detail per SKU-Location")

        # Hitung WAPE per pair
        metrics_list = []
        for (sid, pid), grp in pred_df.groupby(["store_id", "product_id"]):
            y_true = grp["y_true"]
            naive_wape_pair = ((y_true - grp["y_pred_naive"]).abs().sum() / 
                              y_true.abs().sum()) if y_true.abs().sum() > 0 else 0
            seasonal_wape_pair = ((y_true - grp["y_pred_seasonal"]).abs().sum() / 
                                 y_true.abs().sum()) if y_true.abs().sum() > 0 else 0
            
            row = {
                "store_id": sid,
                "product_id": pid,
                "naive_wape": naive_wape_pair,
                "seasonal_wape": seasonal_wape_pair,
                "mean_actual": y_true.mean(),
            }
            
            if "y_pred_model" in grp.columns:
                model_wape_pair = ((y_true - grp["y_pred_model"]).abs().sum() / 
                                  y_true.abs().sum()) if y_true.abs().sum() > 0 else 0
                row["model_wape"] = model_wape_pair
            
            metrics_list.append(row)

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df = metrics_df.sort_values("naive_wape", ascending=False)

        st.dataframe(
            metrics_df.style.format({
                "naive_wape": "{:.4f}",
                "seasonal_wape": "{:.4f}",
                "model_wape": "{:.4f}" if "model_wape" in metrics_df.columns else None,
                "mean_actual": "{:.2f}"
            }),
            use_container_width=True,
            height=400
        )

    # --- Footer ---
    st.markdown("---")
    st.caption("💡 Dashboard ini membaca dari `data/processed/predictions.csv`. Refresh data setelah menjalankan evaluasi baru.")


if __name__ == "__main__":
    main()
