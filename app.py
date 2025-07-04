# ───────────────────────────────────────────────────────────────
# EcoWise Insight Studio – full Streamlit dashboard (app.py)
# ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path

# ML / stats libs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             r2_score, mean_absolute_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# ───────────────────────────────────────────────────────────────
# Config & Page Header
# ───────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="EcoWise Insight Studio")
st.title("🌿 EcoWise Insight Studio 2025")
st.caption("Sustainable-appliance market intelligence dashboard")

# ───────────────────────────────────────────────────────────────
# Robust DATA LOADER – local file first, URL fallback
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_data(local_fname: str, url_fallback: str | None) -> pd.DataFrame:
    """Load from repo if present, otherwise from a URL."""
    local_path = Path(__file__).parent / local_fname
    if local_path.exists():
        st.caption(f"✅ Loaded data from local file: {local_path.name}")
        return pd.read_csv(local_path)

    if url_fallback:
        try:
            st.caption("🔄 Fetching data from remote URL…")
            return pd.read_csv(url_fallback)
        except Exception as e:
            st.error(f"Remote fetch failed → {e}")
            st.stop()

    st.error(
        "⚠️  No dataset found.\n\n"
        "• Add the CSV to the repo (edit LOCAL_CSV if needed) **or**\n"
        "• Set a DATA_URL secret pointing to a publicly readable raw-CSV URL."
    )
    st.stop()

# >>> Point to the CSV you actually committed <<<
LOCAL_CSV = "ecowise_full_arm_ready.csv"   # <-- updated filename
DATA_URL  = st.secrets.get("DATA_URL", "")  # leave blank if not using a URL

df = load_data(LOCAL_CSV, DATA_URL)

# ───────────────────────────────────────────────────────────────
# Sidebar Filters
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("🔍 Global Filters")
    country_filter = st.multiselect(
        "Country", sorted(df["Country"].unique()),
        default=list(df["Country"].unique())
    )
    age_range   = st.slider("Age range", 18, 75, (18, 75))
    adopt_filter = st.selectbox(
        "Adoption status", ["All", "Will Adopt = 1", "Will Adopt = 0"])

filt_df = df[
    df["Country"].isin(country_filter) &
    df["Age"].between(age_range[0], age_range[1])
]
if adopt_filter == "Will Adopt = 1":
    filt_df = filt_df[filt_df["Will_Adopt"] == 1]
elif adopt_filter == "Will Adopt = 0":
    filt_df = filt_df[filt_df["Will_Adopt"] == 0]

# ───────────────────────────────────────────────────────────────
# Tabs
# ───────────────────────────────────────────────────────────────
tabs = st.tabs(
    ["📊 Data Visualisation", "🤖 Classification",
     "🧩 Clustering", "🔗 Association Rules", "📈 Regression"]
)

# ==============================================================
# 1 Data Visualisation
# ==============================================================
with tabs[0]:
    st.header("Descriptive Insights")
    col1, col2 = st.columns(2)

    # Adoption by country
    with col1:
        st.subheader("Adoption rate by country")
        adoption = filt_df.groupby("Country")["Will_Adopt"].mean()
        fig, ax = plt.subplots()
        adoption.plot(kind="bar", ax=ax)
        ax.set_ylabel("Probability of adoption")
        st.pyplot(fig)

    # Income distribution
    with col2:
        st.subheader("Annual household income distribution (USD)")
        fig2, ax2 = plt.subplots()
        ax2.hist(filt_df["Annual_HH_Income_USD"], bins=30)
        ax2.set_xlabel("Income")
        st.pyplot(fig2)

    # More quick charts
    more_charts = [
        ("Age vs. Premium willingness (%)", "Age", "Premium_WTP_pct"),
        ("kWh vs. EE share (%)", "Monthly_kWh", "EE_Appliance_Share"),
        ("Education level vs. Adoption", "Education_Level", "Will_Adopt"),
        ("Payback tolerance by country", "Country", "Acceptable_Payback_yrs"),
        ("Top decision factors", "Primary_Decision_Factor", None),
        ("Main barriers", "Main_Barrier", None),
        ("Feature popularity", "Wanted_Features", None),
        ("Interested categories", "Interested_Categories", None)
    ]
    for title, xcol, ycol in more_charts:
        st.markdown(f"#### {title}")
        if ycol:
            fig, ax = plt.subplots()
            ax.scatter(filt_df[xcol], filt_df[ycol], alpha=0.4)
            ax.set_xlabel(xcol); ax.set_ylabel(ycol)
            st.pyplot(fig)
        else:
            exploded = filt_df[xcol].explode()
            top_vals = exploded.value_counts().head(10)
            fig, ax = plt.subplots()
            top_vals.plot(kind="barh", ax=ax)
            st.pyplot(fig)

# ==============================================================
# 2 Classification
# ==============================================================
with tabs[1]:
    st.header("Classification – Predict Adoption")

    target = "Will_Adopt"
    y = df[target]
    X = df.drop(columns=[target])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    base_models = {
        "KNN":               KNeighborsClassifier(),
        "Decision Tree":     DecisionTreeClassifier(random_state=0),
        "Random Forest":     RandomForestClassifier(random_state=0),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0)
    }

    metrics, roc_data, trained = [], [], {}
    for name, model in base_models.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y)
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_tr, y_tr)

        y_pred = pipe.predict(X_te)
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred, zero_division=0),
            "Recall": recall_score(y_te, y_pred, zero_division=0),
            "F1": f1_score(y_te, y_pred, zero_division=0)
        })

        y_prob = (pipe.predict_proba(X_te)[:, 1]
                  if hasattr(pipe, "predict_proba") else y_pred)
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_data.append((fpr, tpr, name, auc(fpr, tpr)))
        trained[name] = pipe

    st.subheader("Performance summary")
    st.dataframe(pd.DataFrame(metrics)
                 .set_index("Model").style.format("{:.3f}"))

    st.subheader("ROC curves")
    fig, ax = plt.subplots()
    for fpr, tpr, label, roc_auc in roc_data:
        ax.plot(fpr, tpr, label=f"{label} (AUC {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Confusion matrix")
    algo = st.selectbox("Choose algorithm", list(trained.keys()))
    pipe = trained[algo]
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    cm = confusion_matrix(y_te, pipe.predict(X_te))
    st.write(pd.DataFrame(cm,
             index=["Actual 0", "Actual 1"],
             columns=["Pred 0", "Pred 1"]))

    st.markdown("#### Upload new data for prediction")
    up = st.file_uploader("CSV without 'Will_Adopt'", type="csv")
    if up:
        new_df = pd.read_csv(up)
        new_df["Pred_Will_Adopt"] = pipe.predict(new_df)
        st.write(new_df.head())
        st.download_button("Download predictions",
                           new_df.to_csv(index=False).encode(),
                           file_name="ew_predictions.csv",
                           mime="text/csv")

# ==============================================================
# 3 Clustering
# ==============================================================
with tabs[2]:
    st.header("Customer Clustering")
    num_cols_all = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cluster_feats = st.multiselect(
        "Numeric columns to use", num_cols_all,
        default=["Age", "Annual_HH_Income_USD", "Monthly_kWh", "Premium_WTP_pct"]
    )
    k = st.slider("Number of clusters (k)", 2, 10, 4)

    # Elbow
    inertias = [KMeans(n_clusters=k_elb, random_state=0, n_init="auto")
                .fit(df[cluster_feats]).inertia_ for k_elb in range(2, 11)]
    fig_elb, ax_elb = plt.subplots()
    ax_elb.plot(range(2, 11), inertias, marker="o")
    ax_elb.set_xlabel("k"); ax_elb.set_ylabel("Inertia")
    st.pyplot(fig_elb)

    km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(df[cluster_feats])
    df_clust = df.assign(Cluster=km.labels_)
    st.subheader("Cluster personas (feature means)")
    st.write(df_clust.groupby("Cluster")[cluster_feats + ["Will_Adopt"]]
             .mean().round(2))

    st.download_button("Download cluster-labelled data",
                       df_clust.to_csv(index=False).encode(),
                       file_name="ew_clustered.csv",
                       mime="text/csv")

# ==============================================================
# 4 Association Rule Mining
# ==============================================================
with tabs[3]:
    st.header("Association Rule Mining")
    bool_cols = [c for c in df.columns if c.startswith(
        ("Feat_", "Cat_", "Src_", "Factor_", "Barrier_"))]
    cols_sel = st.multiselect("Columns to include", bool_cols, default=bool_cols[:20])
    supp  = st.slider("Min support",     0.01, 0.30, 0.05, 0.01)
    conf  = st.slider("Min confidence",  0.10, 1.00, 0.30, 0.05)
    lift_min = st.slider("Min lift",     1.00, 5.00, 1.10, 0.10)

    freq  = apriori(df[cols_sel], min_support=supp, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf)
    rules = rules[rules["lift"] >= lift_min].sort_values("lift", ascending=False).head(10)
    st.write(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# ==============================================================
# 5 Regression
# ==============================================================
with tabs[4]:
    st.header("Regression Insights")
    target_reg = st.selectbox("Target variable",
                              ["Budget_USD", "Premium_WTP_pct", "Acceptable_Payback_yrs"])
    y_reg = df[target_reg]
    X_reg = df.drop(columns=[target_reg])

    num_cols = X_reg.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X_reg.columns if c not in num_cols]

    pre_reg = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    models_reg = {
        "Linear":         LinearRegression(),
        "Ridge":          Ridge(alpha=1.0),
        "Lasso":          Lasso(alpha=0.01),
        "Decision Tree":  DecisionTreeRegressor(random_state=0, max_depth=5)
    }

    metrics = []
    for name, model in models_reg.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=0)
        pipe = Pipeline([("pre", pre_reg), ("model", model)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        metrics.append({
            "Model": name,
            "R²":  r2_score(y_te, y_pred),
            "MAE": mean_absolute_error(y_te, y_pred)
        })

    st.dataframe(pd.DataFrame(metrics).set_index("Model").style.format("{:.3f}"))
    st.info("Higher R² and lower MAE indicate better predictive power.")


    st.dataframe(pd.DataFrame(metrics).set_index("Model").style.format("{:.3f}"))
    st.info("Higher R² and lower MAE indicate better predictive power.")

