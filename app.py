# ───────────────────────────────────────────────────────────────
# EcoWise Insight Studio – resilient Streamlit dashboard (app.py)
# ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
# Page config
st.set_page_config(layout="wide", page_title="EcoWise Insight Studio")
st.title("🌿 EcoWise Insight Studio 2025")
st.caption("Sustainable-appliance market intelligence dashboard")

# ───────────────────────────────────────────────────────────────
# Data loader
@st.cache_data(show_spinner=True)
def load_csv(local_fname: str, url_fallback: str | None) -> pd.DataFrame:
    p = Path(__file__).parent / local_fname
    if p.exists():
        st.caption(f"✅ Loaded local file: {p.name}")
        return pd.read_csv(p)
    if url_fallback:
        st.caption("🔄 Fetching from remote URL…")
        return pd.read_csv(url_fallback)
    st.error(
        "⚠️  Dataset not found.  "
        "Commit the CSV or set a DATA_URL secret that points to a raw CSV."
    )
    st.stop()

LOCAL_CSV = "ecowise_full_arm_ready.xlsx"      # adjust if your path differs
DATA_URL  = st.secrets.get("DATA_URL", "")
df = load_csv(LOCAL_CSV, DATA_URL)

# ───────────────────────────────────────────────────────────────
# Header normalisation
df.columns = (
    df.columns
      .str.strip()
      .str.replace(" ", "_")
      .str.lower()
)

# Rename any col that includes BOTH “will” and “adopt”
possible = [c for c in df.columns if "will" in c and "adopt" in c]
if possible and "will_adopt" not in df.columns:
    df.rename(columns={possible[0]: "will_adopt"}, inplace=True)

has_target = "will_adopt" in df.columns

# ───────────────────────────────────────────────────────────────
# Sidebar filters
with st.sidebar:
    st.subheader("🔍 Global Filters")
    country_opts = sorted(df["country"].unique()) if "country" in df.columns else []
    country_sel  = st.multiselect("Country", country_opts, default=country_opts)

    age_range = st.slider("Age range", 18, 75, (18, 75))
    if has_target:
        adopt_sel = st.selectbox("Adoption status",
                                 ["All", "Will Adopt = 1", "Will Adopt = 0"])
    else:
        st.info("Classification disabled – no *will_adopt* column found.")
        adopt_sel = "All"

filt_df = df.copy()
if "country" in filt_df.columns:
    filt_df = filt_df[filt_df["country"].isin(country_sel)]
filt_df = filt_df[filt_df["age"].between(*age_range)]
if adopt_sel.endswith("1"):
    filt_df = filt_df[filt_df["will_adopt"] == 1]
elif adopt_sel.endswith("0"):
    filt_df = filt_df[filt_df["will_adopt"] == 0]

# ───────────────────────────────────────────────────────────────
# Tabs (omit Classification if target missing)
tab_labels = ["📊 Data Visualisation"]
if has_target:
    tab_labels.append("🤖 Classification")
tab_labels += ["🧩 Clustering", "🔗 Association Rules", "📈 Regression"]
tabs = st.tabs(tab_labels)

# ==============================================================
# 1  Data Visualisation
# ==============================================================
with tabs[0]:
    st.header("Descriptive Insights")
    if "country" in filt_df.columns and "will_adopt" in filt_df.columns:
        adoption = filt_df.groupby("country")["will_adopt"].mean()
        st.bar_chart(adoption, y="will_adopt")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income distribution (USD)")
        st.hist_chart(filt_df, x="annual_hh_income_usd", bins=30)

    with col2:
        st.subheader("kWh vs. EE-share")
        if {"monthly_kwh", "ee_appliance_share"} <= set(filt_df.columns):
            st.scatter_chart(filt_df, x="monthly_kwh", y="ee_appliance_share")

# ==============================================================
# 2  Classification  (only if *will_adopt* exists)
# ==============================================================
if has_target:
    with tabs[1]:
        st.header("Classification – Predict Adoption")
        target = "will_adopt"
        X, y = df.drop(columns=[target]), df[target]

        num_cols = X.select_dtypes("number").columns.tolist()
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

        metrics, roc_curves, trained = [], [], {}
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )
        for name, model in base_models.items():
            pipe = Pipeline([("pre", pre), ("model", model)]).fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            metrics.append({
                "Model": name,
                "Acc":  accuracy_score(y_te, y_pred),
                "Prec": precision_score(y_te, y_pred),
                "Rec":  recall_score(y_te, y_pred),
                "F1":   f1_score(y_te, y_pred)
            })
            y_prob = (pipe.predict_proba(X_te)[:, 1]
                      if hasattr(pipe, "predict_proba") else y_pred)
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            roc_curves.append((fpr, tpr, name, auc(fpr, tpr)))
            trained[name] = pipe

        st.dataframe(pd.DataFrame(metrics).set_index("Model").style.format("{:.3f}"))

        st.subheader("ROC curves")
        fig, ax = plt.subplots()
        for fpr, tpr, label, roc_auc in roc_curves:
            ax.plot(fpr, tpr, label=f"{label} (AUC {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "--", lw=1); ax.legend()
        st.pyplot(fig)

# ==============================================================
# 3  Clustering
# ==============================================================
with tabs[-3]:
    st.header("Customer Clustering")
    num_cols = df.select_dtypes("number").columns.tolist()
    cluster_feats = st.multiselect(
        "Numeric columns", num_cols,
        default=[c for c in ["age", "annual_hh_income_usd",
                             "monthly_kwh", "premium_wtp_pct"] if c in num_cols]
    )
    k = st.slider("k", 2, 10, 4)
    km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(df[cluster_feats])
    df_clust = df.assign(cluster=km.labels_)
    st.write(df_clust.groupby("cluster")[cluster_feats].mean().round(2))
    st.download_button("Download clusters",
                       df_clust.to_csv(index=False).encode(),
                       "ew_clustered.csv", "text/csv")

# ==============================================================
# 4  Association Rules
# ==============================================================
with tabs[-2]:
    st.header("Association Rule Mining")
    bool_cols = [c for c in df.columns if c.startswith(
        ("feat_", "cat_", "src_", "factor_", "barrier_"))]
    cols_sel = st.multiselect("Columns", bool_cols, default=bool_cols[:20])
    supp  = st.slider("Support",   0.01, 0.30, 0.05, 0.01)
    conf  = st.slider("Confidence",0.10, 1.00, 0.30, 0.05)
    lift_min = st.slider("Lift",   1.00, 5.00, 1.10, 0.10)

    freq = apriori(df[cols_sel], min_support=supp, use_colnames=True)
    rules = association_rules(freq, "confidence", conf)
    rules = rules[rules["lift"] >= lift_min].sort_values("lift", ascending=False).head(10)
    st.write(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# ==============================================================
# 5  Regression
# ==============================================================
with tabs[-1]:
    st.header("Regression Insights")
    reg_targets = ["budget_usd", "premium_wtp_pct", "acceptable_payback_yrs"]
    reg_targets = [t for t in reg_targets if t in df.columns]
    if not reg_targets:
        st.warning("No numeric target columns available.")
        st.stop()
    target = st.selectbox("Target variable", reg_targets)
    X_reg, y_reg = df.drop(columns=[target]), df[target]

    num_cols = X_reg.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in X_reg.columns if c not in num_cols]
    pre_reg = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    reg_models = {
        "Linear": LinearRegression(),
        "Ridge":  Ridge(),
        "Lasso":  Lasso(alpha=0.01),
        "Tree":   DecisionTreeRegressor(max_depth=5, random_state=0)
    }
    rows = []
    for name, model in reg_models.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=0)
        pipe = Pipeline([("pre", pre_reg), ("model", model)]).fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        rows.append({"Model": name,
                     "R²":  r2_score(y_te, y_pred),
                     "MAE": mean_absolute_error(y_te, y_pred)})
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.3f}"))
