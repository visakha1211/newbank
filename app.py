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
# ───────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="EcoWise Insight Studio")
st.title("🌿 EcoWise Insight Studio 2025")
st.caption("Sustainable-appliance market intelligence dashboard")

# ───────────────────────────────────────────────────────────────
# Robust data loader
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_data(local_fname: str, url_fallback: str | None) -> pd.DataFrame:
    """Try repo file, otherwise URL; stop nicely if neither works."""
    local_path = Path(__file__).parent / local_fname
    if local_path.exists():
        st.caption(f"✅ Loaded local file: {local_path.name}")
        return pd.read_csv(local_path)

    if url_fallback:
        st.caption("🔄 Fetching CSV from remote URL…")
        try:
            return pd.read_csv(url_fallback)
        except Exception as e:
            st.error(f"Remote fetch failed → {e}")
            st.stop()

    st.error(
        "⚠️  No dataset found.\n"
        "• Commit the CSV (edit LOCAL_CSV if needed) **or**\n"
        "• Add a DATA_URL secret pointing to a raw CSV URL."
    )
    st.stop()

LOCAL_CSV = "ecowise_full_arm_ready.csv"     # adjust path if different
DATA_URL  = st.secrets.get("DATA_URL", "")
df = load_data(LOCAL_CSV, DATA_URL)

# ───────────────────────────────────────────────────────────────
# Header normalisation  (strip → lower → underscores)
# ───────────────────────────────────────────────────────────────
df.columns = (
    df.columns
      .str.strip()
      .str.replace(" ", "_")
      .str.lower()
)

# Quick sanity check: stop if core columns missing
required = {"country", "age", "will_adopt"}
missing  = required - set(df.columns)
if missing:
    st.error(f"Dataset missing required cols: {', '.join(missing)}")
    st.stop()

# ───────────────────────────────────────────────────────────────
# Sidebar filters
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("🔍 Global Filters")
    country_filter = st.multiselect(
        "Country", sorted(df["country"].unique()),
        default=list(df["country"].unique())
    )
    age_range   = st.slider("Age range", 18, 75, (18, 75))
    adopt_opt   = st.selectbox("Adoption status",
                               ["All", "Will Adopt = 1", "Will Adopt = 0"])

filt_df = df[
    df["country"].isin(country_filter) &
    df["age"].between(*age_range)
]
if adopt_opt.endswith("1"):
    filt_df = filt_df[filt_df["will_adopt"] == 1]
elif adopt_opt.endswith("0"):
    filt_df = filt_df[filt_df["will_adopt"] == 0]

# ───────────────────────────────────────────────────────────────
# Tabs
# ───────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Data Visualisation", "🤖 Classification",
    "🧩 Clustering", "🔗 Association Rules", "📈 Regression"
])

# ==============================================================
# 1  Data Visualisation
# ==============================================================
with tabs[0]:
    st.header("Descriptive Insights")
    col1, col2 = st.columns(2)

    # Adoption by country
    with col1:
        st.subheader("Adoption rate by country")
        adoption = filt_df.groupby("country")["will_adopt"].mean()
        fig, ax = plt.subplots()
        adoption.plot(kind="bar", ax=ax)
        ax.set_ylabel("Probability of adoption")
        st.pyplot(fig)

    # Income distribution
    with col2:
        st.subheader("Annual household income (USD)")
        fig2, ax2 = plt.subplots()
        ax2.hist(filt_df["annual_hh_income_usd"], bins=30)
        ax2.set_xlabel("Income")
        st.pyplot(fig2)

    more_charts = [
        ("Age vs. Premium willingness (%)", "age", "premium_wtp_pct"),
        ("kWh vs. EE share (%)", "monthly_kwh", "ee_appliance_share"),
        ("Education level vs. Adoption", "education_level", "will_adopt"),
        ("Payback tolerance by country", "country", "acceptable_payback_yrs"),
        ("Top decision factors", "primary_decision_factor", None),
        ("Main barriers", "main_barrier", None),
        ("Feature popularity", "wanted_features", None),
        ("Interested categories", "interested_categories", None)
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
# 2  Classification
# ==============================================================
with tabs[1]:
    st.header("Classification – Predict Adoption")
    target = "will_adopt"
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )

    base_models = {
        "KNN":               KNeighborsClassifier(),
        "Decision Tree":     DecisionTreeClassifier(random_state=0),
        "Random Forest":     RandomForestClassifier(random_state=0),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0)
    }

    metrics, roc_data, trained = [], [], {}
    for name, model in base_models.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )
        pipe = Pipeline([("pre", pre), ("model", model)]).fit(X_tr, y_tr)

        y_pred = pipe.predict(X_te)
        metrics.append({
            "Model": name,
            "Accuracy":  accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred),
            "Recall":    recall_score(y_te, y_pred),
            "F1":        f1_score(y_te, y_pred)
        })

        y_prob = (pipe.predict_proba(X_te)[:, 1]
                  if hasattr(pipe, "predict_proba") else y_pred)
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_data.append((fpr, tpr, name, auc(fpr, tpr)))
        trained[name] = pipe

    st.subheader("Performance")
    st.dataframe(pd.DataFrame(metrics).set_index("Model")
                 .style.format("{:.3f}"))

    st.subheader("ROC curves")
    fig, ax = plt.subplots()
    for fpr, tpr, label, roc_auc in roc_data:
        ax.plot(fpr, tpr, label=f"{label} (AUC {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    st.pyplot(fig)

    st.subheader("Confusion matrix")
    algo = st.selectbox("Algorithm", list(trained))
    pipe = trained[algo]
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    cm = confusion_matrix(y_te, pipe.predict(X_te))
    st.write(pd.DataFrame(cm,
             index=["Actual 0", "Actual 1"],
             columns=["Pred 0", "Pred 1"]))

    st.markdown("#### Upload new data for prediction")
    up = st.file_uploader("CSV without 'will_adopt'", type="csv")
    if up:
        new_df = pd.read_csv(up)
        new_df["pred_will_adopt"] = pipe.predict(new_df)
        st.write(new_df.head())
        st.download_button("Download",
                           new_df.to_csv(index=False).encode(),
                           "ew_predictions.csv", "text/csv")

# ==============================================================
# 3  Clustering
# ==============================================================
with tabs[2]:
    st.header("Customer Clustering")
    num_cols_all = df.select_dtypes("number").columns.tolist()
    cluster_feats = st.multiselect(
        "Numeric columns for clustering", num_cols_all,
        default=["age", "annual_hh_income_usd",
                 "monthly_kwh", "premium_wtp_pct"]
    )
    k = st.slider("k (clusters)", 2, 10, 4)

    inertias = [
        KMeans(n_clusters=i, random_state=0, n_init="auto")
        .fit(df[cluster_feats]).inertia_
        for i in range(2, 11)
    ]
    fig_e, ax_e = plt.subplots(); ax_e.plot(range(2, 11), inertias, marker="o")
    ax_e.set_xlabel("k"); ax_e.set_ylabel("Inertia"); st.pyplot(fig_e)

    km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(df[cluster_feats])
    df_clust = df.assign(cluster=km.labels_)
    st.subheader("Cluster personas (means)")
    st.write(df_clust.groupby("cluster")[cluster_feats + ["will_adopt"]]
             .mean().round(2))

    st.download_button("Download cluster data",
                       df_clust.to_csv(index=False).encode(),
                       "ew_clustered.csv", "text/csv")

# ==============================================================
# 4  Association Rules
# ==============================================================
with tabs[3]:
    st.header("Association Rule Mining")
    bool_cols = [c for c in df.columns if c.startswith(
        ("feat_", "cat_", "src_", "factor_", "barrier_"))]
    cols_sel = st.multiselect("Columns", bool_cols, default=bool_cols[:20])
    supp  = st.slider("Min support",    0.01, 0.30, 0.05, 0.01)
    conf  = st.slider("Min confidence", 0.10, 1.00, 0.30, 0.05)
    lift_min = st.slider("Min lift",    1.00, 5.00, 1.10, 0.10)

    freq  = apriori(df[cols_sel], min_support=supp, use_colnames=True)
    rules = association_rules(freq, "confidence", conf)
    rules = rules[rules["lift"] >= lift_min].sort_values("lift", ascending=False).head(10)
    st.write(rules[["antecedents", "consequents",
                    "support", "confidence", "lift"]])

# ==============================================================
# 5  Regression
# ==============================================================
with tabs[4]:
    st.header("Regression Insights")
    target_reg = st.selectbox("Target",
                              ["budget_usd", "premium_wtp_pct",
                               "acceptable_payback_yrs"])
    X_reg, y_reg = df.drop(columns=[target_reg]), df[target_reg]
    num_cols = X_reg.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in X_reg.columns if c not in num_cols]

    pre_reg = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )
    reg_models = {
        "Linear":        LinearRegression(),
        "Ridge":         Ridge(alpha=1.0),
        "Lasso":         Lasso(alpha=0.01),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=0)
    }

    metrics = []
    for name, model in reg_models.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=0)
        pipe = Pipeline([("pre", pre_reg), ("model", model)]).fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        metrics.append({
            "Model": name,
            "R²":  r2_score(y_te, y_pred),
            "MAE": mean_absolute_error(y_te, y_pred)
        })

    st.dataframe(pd.DataFrame(metrics).set_index("Model").style.format("{:.3f}"))
    st.info("Higher R² & lower MAE = better model.")
