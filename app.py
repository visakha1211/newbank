import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, r2_score, mean_absolute_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide", page_title="EcoWise Insight Studio")
st.title("🌿 EcoWise Insight Studio 2025")
st.caption("Sustainable‑appliance market intelligence dashboard")

# ---------- DATA LOADING ----------
from pathlib import Path
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=True)
def load_data(local_fname: str, url_fallback: str | None) -> pd.DataFrame:
    """
    Try to read a CSV from the repo first; if absent, fall back to a remote URL.
    Stops the app with a friendly error if neither source is available.
    """
    local_path = Path(__file__).parent / local_fname
    if local_path.exists():
        st.caption(f"✅ Loaded data from local file: {local_path.relative_to(Path(__file__).parent)}")
        return pd.read_csv(local_path)

    if url_fallback:
        try:
            st.caption("🔄 Fetching data from remote URL…")
            return pd.read_csv(url_fallback)
        except Exception as e:
            st.error(f"Could not fetch CSV from `{url_fallback}`\\n{e}")
            st.stop()

    st.error("⚠️  No dataset found. Add the CSV to the repo **or** set a DATA_URL secret.")
    st.stop()

# ──> EDIT THIS PATH if you put the file elsewhere (e.g. just "ecowise_survey_arm_ready.csv")
LOCAL_CSV = "data/ecowise_survey_arm_ready.csv"

# If you created a Streamlit secret, this pulls it; otherwise an empty string
DATA_URL  = st.secrets.get("DATA_URL", "")

# Call the helper; the returned DataFrame is what the rest of the app uses
df = load_data(LOCAL_CSV, DATA_URL)



# Sidebar filters
with st.sidebar:
    st.subheader("🔍 Global Filters")
    country_filter = st.multiselect("Country", sorted(df["Country"].unique()), default=list(df["Country"].unique()))
    age_range = st.slider("Age range", 18, 75, (18, 75))
    adopt_filter = st.selectbox("Adoption status", ["All", "Will Adopt = 1", "Will Adopt = 0"])

filt_df = df[
    df["Country"].isin(country_filter)
    & df["Age"].between(age_range[0], age_range[1])
]
if adopt_filter == "Will Adopt = 1":
    filt_df = filt_df[filt_df["Will_Adopt"] == 1]
elif adopt_filter == "Will Adopt = 0":
    filt_df = filt_df[filt_df["Will_Adopt"] == 0]

# ---------- TABS ----------
tabs = st.tabs(["📊 Data Visualisation", "🤖 Classification", "🧩 Clustering", "🔗 Association Rules", "📈 Regression"])

# -------------------------------------------------
# 1. DATA VISUALISATION
# -------------------------------------------------
with tabs[0]:
    st.header("Descriptive Insights")
    col1, col2 = st.columns(2)

    # Chart 1: Adoption by country
    with col1:
        st.subheader("Adoption rate by country")
        adoption = filt_df.groupby("Country")["Will_Adopt"].mean()
        fig, ax = plt.subplots()
        adoption.plot(kind="bar", ax=ax)
        ax.set_ylabel("Probability of adoption")
        st.pyplot(fig)
        st.caption("Shows proportion of consumers likely to buy EcoWise within 6 months.")

    # Chart 2: Income distribution
    with col2:
        st.subheader("Annual household income distribution (USD)")
        fig2, ax2 = plt.subplots()
        ax2.hist(filt_df["Annual_HH_Income_USD"], bins=30)
        ax2.set_xlabel("Income")
        st.pyplot(fig2)
        st.caption("Right‑skewed; note the long‑tail high‑income outliers.")

    # Additional quick charts
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

    st.success("10+ visuals rendered. Apply sidebar filters to refine insights.")

# -------------------------------------------------
# 2. CLASSIFICATION
# -------------------------------------------------
with tabs[1]:
    st.header("Classification – Predict Adoption")
    # Preprocess
    target = "Will_Adopt"
    y = df[target]
    X = df.drop(columns=[target])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(random_state=0),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0)
    }

    metrics_table = []
    roc_data = []  # (fpr, tpr, label)
    for name, model in models.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        metrics_table.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})

        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:,1]
        else:  # KNN etc.
            y_prob = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else y_pred
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data.append((fpr, tpr, name, auc(fpr, tpr)))

        models[name] = pipe  # overwrite with trained pipeline

    st.subheader("Performance summary")
    st.dataframe(pd.DataFrame(metrics_table).set_index("Model").style.format("{:.3f}"))

    # ROC plot
    st.subheader("ROC curves")
    fig, ax = plt.subplots()
    for fpr, tpr, label, roc_auc in roc_data:
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})")
    ax.plot([0,1],[0,1],"--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Confusion matrix viewer
    st.subheader("Confusion matrix")
    algo = st.selectbox("Choose algorithm", list(models.keys()))
    pipe = models[algo]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write(pd.DataFrame(cm,
                          index=["Actual 0","Actual 1"],
                          columns=["Pred 0","Pred 1"]))

    # Upload new data
    st.markdown("#### Upload new data for prediction")
    uploaded = st.file_uploader("CSV without 'Will_Adopt' column", type="csv")
    if uploaded:
        new_df = pd.read_csv(uploaded)
        preds = pipe.predict(new_df)
        new_df["Pred_Will_Adopt"] = preds
        st.write(new_df.head())
        st.download_button("Download predictions",
                           data=new_df.to_csv(index=False).encode(),
                           file_name="ew_predictions.csv",
                           mime="text/csv")

# -------------------------------------------------
# 3. CLUSTERING
# -------------------------------------------------
with tabs[2]:
    st.header("Customer Clustering")
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    cluster_features = st.multiselect("Numeric columns to use", numeric_cols, default=["Age","Annual_HH_Income_USD","Monthly_kWh","Premium_WTP_pct"])

    k = st.slider("Number of clusters (k)", 2, 10, 4, 1)

    # Elbow
    inertias = []
    for k_elb in range(2, 11):
        km = KMeans(n_clusters=k_elb, random_state=0, n_init="auto").fit(df[cluster_features])
        inertias.append(km.inertia_)
    fig_elb, ax_elb = plt.subplots()
    ax_elb.plot(range(2,11), inertias, marker="o")
    ax_elb.set_xlabel("k"); ax_elb.set_ylabel("Inertia")
    st.pyplot(fig_elb)

    km_final = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(df[cluster_features])
    df_clustered = df.copy()
    df_clustered["Cluster"] = km_final.labels_

    st.subheader("Cluster personas (means)")
    persona = df_clustered.groupby("Cluster")[cluster_features+["Will_Adopt"]].mean().round(2)
    st.write(persona)

    st.download_button("Download data with cluster labels",
                       data=df_clustered.to_csv(index=False).encode(),
                       file_name="ew_clustered.csv",
                       mime="text/csv")

# -------------------------------------------------
# 4. ASSOCIATION RULES
# -------------------------------------------------
with tabs[3]:
    st.header("Association Rule Mining")
    # Boolean columns start with prefixes used in dataset
    bool_cols = [c for c in df.columns if c.startswith(("Feat_","Cat_","Src_","Factor_","Barrier_"))]
    cols_selected = st.multiselect("Columns to include", bool_cols, default=bool_cols[:20])
    supp = st.slider("Min support", 0.01, 0.3, 0.05, 0.01)
    conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)
    lift_min = st.slider("Min lift", 1.0, 5.0, 1.1, 0.1)

    freq = apriori(df[cols_selected], min_support=supp, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf)
    rules = rules[rules["lift"] >= lift_min]
    rules = rules.sort_values("lift", ascending=False).head(10)

    st.write(rules[["antecedents","consequents","support","confidence","lift"]])

# -------------------------------------------------
# 5. REGRESSION
# -------------------------------------------------
with tabs[4]:
    st.header("Regression Insights")
    target_reg = st.selectbox("Choose target variable",
                              ["Budget_USD", "Premium_WTP_pct", "Acceptable_Payback_yrs"])
    yreg = df[target_reg]
    Xreg = df.drop(columns=[target_reg])
    num_cols = Xreg.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = [c for c in Xreg.columns if c not in num_cols]

    pre_reg = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    models_reg = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "DecisionTree": DecisionTreeRegressor(random_state=0, max_depth=5)
    }

    metrics = []
    for name, model in models_reg.items():
        Xtr, Xts, ytr, yts = train_test_split(Xreg, yreg, test_size=0.3, random_state=0)
        pipe = Pipeline([("pre", pre_reg), ("model", model)])
        pipe.fit(Xtr, ytr)
        y_pred_reg = pipe.predict(Xts)
        r2 = r2_score(yts, y_pred_reg)
        mae = mean_absolute_error(yts, y_pred_reg)
        metrics.append({"Model": name, "R2": r2, "MAE": mae})

    st.dataframe(pd.DataFrame(metrics).set_index("Model").style.format("{:.3f}"))

    st.info("Higher R² and lower MAE indicate better predictive power.")
