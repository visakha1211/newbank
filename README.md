# EcoWise Insight Studio – Streamlit Dashboard

An interactive dashboard for exploring the **EcoWise Appliances** synthetic consumer‑survey dataset, built with Streamlit and ready to deploy on **Streamlit Cloud**.

## ✨ Features

| Tab | Highlights |
|-----|------------|
| **Data Visualisation** | 10+ rich descriptive charts with dynamic filters (country, age range, income, adoption status). |
| **Classification** | Train/test comparison for K‑NN, Decision Tree, Random Forest, Gradient Boosting. Confusion‑matrix toggle, unified ROC plot, CSV upload for new predictions, one‑click download of results. |
| **Clustering** | Elbow plot, cluster‑count slider (2‑10), persona summary table, download cluster‑labelled data. |
| **Association Rules** | Apriori with support/confidence lift filters and column selector; top‑10 rule display. |
| **Regression** | Linear, Ridge, Lasso, and Decision‑Tree regressors surface 5–7 quick insights on spend willingness. |

## 🛠 Quick start (local)

```bash
# clone repo & install
pip install -r requirements.txt
streamlit run app.py
```

## 🚀 Deploy on Streamlit Cloud

1. Push all files in this folder to a GitHub repo (public/private).  
2. In Streamlit Cloud “New app”, point to `app.py`.  
3. **Secrets**: add  

```toml
DATA_URL = "https://raw.githubusercontent.com/<user>/<repo>/main/ecowise_survey_arm_ready.csv"
```

4. Click **Deploy** – that’s it!

## 📂 Project structure

```
ecowise_dashboard/
├── app.py                # main Streamlit app
├── requirements.txt
└── README.md
```

The app fetches the CSV from the `DATA_URL` secret at runtime, so you can update the dataset without redeploying.
