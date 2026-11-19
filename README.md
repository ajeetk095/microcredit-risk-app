# 📶 Micro-Credit Risk Hub

Streamlit app that predicts **delinquency on telecom airtime micro-loans** (5 / 10 INR, repay 6 / 12 INR within 5 days),
turns the predictions into **approve / limit / decline decisions**, and adds **live market context from Yahoo Finance**.
The Home page shows auto-generated recommendations for management.

| Page | What it does |
|---|---|
| **Home** | KPIs, live market regime, prioritised recommendations |
| 🎯 Credit Scorer | Score one customer, see risk tier, action, expected profit, reasons |
| 📂 Batch Scoring | Upload a CSV, get decisions for every customer, download results |
| 📈 Market Intelligence | Airtel / Vodafone Idea / Reliance / Indus Towers, Nifty, Nifty Bank, India VIX, USD/INR → stress index |
| 📊 Model Insights | Model comparison, drivers, calibration, segment default rates, cut-off trade-off |
| 💰 Business Simulator | Change fee / recovery / cut-off and see monthly profit |

## Repository layout
```
app.py                  Home page + recommendations
pages/                  Other Streamlit pages
src/preprocess.py       Cleaning + feature engineering (shared by training and app)
src/model_utils.py      Model loading, scoring, tiers, reason codes
src/market.py           Yahoo Finance download + market stress index
src/recommendations.py  Turns results into management recommendations
train_model.py          Re-trains everything and rewrites models/ and data/
models/                 model.joblib, metrics.json, insights.json (committed)
data/                   holdout_scores.csv, sample_customers.csv (committed)
legacy/                 Your original hackathon script (reference only)
requirements.txt        Pinned dependencies
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Re-train (optional)
Put `Micro_delinquency_train_test_dataset_v2.csv` in the project root (it is git-ignored) and run:
```bash
python train_model.py --data Micro_delinquency_train_test_dataset_v2.csv
```

---
# Deploy: GitHub → Streamlit Community Cloud

### 1. Put the project on GitHub
1. Create a free account at github.com and click **New repository** → name it e.g. `microcredit-risk-app` → **Public** → *don't* add a README → **Create**.
2. In a terminal inside the project folder:
```bash
git init
git add .
git commit -m "Micro-credit risk app"
git branch -M main
git remote add origin https://github.com/<your-username>/microcredit-risk-app.git
git push -u origin main
```
(No Git? Alternative: on the repo page click **Add file → Upload files**, drag in the *contents* of the folder, including `pages/`, `src/`, `models/`, `data/`, `.streamlit/`, and commit.)

### 2. Make it live on Streamlit
1. Go to **share.streamlit.io** and sign in with GitHub (authorise access to the repo).
2. Click **Create app → Deploy a public app from GitHub**.
3. Choose your repo, branch `main`, main file path **`app.py`**.
4. Open **Advanced settings → Python version: 3.12** (the model file was saved with scikit-learn 1.9.1, which `requirements.txt` pins).
5. Click **Deploy**. First build takes 2-4 minutes; you get a public URL like `https://<your-app>.streamlit.app`.
6. Updates: every `git push` to `main` redeploys automatically.

### Troubleshooting
* **Model fails to load / version error** → Python or scikit-learn differs from training. Run `python train_model.py` locally with your installed versions, commit the new `models/` folder, push.
* **Market page says "Could not reach Yahoo Finance"** → Yahoo occasionally rate-limits shared cloud IPs. Click *Refresh now* after a minute; the rest of the app keeps working with the standard cut-off.
* **Private data**: don't push real customer files. The bundled `sample_customers.csv` has masked IDs.

## Honest notes on method
* **Label maturity:** every loan issued after ~24 Jul 2016 (53k rows, 28%) has label = 1, i.e. exactly 0% default for weeks versus 15-19% before. This is almost certainly unmatured labels, so `train_model.py` detects and drops that trailing window. Matured default rate is 17.4% (vs 12.5% if they were counted) - above the 16.7% break-even. Confirm with the data owner.
* **Output per the brief:** probability of repayment within 5 days + label (1 = repays, 0 = delinquent) per loan - see Batch Scoring and `data/holdout_predictions.csv`.
* The original script scored on its own training file and fitted PCA separately on test data; this version evaluates on a **customer-level hold-out** and uses a NaN-aware gradient-boosting model on cleaned + engineered features (hold-out ROC-AUC ≈ 0.89; out-of-time ≈ 0.85).
* Yahoo Finance contains **no customer data**. It feeds a transparent, rule-based *market stress overlay* that tightens the approval cut-off by 0 / 1.5 / 3 points in Calm / Elevated / Stressed regimes. Calibrate those numbers against your own loss history.
* Dataset dates are 2016 and cover one telecom circle - retrain on recent data before real lending decisions.
* "Default" = not repaid within 5 days; late payers may still repay. Use the Business Simulator's recovery slider to see the effect.
