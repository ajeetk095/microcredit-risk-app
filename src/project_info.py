"""Project overview (problem, method, app map, tech, limits) - shown on the Business Simulator page."""
import pandas as pd
import streamlit as st


def render(metrics: dict, ins: dict) -> None:
    e, h, c, oot = metrics["econ"], metrics["holdout"], metrics["classification_at_economic_cut"], metrics["out_of_time"]
    lc = ins.get("label_censor", {})

    st.subheader("About this project")
    st.caption("Figures below are read live from `models/metrics.json` and `models/insights.json`, so they always match the deployed model.")

    k = st.columns(5)
    k[0].metric("Hold-out ROC-AUC", f"{metrics['roc_auc']:.3f}", metrics["best_model"], delta_color="off")
    k[1].metric("Out-of-time ROC-AUC", f"{oot['roc_auc']:.3f}", f"train < {oot['split_date']}", delta_color="off")
    k[2].metric("Defaulters caught", f"{c['recall_defaulters']*100:.0f}%", f"precision {c['precision_defaulters']*100:.0f}%", delta_color="off")
    k[3].metric("Hold-out profit", f"{h['pnl_model']:,.0f} INR", f"approve-all: {h['pnl_approve_all']:,.0f}", delta_color="off")
    k[4].metric("Loans modelled", f"{ins['n_rows']:,}", f"{ins['n_customers']:,} customers", delta_color="off")

    t1, t2, t3, t4, t5 = st.tabs(["🎯 Problem & solution", "🧪 Method", "🗺️ App map", "🧰 Tech & repo", "⚠️ Limits"])

    with t1:
        st.markdown(
            f"""
**The brief** (*Delinquency Telecom Model*, Algo8 interview use case): a telecom partners with a microfinance
institution to lend airtime on mobile balances, repayable within **5 days**. A customer is **delinquent** if they
don't repay in time. For every loan the model must output the **probability of repayment** and a **label**
(**1** = repays within 5 days, **0** = delinquent).

**What this app delivers beyond a notebook**
- **Per-loan probability + label**, with a risk tier, action (approve / limit / decline) and plain-English reason codes.
- **Loan economics:** a 5 INR loan returns 6 and a 10 INR loan returns 12 (a **{e['fee']*100:.0f}% fee**). With full loss on default,
  lending is profitable only when P(default) is below **{e['pd_cut']*100:.1f}%** — the decision threshold is derived from this, not the default 0.5.
- **Business impact:** on the customer-level hold-out of {h['n']:,} loans, the model policy earns **{h['pnl_model']:,.0f} INR** vs
  **{h['pnl_approve_all']:,.0f} INR** for approving everyone — an uplift of **{h['uplift']:,.0f} INR**.
- **Live market overlay:** Airtel, Vodafone Idea, Reliance, Indus Towers, Nifty 50, Nifty Bank, India VIX and USD/INR feed a
  stress index that tightens the cut-off by 0 / 1.5 / 3 points in Calm / Elevated / Stressed regimes.
- **Auto-generated management recommendations** on the Home page.
"""
        )

    with t2:
        st.markdown(
            f"""
**Improvements over the original hackathon script** (kept in `legacy/team_2_original.py`):

| Original approach | This version |
|---|---|
| Scored on its own training file | **Customer-level hold-out** (`GroupShuffleSplit` on `msisdn`) — no customer in both train and test |
| PCA fitted separately on test data | No PCA; engineered features (e.g. `spend_trend`, `balance_to_spend`) shared by training and app |
| Corrupted values `abs()`-ed / mean-filled | Corrupted values → **NaN**, handled natively by the tree model ({ins['data_quality']['corrupted_or_missing_cells']:,} of {ins['data_quality']['total_cells']:,} cells) |
| Default 0.5 threshold | Threshold from **loan economics** (break-even {e['pd_cut']*100:.1f}%) |
| Accuracy only | ROC-AUC, PR-AUC, Brier score, calibration, **out-of-time** test and profit |

**Label-maturity check.** Loans issued on/after **{lc.get('from_date', 'n/a')}** show exactly 0% default, which points to unmatured labels
rather than real behaviour. {lc.get('rows_dropped', 0):,} rows were excluded; the default rate on matured loans is
**{lc.get('default_rate_matured', 0)*100:.1f}%** (vs {lc.get('default_rate_all_rows', 0)*100:.1f}% if they were counted).
"""
        )
        st.markdown("**Model comparison (hold-out)**")
        comp = pd.DataFrame(metrics["comparison"]).rename(
            columns={"model": "Model", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier (lower = better)"})
        st.dataframe(comp.style.format({"ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}", "Brier (lower = better)": "{:.3f}"}),
                     hide_index=True, width="stretch")
        st.caption(f"Out-of-time check: trained on loans before {oot['split_date']} ({oot['n_train']:,} rows), tested on "
                   f"{oot['n_test']:,} later rows — ROC-AUC {oot['roc_auc']:.3f}, PR-AUC {oot['pr_auc']:.3f}. "
                   f"Model last trained {metrics['trained_at']}.")

    with t3:
        st.markdown("Each page in the sidebar answers a different question:")
        st.page_link("app.py", label="Home — KPIs, market regime, management recommendations", icon="📶")
        st.page_link("pages/1_Credit_Scorer.py", label="Credit Scorer — score one customer, see tier, action, expected profit, reasons", icon="🎯")
        st.page_link("pages/2_Batch_Scoring.py", label="Batch Scoring — upload a CSV, download decisions for every customer", icon="📂")
        st.page_link("pages/3_Market_Intelligence.py", label="Market Intelligence — telecom/macro tickers → stress index", icon="📈")
        st.page_link("pages/4_Model_Insights.py", label="Model Insights — comparison, drivers, calibration, cut-off trade-off", icon="📊")
        st.page_link("pages/5_Business_Simulator.py", label="Business Simulator — change fee / recovery / cut-off, see monthly profit", icon="💰")
        st.page_link("pages/6_About_Developer.py", label="About & Contact — developer profile and contact form", icon="👨‍💻")

    with t4:
        left, right = st.columns(2)
        with left:
            st.markdown(
                """
**Stack**
- **Modelling:** scikit-learn — Logistic Regression, Random Forest, HistGradientBoosting
- **App:** Streamlit multi-page, Plotly
- **Market data:** Yahoo Finance via `yfinance` (cached 1 h)
- **Deployment:** GitHub → Streamlit Community Cloud
"""
            )
        with right:
            st.code(
                """app.py                  Home + recommendations
pages/                  Scorer, Batch, Market, Insights,
                        Simulator, About & Contact
src/preprocess.py       Cleaning + feature engineering
src/model_utils.py      Loading, scoring, tiers, reasons
src/market.py           Yahoo Finance + stress index
src/recommendations.py  Results → management actions
src/project_info.py     This project overview section
train_model.py          Retrains models/ and data/
models/  data/          Committed artefacts + sample file""",
                language="text",
            )

    with t5:
        st.warning(
            "Treat this as a decision-support demo, not a production credit engine.\n\n"
            f"- **Old, narrow data:** loans are from 2016 ({ins['date_from']} to {ins['date_to']}) and one telecom circle — "
            "retrain on current data before real lending.\n"
            "- **Market overlay is rule-based:** Yahoo Finance holds no customer data; the 0 / 1.5 / 3-point tightening is a transparent "
            "judgement to calibrate against your own loss history.\n"
            "- **\"Default\" = not repaid within 5 days:** late payers may still repay (see the recovery slider above).\n"
            "- **Label maturity should be confirmed** with the data owner."
        )
