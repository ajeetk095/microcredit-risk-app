import streamlit as st
from src import market, model_utils as mu, recommendations as rc, ui

st.set_page_config(page_title="Micro-Credit Risk Hub", page_icon="📶", layout="wide")
ui.inject()

metrics, ins = mu.load_json("metrics.json"), mu.load_json("insights.json")
with st.spinner("Loading live market context from Yahoo Finance..."):
    prices, mkt = market.get_market()

st.title("📶 Micro-Credit Risk Hub")
st.caption("Telecom airtime-loan delinquency prediction · decisions, recommendations and live market context in one place")

c = metrics["classification_at_economic_cut"]; h = metrics["holdout"]; e = metrics["econ"]
k = st.columns(5)
k[0].metric("Default rate (matured loans)", f"{ins['default_rate']*100:.1f}%", f"break-even {e['pd_cut']*100:.1f}%", delta_color="off")
k[1].metric("Model ROC-AUC", f"{metrics['roc_auc']:.3f}", metrics["best_model"], delta_color="off")
k[2].metric("Defaulters caught", f"{c['recall_defaulters']*100:.0f}%", f"precision {c['precision_defaulters']*100:.0f}%", delta_color="off")
k[3].metric("Hold-out profit (model policy)", f"{h['pnl_model']:,.0f} INR", f"approve-all: {h['pnl_approve_all']:,.0f} INR", delta_color="off")
if mkt:
    k[4].metric("Market stress (live)", f"{mkt['score']:.0f}/100", mkt["regime"], delta_color="off")
else:
    k[4].metric("Market stress (live)", "n/a", "Yahoo unreachable", delta_color="off")

with st.expander("📋 Business problem this app addresses", expanded=False):
    st.markdown("""
A telecom partners with a microfinance institution (MFI) to lend airtime on mobile balances, repayable within **5 days**.
A customer is **delinquent** if they do not repay in time. For **each loan transaction** the model outputs the
**probability of repayment** and a **label (1 = repays within 5 days, 0 = delinquent)**, so credit can be granted, limited or withheld
- extending mobile financial services to unbanked customers without taking on unaffordable losses.
""")
lc = ins.get("label_censor", {})
if lc.get("applied"):
    st.warning(f"Data check: {lc['rows_dropped']:,} loans issued on/after {lc['from_date']} show 0% default (label never 0) and were excluded as unmatured labels. "
               f"Default rate on matured loans is {lc['default_rate_matured']*100:.1f}% vs {lc['default_rate_all_rows']*100:.1f}% if they were counted.")

st.subheader("Recommendations for management")
st.caption("Generated automatically from the model results, the portfolio data and the live market regime.")
recs = rc.build(metrics, ins, mkt)
left, right = st.columns(2)
for i, r in enumerate(recs):
    with (left if i % 2 == 0 else right):
        ui.rec_card(r)

st.subheader("What would you like to do?")
n = st.columns(4)
n[0].page_link("pages/1_Credit_Scorer.py", label="🎯 Score one customer", width="stretch")
n[1].page_link("pages/2_Batch_Scoring.py", label="📂 Score a customer file", width="stretch")
n[2].page_link("pages/3_Market_Intelligence.py", label="📈 Market intelligence", width="stretch")
n[3].page_link("pages/5_Business_Simulator.py", label="💰 Profit simulator", width="stretch")

with st.expander("About this model and its limits"):
    st.markdown(f"""
- **Target:** probability a customer will **not** repay a 5/10 INR airtime loan within 5 days (default = 1).
- **Data:** {ins['n_rows']:,} matured loan records from {ins['n_customers']:,} customers, {ins['date_from']} to {ins['date_to']} (one telecom circle).
- **Evaluation:** hold-out split *by customer* so no customer appears in both training and testing.
- **Economics:** a 5 INR loan returns 6 and a 10 INR loan returns 12 (20% fee). A loan is profitable only if P(default) < **{e['pd_cut']*100:.1f}%** when a default loses the full principal.
- **Market overlay:** Yahoo Finance gives sector/macro stress, not customer data. It adjusts the cut-off by a transparent rule and is **not** learned from the data.
- **Caveats:** data is from 2016; retrain on current data before real lending decisions. Scores support, not replace, credit-policy judgement.
""")
