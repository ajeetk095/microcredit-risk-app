import pandas as pd, plotly.express as px, streamlit as st
from src import market, model_utils as mu, ui
from src.preprocess import RAW_FEATURES, clean_raw, loan_principal

st.set_page_config(page_title="Batch Scoring", page_icon="📂", layout="wide")
ui.inject()
st.title("📂 Batch Scoring")
st.caption("Output per loan: probability of repayment within 5 days, label (1 = repays, 0 = delinquent), risk tier and action. Upload a CSV of customers (same columns as the training data, `label` not needed) and download risk decisions.")

metrics = mu.load_json("metrics.json")
_, mkt = market.get_market()
tighten = mkt["tighten_pts"] if mkt else 0.0
pol = mu.policy(metrics, tighten)
e = metrics["econ"]

up = st.file_uploader("Customer CSV", type="csv")
use_demo = st.checkbox("Use the bundled demo file (3,000 customers)", value=up is None)
df = pd.read_csv(up) if up is not None else (mu.load_sample() if use_demo else None)
if df is None:
    st.stop()
missing = [c for c in RAW_FEATURES if c not in df.columns]
if missing:
    st.warning(f"{len(missing)} expected columns are missing and will be treated as unknown: {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")

p = mu.predict_default(df)
clean = clean_raw(df)
out = df.copy()
out["repay_probability"] = (1 - p).round(4)
out["p_default"] = p.round(4)
out["predicted_label"] = (p <= pol["high"]).astype(int)   # 1 = expected to repay within 5 days
out["risk_tier"] = [mu.tier(x, pol) for x in p]
out["action"] = out["risk_tier"].map(mu.ACTION)
out["expected_profit_inr"] = mu.expected_profit(p, loan_principal(clean), e["fee"], e["lgd"]).round(2)

k = st.columns(4)
k[0].metric("Customers", f"{len(out):,}")
k[1].metric("Average default risk", f"{p.mean()*100:.1f}%")
k[2].metric("Approve rate", f"{(out.risk_tier != 'High').mean()*100:.0f}%")
approved = out[out.risk_tier != "High"]
k[3].metric("Expected profit if policy applied", f"{approved.expected_profit_inr.sum():,.0f} INR",
            f"vs {out.expected_profit_inr.sum():,.0f} approving all", delta_color="off")

g = out.risk_tier.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
g.columns = ["tier", "customers"]
fig = px.bar(g, x="tier", y="customers", color="tier", template="plotly_white",
             color_discrete_map={"Low": "#27ae60", "Medium": "#f39c12", "High": "#c0392b"})
fig.update_layout(showlegend=False, height=320)
st.plotly_chart(fig, width="stretch")
if tighten:
    st.info(f"Live market overlay applied: cut-off tightened by {tighten:.1f} pts ({mkt['regime']} regime).")
show = [c for c in ["msisdn", "repay_probability", "predicted_label", "p_default", "risk_tier", "action", "expected_profit_inr"] if c in out.columns]
st.dataframe(out.sort_values("p_default", ascending=False)[show].head(200), width="stretch", height=360)
st.download_button("⬇️ Download all scored customers", out.to_csv(index=False).encode(), "scored_customers.csv", "text/csv")
