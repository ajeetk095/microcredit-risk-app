import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from src import model_utils as mu, ui

st.set_page_config(page_title="Model Insights", page_icon="📊", layout="wide")
ui.inject()
st.title("📊 Model Insights")
m, ins = mu.load_json("metrics.json"), mu.load_json("insights.json")
e = m["econ"]

st.subheader("Model comparison (customer-level hold-out)")
cmp_ = pd.DataFrame(m["comparison"]).set_index("model")
c1, c2 = st.columns([1, 1])
c1.dataframe(cmp_.style.format("{:.3f}").highlight_max(subset=["roc_auc", "pr_auc"], color="#c8e6c9").highlight_min(subset=["brier"], color="#c8e6c9"), width="stretch")
c2.write(f"**Selected:** {m['best_model']} - best precision-recall AUC, which matters most with only {ins['default_rate']*100:.1f}% defaulters. Trained {m['trained_at']}.")
c2.caption("Original hackathon pipeline (PCA + Random Forest, scored on its own training file) was replaced to remove leakage; see README.")

a, b = st.columns(2)
imp = pd.DataFrame(m["importance"]).sort_values("importance")
a.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h", template="plotly_white", title="What drives default risk (permutation importance, AUC drop)").update_layout(height=480), width="stretch")
cal = pd.DataFrame(m["calibration"])
f = go.Figure([go.Scatter(x=cal.predicted, y=cal.actual, mode="lines+markers", name="Model"),
               go.Scatter(x=[0, cal.predicted.max()], y=[0, cal.predicted.max()], mode="lines", name="Perfect", line=dict(dash="dash"))])
f.update_layout(template="plotly_white", title="Calibration: predicted vs actual default (deciles)", xaxis_title="Predicted", yaxis_title="Actual", height=480)
b.plotly_chart(f, width="stretch")

st.subheader("Cut-off trade-off")
cur = pd.DataFrame(m["cut_curve"])
f = go.Figure()
f.add_trace(go.Scatter(x=cur.cut * 100, y=cur.pnl, name="Profit (INR)", yaxis="y1"))
f.add_trace(go.Scatter(x=cur.cut * 100, y=cur.defaulters_caught * 100, name="Defaulters caught %", yaxis="y2"))
f.add_vline(x=e["pd_cut"] * 100, line_dash="dot", annotation_text="break-even")
f.update_layout(template="plotly_white", height=380, xaxis_title="Approve if P(default) below (%)",
                yaxis=dict(title="Hold-out profit (INR)"), yaxis2=dict(title="Defaulters caught %", overlaying="y", side="right"))
st.plotly_chart(f, width="stretch")
st.caption(f"Profit-maximising cut-off on the hold-out: {e['best_cut']*100:.0f}% - essentially the {e['pd_cut']*100:.1f}% theoretical break-even, which confirms the model is well calibrated.")

st.subheader("Default rate by customer segment")
tabs = st.tabs(["Recharges 30d", "Loans 30d", "Tenure", "Days since recharge", "Payback days"])
for t, k, ttl in zip(tabs, ["by_recharges30", "by_loans30", "by_tenure", "by_last_recharge_days", "by_payback"], [""] * 5):
    d = pd.DataFrame(ins[k]); d["default_rate"] *= 100
    fig = px.bar(d, x="segment", y="default_rate", text=d.default_rate.round(1).astype(str) + "%", template="plotly_white", hover_data=["customers"])
    fig.add_hline(y=e["pd_cut"] * 100, line_dash="dot", line_color="red", annotation_text="break-even")
    fig.update_layout(yaxis_title="Default rate %", height=340)
    t.plotly_chart(fig, width="stretch")

st.subheader("Does it hold up on future loans? (out-of-time test)")
o = m["out_of_time"]
x = st.columns(4)
x[0].metric("Trained on", f"before {o['split_date']}", f"{o['n_train']:,} loans", delta_color="off")
x[1].metric("Tested on", f"{o['split_date']} onward", f"{o['n_test']:,} loans", delta_color="off")
x[2].metric("ROC-AUC", f"{o['roc_auc']:.3f}", f"random split {m['roc_auc']:.3f}", delta_color="off")
x[3].metric("Defaulters caught", f"{o['recall_defaulters']*100:.0f}%", f"precision {o['precision_defaulters']*100:.0f}%", delta_color="off")
st.caption(f"Default rate rose from {o['default_rate_train']*100:.1f}% (train) to {o['default_rate_test']*100:.1f}% (test) - the model ranks risk well on later loans, but expect drift and monitor monthly.")

st.subheader("Operating points")
op = pd.DataFrame(m["cut_curve"]); op = op[op.cut.round(2).isin([0.08, 0.10, 0.12, 0.16, 0.20, 0.25, 0.30])].copy()
op = op.assign(**{"Approve if P(default) below": (op.cut * 100).round(0).astype(int).astype(str) + "%", "Approval rate": (op.approve_rate * 100).round(0).astype(int).astype(str) + "%",
                  "Defaulters caught": (op.defaulters_caught * 100).round(0).astype(int).astype(str) + "%", "Precision of declines": (op.precision * 100).round(0).astype(int).astype(str) + "%",
                  "Hold-out profit (INR)": op.pnl.round(0)})
st.dataframe(op[["Approve if P(default) below", "Approval rate", "Defaulters caught", "Precision of declines", "Hold-out profit (INR)"]], hide_index=True, width="stretch")
st.caption("A lower cut-off catches more defaulters (higher recall, the goal in the original team notes) but declines more good customers.")

st.subheader("Per-transaction predictions (hold-out)")
hp = pd.read_csv(mu.ROOT / "data" / "holdout_predictions.csv")
st.dataframe(hp.head(50), hide_index=True, width="stretch")
st.download_button("⬇️ Download hold-out predictions (probability + 1/0 label)", hp.to_csv(index=False).encode(), "holdout_predictions.csv", "text/csv")
