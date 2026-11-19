import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from src import market, model_utils as mu, ui

st.set_page_config(page_title="Credit Scorer", page_icon="🎯", layout="wide")
ui.inject()
st.title("🎯 Credit Scorer")
st.caption("Enter a customer's telecom behaviour to get default risk, a decision and the reasons behind it.")

metrics, bundle = mu.load_json("metrics.json"), mu.load_bundle()
_, mkt = market.get_market()
med = bundle["medians"]

with st.sidebar:
    st.header("Policy")
    use_mkt = st.toggle("Apply live market overlay", value=bool(mkt), disabled=not mkt,
                        help="Tightens the cut-off when Yahoo Finance shows market stress")
    tighten = mkt["tighten_pts"] if (mkt and use_mkt) else 0.0
    pol = mu.policy(metrics, tighten)
    st.write(f"Decline above **{pol['high']*100:.1f}%** default risk")
    st.write(f"10 INR limit below **{pol['low']*100:.1f}%**")
    if mkt: st.caption(f"Market regime: {mkt['regime']} ({mkt['score']:.0f}/100)")

def num(col, container):
    label, help_ = mu.FIELD_INFO[col]
    if col in st.session_state:
        return container.number_input(label, help=help_ or None, key=col)
    v = float(med.get(col, 0) or 0)
    return container.number_input(label, value=round(v, 2), help=help_ or None, key=col)

preset = st.radio("Start from", ["Typical customer", "Risky example", "Excellent example"], horizontal=True)
P = {"Typical customer": {}, 
     "Risky example": dict(cnt_ma_rech30=0, sumamnt_ma_rech30=0, last_rech_date_ma=45, daily_decr30=300, daily_decr90=500, cnt_loans30=1, amnt_loans30=6, cnt_loans90=1, payback30=0, payback90=0, aon=120),
     "Excellent example": dict(cnt_ma_rech30=9, sumamnt_ma_rech30=15000, last_rech_date_ma=1, daily_decr30=5000, daily_decr90=4800, cnt_loans30=6, amnt_loans30=48, cnt_loans90=9, payback30=1.5, payback90=1.8, aon=900)}[preset]
if st.session_state.get("_preset") != preset:
    for k_, v_ in P.items(): st.session_state[k_] = float(v_)
    st.session_state["_preset"] = preset

cols = st.columns(3)
vals = {}
for i, col in enumerate(mu.PRIMARY):
    vals[col] = num(col, cols[i % 3])
with st.expander("More fields (default to portfolio medians)"):
    rest = [c for c in mu.FIELD_INFO if c not in mu.PRIMARY]
    cc = st.columns(3)
    for i, col in enumerate(rest):
        vals[col] = num(col, cc[i % 3])

row = pd.DataFrame([{**{c: med.get(c, np.nan) for c in mu.FIELD_INFO}, **vals}])
base, reasons = mu.reason_codes(row)
t = mu.tier(base, pol)
principal = 10.0 if vals["maxamnt_loans90"] >= 12 else 5.0
ep = mu.expected_profit(base, principal, metrics["econ"]["fee"], metrics["econ"]["lgd"])

st.divider()
a, b = st.columns([1, 1.2])
with a:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=base * 100, number={"suffix": "%"},
        title={"text": "Probability of default"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#34495e"},
               "steps": [{"range": [0, pol["low"] * 100], "color": "#27ae60"},
                         {"range": [pol["low"] * 100, pol["high"] * 100], "color": "#f39c12"},
                         {"range": [pol["high"] * 100, 100], "color": "#c0392b"}]}))
    fig.update_layout(height=300, margin=dict(t=60, b=10, l=20, r=20))
    st.plotly_chart(fig, width="stretch")
with b:
    colr = {"Low": "#1e8449", "Medium": "#d68910", "High": "#c0392b"}[t]
    st.markdown(f"### Risk tier: {ui.pill(t, colr)}", unsafe_allow_html=True)
    st.markdown(f"**Recommended action:** {mu.ACTION[t]}")
    m1, m2 = st.columns(2)
    m1.metric("Probability of repayment (5 days)", f"{(1-base)*100:.1f}%")
    m2.metric("Predicted label", "1 - repays" if base <= pol["high"] else "0 - delinquent")
    st.metric("Expected profit on a typical loan", f"{ep:+.2f} INR",
              f"{'profitable' if ep > 0 else 'loss-making'} at {principal:.0f} INR principal", delta_color="off")
    st.markdown("**Why this score** (effect of moving each input to the portfolio median):")
    for c_, d in reasons:
        lab = mu.FIELD_INFO[c_][0]
        st.write(f"{'🔺' if d > 0 else '🔻'} **{lab}** = {vals[c_]:,.1f} → {'raises' if d > 0 else 'lowers'} risk by ~{abs(d)*100:.1f} pts")
