import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from src import model_utils as mu, ui

st.set_page_config(page_title="Business Simulator", page_icon="💰", layout="wide")
ui.inject()
st.title("💰 Business Simulator")
st.caption("Re-price the lending policy on the 38k-customer hold-out. Change the assumptions and see profit move instantly.")
m, H = mu.load_json("metrics.json"), mu.load_holdout()

with st.sidebar:
    fee = st.slider("Fee on loan (%)", 5, 40, 20) / 100
    rec = st.slider("Recovery on defaulted loans (%)", 0, 80, 0,
                    help="Share of principal later recovered from customers flagged as 'not repaid in 5 days'") / 100
    lgd = 1 - rec
    be = fee / (fee + lgd)
    cut = st.slider("Approve if P(default) below (%)", 2, 60, int(round(be * 100))) / 100
    vol = st.number_input("Scale to loans per month", 1_000, 100_000_000, 1_000_000, step=100_000)

def pnl(c):
    appr = H.p_default <= c
    gain = np.where(H["default"] == 0, fee * H.principal, -lgd * H.principal) * appr
    return gain.sum(), appr.mean()

allp = np.where(H["default"] == 0, fee * H.principal, -lgd * H.principal).sum()
pm, ar = pnl(cut)
sc = vol / len(H)
k = st.columns(4)
k[0].metric("Break-even default rate", f"{be*100:.1f}%")
k[1].metric("Approve-all profit / month", f"{allp*sc:,.0f} INR")
k[2].metric("Model-policy profit / month", f"{pm*sc:,.0f} INR", f"{(pm-allp)*sc:+,.0f}")
k[3].metric("Approval rate", f"{ar*100:.0f}%")
grid = np.arange(0.02, 0.6, 0.01)
y = [pnl(c)[0] * sc for c in grid]
f = go.Figure(go.Scatter(x=grid * 100, y=y, mode="lines", name="Profit"))
f.add_vline(x=cut * 100, line_color="green", annotation_text="your cut-off")
f.add_hline(y=allp * sc, line_dash="dot", annotation_text="approve all")
f.update_layout(template="plotly_white", height=400, xaxis_title="Approve if P(default) below (%)", yaxis_title="Monthly profit (INR)")
st.plotly_chart(f, width="stretch")
if rec > 0:
    st.info("With recovery, losses shrink and the best cut-off rises - more customers are worth lending to.")
st.caption("Assumes the hold-out mix of 5 INR and 10 INR loans is representative. Simulation, not a forecast.")


