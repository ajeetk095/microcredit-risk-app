import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from src import market, ui

st.set_page_config(page_title="Market Intelligence", page_icon="📈", layout="wide")
ui.inject()
st.title("📈 Market Intelligence (Yahoo Finance)")
st.caption("Live telecom-sector and macro context used as a risk overlay on lending decisions. Data refreshes hourly.")

if st.button("🔄 Refresh now"):
    st.cache_data.clear()
prices, mkt = market.get_market()
if prices.empty or not mkt:
    st.error("Could not reach Yahoo Finance (network blocked or rate-limited). The app falls back to the standard cut-off. Try again in a minute.")
    st.stop()

col = ui.REGIME[mkt["regime"]]
a, b = st.columns([1, 2])
with a:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=mkt["score"], title={"text": "Market stress index"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#34495e"},
               "steps": [{"range": [0, 33], "color": "#27ae60"}, {"range": [33, 66], "color": "#f39c12"}, {"range": [66, 100], "color": "#c0392b"}]}))
    fig.update_layout(height=280, margin=dict(t=60, b=10))
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"Regime: {ui.pill(mkt['regime'], col)} &nbsp; as of {mkt['as_of']}", unsafe_allow_html=True)
with b:
    comp = pd.DataFrame(mkt["components"]).T.reset_index().rename(columns={"index": "Component", "stress": "Stress (0-1)", "weight": "Weight", "reading": "Reading"})
    st.dataframe(comp, hide_index=True, width="stretch")
    st.markdown({"Calm": "**Implication:** standard approval cut-off. Good moment to grow the 10 INR product among Low-tier customers.",
                 "Elevated": f"**Implication:** cut-off tightened by {mkt['tighten_pts']} pts; watch the Medium tier and slow limit increases.",
                 "Stressed": f"**Implication:** cut-off tightened by {mkt['tighten_pts']} pts; restrict to 5 INR loans, raise provisions, review weekly."}[mkt["regime"]])

st.subheader("Telecom stocks vs market (rebased to 100)")
names = st.multiselect("Series", [c for c in prices.columns if c != "India VIX"],
                       default=[c for c in market.TELECOM if c in prices.columns] + [c for c in ["Nifty 50"] if c in prices.columns])
if names:
    rb = prices[names].dropna(how="all"); rb = rb / rb.bfill().iloc[0] * 100
    st.plotly_chart(px.line(rb, template="plotly_white").update_layout(height=380, legend_title=""), width="stretch")
st.subheader("Performance & risk table")
st.dataframe(market.stock_table(prices).round(1), width="stretch")
if "India VIX" in prices:
    st.plotly_chart(px.area(prices["India VIX"].dropna(), template="plotly_white", title="India VIX (market fear gauge)").update_layout(height=260, showlegend=False), width="stretch")
st.info("**How to use this in real life:** the sector/macro stress index changes *how strict* you are, not *who* is risky. "
        "Back-test the thresholds against your own monthly default rates before relying on them. Source: Yahoo Finance via yfinance (informational, not investment advice).")
