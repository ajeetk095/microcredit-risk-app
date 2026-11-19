"""Live market context from Yahoo Finance (via yfinance).

IMPORTANT: this is a *macro risk overlay*, not part of the trained model. Yahoo Finance has no
customer-level data. We use market stress (volatility, index drawdown, INR weakness, telecom
and bank sector momentum) to tighten or relax the approval cut-off - a rule-based judgement that
the risk team should calibrate against its own loss history."""
import numpy as np
import pandas as pd
import streamlit as st

TELECOM = {"Bharti Airtel": "BHARTIARTL.NS", "Vodafone Idea": "IDEA.NS",
           "Reliance (Jio)": "RELIANCE.NS", "Indus Towers": "INDUSTOWER.NS"}
MACRO = {"Nifty 50": "^NSEI", "Nifty Bank": "^NSEBANK", "India VIX": "^INDIAVIX", "USD/INR": "USDINR=X"}
ALL = {**TELECOM, **MACRO}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(period: str = "1y") -> pd.DataFrame:
    """Daily closes for all tickers. Returns an empty frame if Yahoo is unreachable."""
    try:
        import yfinance as yf
        raw = yf.download(list(ALL.values()), period=period, interval="1d", auto_adjust=True,
                          progress=False, group_by="column", threads=True)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        close = close.rename(columns={v: k for k, v in ALL.items()}).dropna(how="all")
        return close.ffill()
    except Exception:
        return pd.DataFrame()


def _ret(s: pd.Series, days: int):
    s = s.dropna()
    return np.nan if len(s) <= days else s.iloc[-1] / s.iloc[-days - 1] - 1


def stock_table(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in [c for c in ALL if c in prices.columns and c not in ("India VIX",)]:
        s = prices[name].dropna()
        if len(s) < 30:
            continue
        dd = (s / s.cummax() - 1).min()
        rows.append({"Name": name, "Last": s.iloc[-1], "1M %": _ret(s, 21) * 100, "3M %": _ret(s, 63) * 100,
                     "1Y %": (s.iloc[-1] / s.iloc[0] - 1) * 100,
                     "Volatility % (ann.)": s.pct_change().std() * np.sqrt(252) * 100, "Max drawdown %": dd * 100})
    return pd.DataFrame(rows).set_index("Name")


def stress_index(prices: pd.DataFrame) -> dict | None:
    """0-100 composite. Components are skipped (and weights renormalised) if a ticker is missing."""
    if prices.empty:
        return None
    comps = {}
    if "India VIX" in prices and prices["India VIX"].notna().any():
        v = prices["India VIX"].dropna().iloc[-1]
        comps["India VIX level"] = (np.clip((v - 12) / 13, 0, 1), 0.30, f"VIX {v:.1f}")
    if "Nifty 50" in prices:
        s = prices["Nifty 50"].dropna().iloc[-126:]
        dd = s.iloc[-1] / s.max() - 1
        comps["Nifty drawdown (6M high)"] = (np.clip(-dd / 0.15, 0, 1), 0.25, f"{dd*100:.1f}%")
    if "USD/INR" in prices:
        ch = _ret(prices["USD/INR"], 21)
        comps["INR weakness (1M)"] = (np.clip(ch / 0.03, 0, 1), 0.15, f"{ch*100:+.1f}%")
    tel = [c for c in TELECOM if c in prices.columns]
    if tel:
        r = np.nanmean([_ret(prices[c], 63) for c in tel])
        comps["Telecom basket (3M)"] = (np.clip(-r / 0.15, 0, 1), 0.15, f"{r*100:+.1f}%")
    if "Nifty Bank" in prices:
        r = _ret(prices["Nifty Bank"], 63)
        comps["Banks / credit proxy (3M)"] = (np.clip(-r / 0.15, 0, 1), 0.15, f"{r*100:+.1f}%")
    comps = {k: v for k, v in comps.items() if not np.isnan(v[0])}
    if not comps:
        return None
    w = sum(v[1] for v in comps.values())
    score = 100 * sum(v[0] * v[1] for v in comps.values()) / w
    regime, tighten = ("Calm", 0.0) if score < 33 else (("Elevated", 1.5) if score < 66 else ("Stressed", 3.0))
    return dict(score=float(score), regime=regime, tighten_pts=tighten, as_of=str(prices.index[-1].date()),
                components={k: dict(stress=float(v[0]), weight=v[1], reading=v[2]) for k, v in comps.items()})


@st.cache_data(ttl=3600, show_spinner=False)
def get_market():
    prices = fetch_prices()
    return prices, stress_index(prices)
