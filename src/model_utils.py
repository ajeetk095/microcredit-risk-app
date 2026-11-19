"""Loading the trained bundle, scoring, tiers and per-customer reason codes."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocess import build_features, clean_raw, engineer

ROOT = Path(__file__).resolve().parents[1]

# Friendly names + plain-English meaning for the inputs shown in the app
FIELD_INFO = {
    "aon": ("Age on network (days)", "How long the customer has been on the network"),
    "daily_decr30": ("Avg daily spend, 30d (INR)", "Average amount spent per day from main account"),
    "daily_decr90": ("Avg daily spend, 90d (INR)", "Average amount spent per day over 90 days"),
    "rental30": ("Avg main balance, 30d", "Average main-account balance, last 30 days"),
    "rental90": ("Avg main balance, 90d", "Average main-account balance, last 90 days"),
    "last_rech_date_ma": ("Days since last recharge", "Days till last recharge of the main account"),
    "last_rech_date_da": ("Days since last data recharge", ""),
    "last_rech_amt_ma": ("Last recharge amount", "Amount of the last main-account recharge"),
    "cnt_ma_rech30": ("Recharges in last 30d", "Number of main-account recharges, 30 days"),
    "fr_ma_rech30": ("Recharge frequency 30d", ""),
    "sumamnt_ma_rech30": ("Total recharged, 30d", "Total recharge amount, 30 days"),
    "medianamnt_ma_rech30": ("Median recharge, 30d", ""),
    "medianmarechprebal30": ("Median balance before recharge, 30d", "How low the balance gets before topping up"),
    "cnt_ma_rech90": ("Recharges in last 90d", ""),
    "fr_ma_rech90": ("Recharge frequency 90d", ""),
    "sumamnt_ma_rech90": ("Total recharged, 90d", ""),
    "medianamnt_ma_rech90": ("Median recharge, 90d", ""),
    "medianmarechprebal90": ("Median balance before recharge, 90d", ""),
    "cnt_da_rech30": ("Data recharges, 30d", ""), "fr_da_rech30": ("Data recharge frequency 30d", ""),
    "cnt_da_rech90": ("Data recharges, 90d", ""), "fr_da_rech90": ("Data recharge frequency 90d", ""),
    "cnt_loans30": ("Loans taken, 30d", "Number of micro-loans in the last 30 days"),
    "amnt_loans30": ("Loan amount, 30d (INR)", ""), "maxamnt_loans30": ("Max loan, 30d", ""),
    "medianamnt_loans30": ("Median loan, 30d", ""), "cnt_loans90": ("Loans taken, 90d", ""),
    "amnt_loans90": ("Loan amount, 90d (INR)", ""), "maxamnt_loans90": ("Max loan repay-amount, 90d", "6 = 5 INR loan, 12 = 10 INR loan"),
    "medianamnt_loans90": ("Median loan, 90d", ""),
    "payback30": ("Avg payback time, 30d (days)", "0 = no loan history in the window"),
    "payback90": ("Avg payback time, 90d (days)", ""),
}
PRIMARY = ["aon", "cnt_ma_rech30", "sumamnt_ma_rech30", "last_rech_date_ma", "last_rech_amt_ma",
           "daily_decr30", "daily_decr90", "rental30", "cnt_loans30", "amnt_loans30",
           "cnt_loans90", "payback30", "payback90", "medianmarechprebal30"]


@st.cache_resource(show_spinner=False)
def load_bundle():
    return joblib.load(ROOT / "models" / "model.joblib")


@st.cache_data(show_spinner=False)
def load_json(name: str):
    return json.loads((ROOT / "models" / name).read_text())


@st.cache_data(show_spinner=False)
def load_holdout():
    return pd.read_csv(ROOT / "data" / "holdout_scores.csv")


@st.cache_data(show_spinner=False)
def load_sample():
    return pd.read_csv(ROOT / "data" / "sample_customers.csv")


def predict_default(df_raw: pd.DataFrame) -> np.ndarray:
    b = load_bundle()
    X = build_features(df_raw)[b["features"]]
    return b["model"].predict_proba(X)[:, 1]


def policy(metrics: dict, tighten_pts: float = 0.0) -> dict:
    """Decision cut-offs on P(default). `tighten_pts` (in %-points) comes from the market overlay."""
    hi = max(metrics["econ"]["pd_cut"] - tighten_pts / 100.0, 0.03)
    return dict(high=hi, low=hi / 2)


def tier(p: float, pol: dict) -> str:
    return "Low" if p <= pol["low"] else ("Medium" if p <= pol["high"] else "High")


ACTION = {
    "Low": "APPROVE - offer up to 10 INR",
    "Medium": "APPROVE - 5 INR limit only",
    "High": "DECLINE / ask for a recharge first",
}


def expected_profit(p_def, principal, fee, lgd):
    return (1 - p_def) * fee * principal - p_def * lgd * principal


def reason_codes(raw_row: pd.DataFrame, n: int = 4):
    """Local sensitivity: how much does P(default) move if one input is set to the portfolio median?
    Positive delta = this input is pushing the customer's risk UP."""
    b = load_bundle()
    base = predict_default(raw_row)[0]
    out = []
    for c in PRIMARY:
        alt = raw_row.copy()
        alt[c] = b["medians"].get(c, np.nan)
        out.append((c, base - predict_default(alt)[0]))
    out.sort(key=lambda t: -abs(t[1]))
    return base, out[:n]
