"""Shared data-cleaning + feature engineering.
Used by BOTH train_model.py and the Streamlit app, so training and
production always see exactly the same transformations."""
import numpy as np
import pandas as pd

# ---- raw numeric columns of the dataset (excluding label) ------------------
RAW_FEATURES = [
    "aon", "daily_decr30", "daily_decr90", "rental30", "rental90",
    "last_rech_date_ma", "last_rech_date_da", "last_rech_amt_ma",
    "cnt_ma_rech30", "fr_ma_rech30", "sumamnt_ma_rech30", "medianamnt_ma_rech30",
    "medianmarechprebal30", "cnt_ma_rech90", "fr_ma_rech90", "sumamnt_ma_rech90",
    "medianamnt_ma_rech90", "medianmarechprebal90", "cnt_da_rech30", "fr_da_rech30",
    "cnt_da_rech90", "fr_da_rech90", "cnt_loans30", "amnt_loans30", "maxamnt_loans30",
    "medianamnt_loans30", "cnt_loans90", "amnt_loans90", "maxamnt_loans90",
    "medianamnt_loans90", "payback30", "payback90",
]

# Plausible upper limits. The raw file contains ~0.1-1% corrupted values
# (e.g. 999,520.6 days of tenure, text such as 'UA') -> treated as missing.
HARD_MAX = {
    "aon": 20000, "last_rech_date_ma": 1000, "last_rech_date_da": 1000,
    "fr_ma_rech30": 1000, "fr_da_rech30": 1000, "cnt_da_rech30": 1000,
    "maxamnt_loans30": 12, "cnt_loans90": 1000, "medianmarechprebal30": 50000,
    "medianmarechprebal90": 50000, "daily_decr30": 1e6, "daily_decr90": 1e6,
}
# Columns where a negative number is impossible (balances/rental CAN be negative)
NON_NEGATIVE = ["aon", "daily_decr30", "daily_decr90", "last_rech_date_ma",
                "last_rech_date_da"]


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric, blank out impossible values (kept as NaN, not abs()-ed)."""
    out = pd.DataFrame(index=df.index)
    for c in RAW_FEATURES:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    for c, mx in HARD_MAX.items():
        out.loc[out[c] > mx, c] = np.nan
    for c in NON_NEGATIVE:
        out.loc[out[c] < 0, c] = np.nan
    return out


def _ratio(a, b):
    return a / (b + 1.0)


def engineer(clean: pd.DataFrame) -> pd.DataFrame:
    """Business-meaningful ratio features on top of the raw columns."""
    f = clean.copy()
    f["rech_trend"] = _ratio(f.cnt_ma_rech30, f.cnt_ma_rech90)          # recent vs 90d activity
    f["spend_trend"] = _ratio(f.daily_decr30, f.daily_decr90)
    f["loan_share_30_90"] = _ratio(f.cnt_loans30, f.cnt_loans90)
    f["loan_intensity"] = _ratio(f.cnt_loans30, f.cnt_ma_rech30)         # loans per recharge
    f["recharge_cover"] = _ratio(f.sumamnt_ma_rech30, f.amnt_loans30)    # recharge money per loan INR
    f["balance_to_spend"] = _ratio(f.rental30, f.daily_decr30)
    f["avg_loan_size30"] = f.amnt_loans30 / f.cnt_loans30.replace(0, np.nan)
    f["no_recharge_30d"] = (f.cnt_ma_rech30 == 0).astype(float)
    return f


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return engineer(clean_raw(df))


def loan_principal(df_clean: pd.DataFrame):
    """Users can borrow 5 INR (repay 6) or 10 INR (repay 12). Infer from history."""
    return np.where(df_clean["maxamnt_loans90"].fillna(0) >= 12, 10.0, 5.0)
