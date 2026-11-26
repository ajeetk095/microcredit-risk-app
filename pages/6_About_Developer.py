"""👨‍💻 About Developer & Contact — profile, project showcase and contact form."""
import base64
import logging
import re
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import streamlit as st

from src import model_utils as mu, ui

st.set_page_config(page_title="About & Contact", page_icon="👨‍💻", layout="wide")
ui.inject()
st.title("👨‍💻 About & Contact")
st.caption("The developer behind the Micro-Credit Risk Hub, how the project was built, and how to get in touch.")

NAME = "Ajeet Kumar"
EMAIL = "ajeetk095@gmail.com"
LINKEDIN = "https://www.linkedin.com/in/ajeek095/"
GITHUB = "https://github.com/ajeetk095"
REPO = "https://github.com/ajeetk095/microcredit-risk-app"  # update if your repo name differs
PHOTO_PATH = Path(__file__).resolve().parent.parent / "assets" / "ajeet_img.png"  # developer photo

metrics, ins = mu.load_json("metrics.json"), mu.load_json("insights.json")
e, h, c, oot = metrics["econ"], metrics["holdout"], metrics["classification_at_economic_cut"], metrics["out_of_time"]
lc = ins.get("label_censor", {})


def avatar_html() -> str:
    """Developer photo from assets/ (embedded as base64), with an initials badge as fallback."""
    if PHOTO_PATH.exists():
        b64 = base64.b64encode(PHOTO_PATH.read_bytes()).decode()
        return (f'<img class="avatar" src="data:image/png;base64,{b64}" alt="{NAME}" '
                'style="width:150px;height:150px;min-width:150px;aspect-ratio:1/1;border-radius:50%;'
                'object-fit:cover;object-position:center;display:block;margin:0 auto .8rem auto;'
                'border:3px solid #1f6feb;box-shadow:0 2px 8px rgba(0,0,0,.25)">')
    initials = "".join(w[0] for w in NAME.split()[:2]).upper()
    return f'<div class="avatar-fallback">{initials}</div>'


# ───────────────────────────── Profile + contact ─────────────────────────────
profile_col, contact_col = st.columns(2, gap="large")

with profile_col:
    st.subheader("About the Developer")
    st.markdown(
        f"""<div class="info-card">{avatar_html()}<h3>{NAME}</h3>
        <div class="section-note">Quantitative Finance Professional<br>Bangalore, India</div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"Hi, I'm **{NAME}** — a quantitative finance professional based in Bangalore. I hold a "
        "**B.Tech in IT** (AKTU), an **M.Tech in CSE** (NSUT Delhi) and an **M.Sc. in Financial "
        "Engineering** from **WorldQuant University**.\n\n"
        "I apply quantitative methods and machine learning to real-world finance problems. The "
        "**Micro-Credit Risk Hub** is a worked example: it takes a telecom airtime-loan delinquency "
        "brief, builds a leakage-free model, converts probabilities into loan economics and "
        "approve / limit / decline decisions, and layers on a live market-stress view.\n\n"
        "I am seeking a full-time role in **quantitative research or financial engineering**."
    )
    b1, b2, b3 = st.columns(3)
    b1.link_button("📧 Email", f"mailto:{EMAIL}", width="stretch")
    b2.link_button("💼 LinkedIn", LINKEDIN, width="stretch")
    b3.link_button("🐙 GitHub", GITHUB, width="stretch")
    st.link_button("📦 View this project's source code", REPO, width="stretch")

with contact_col:
    st.subheader("Contact")
    st.caption("Send a message about opportunities, research collaborations, or feedback on this app.")

    with st.form("contact_form", clear_on_submit=False):
        name = st.text_input("Name")
        email = st.text_input("Email")
        subject = st.text_input("Subject")
        message = st.text_area("Message", height=180)
        submitted = st.form_submit_button("Send Message", width="stretch")

    if submitted:
        errors = []
        if len(name.strip()) < 2:
            errors.append("Please enter your name (min. 2 characters).")
        if "\r" in name or "\n" in name:
            errors.append("Name cannot contain line breaks.")
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()):
            errors.append("Please enter a valid email address.")
        if len(subject.strip()) < 3:
            errors.append("Subject must be at least 3 characters.")
        if "\r" in subject or "\n" in subject:
            errors.append("Subject cannot contain line breaks.")
        if len(message.strip()) < 10:
            errors.append("Message must be at least 10 characters.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            try:
                cfg = st.secrets.get("email", {})
                host = cfg.get("smtp_host", "smtp.gmail.com")
                port = int(cfg.get("smtp_port", 465))
                user, pwd = cfg.get("username", ""), cfg.get("password", "")
                to_addr = cfg.get("to_address", EMAIL)
                if not user or not pwd:
                    raise RuntimeError("Contact email credentials are not configured.")

                mail = EmailMessage()
                mail["From"] = user
                mail["To"] = to_addr
                mail["Reply-To"] = email.strip()
                mail["Subject"] = f"[Micro-Credit Risk Hub] {subject.strip()}"
                mail.set_content(
                    "New contact form message\n\n"
                    f"Name: {name.strip()}\nEmail: {email.strip()}\nSubject: {subject.strip()}\n\n"
                    f"Message:\n{message.strip()}"
                )
                with smtplib.SMTP_SSL(host, port, timeout=20) as smtp:
                    smtp.login(user, pwd)
                    smtp.send_message(mail)
                st.success("✅ Your message was sent successfully. Thank you!")
            except Exception:
                logging.exception("Failed to send contact form email")
                st.error(f"We couldn't send your message right now. Please try again later or email {EMAIL} directly.")

    st.markdown(
        '<div class="section-note">Typical response time: within 24–48 hours. Please include the '
        "role or collaboration context in your message.</div>",
        unsafe_allow_html=True,
    )

# ───────────────────────────── Project showcase ─────────────────────────────
st.divider()
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
train_model.py          Retrains models/ and data/
models/  data/          Committed artefacts + sample file""",
            language="text",
        )

with t5:
    st.warning(
        "Treat this as a decision-support demo, not a production credit engine.\n\n"
        "- **Old, narrow data:** loans are from 2016 ("
        f"{ins['date_from']} to {ins['date_to']}) and one telecom circle — retrain on current data before real lending.\n"
        "- **Market overlay is rule-based:** Yahoo Finance holds no customer data; the 0 / 1.5 / 3-point tightening is a transparent "
        "judgement to calibrate against your own loss history.\n"
        "- **\"Default\" = not repaid within 5 days:** late payers may still repay (see the recovery slider in the Business Simulator).\n"
        "- **Label maturity should be confirmed** with the data owner."
    )
