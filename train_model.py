"""Train the micro-credit delinquency model and write every artefact the app needs.

    python train_model.py --data Micro_delinquency_train_test_dataset_v2.csv

Fixes vs. the original hackathon script (team_2.py):
  * honest hold-out: split by customer (msisdn) - the original scored on its own training file
  * no PCA fitted separately on test data (that leaks / makes components incomparable)
  * corrupted values -> NaN instead of abs()/mean-replacement; tree model handles NaN natively
  * decision threshold chosen from loan economics, not the default 0.5
"""
import argparse, json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.preprocess import RAW_FEATURES, build_features, clean_raw, loan_principal

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="Micro_delinquency_train_test_dataset_v2.csv")
args = ap.parse_args()
OUT = Path("models"); OUT.mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
t0 = time.time()

raw = pd.read_csv(args.data, index_col=0)

# ---- label-maturity check -------------------------------------------------------
# In the supplied file every loan issued after ~25-Jul-2016 is label=1 (0.0% default for weeks), while earlier
# weeks run 15-19%. That is not a real portfolio behaviour - the most likely cause is labels that had not
# matured / were not yet recorded. Such rows teach the model that "everything repays", so we detect a trailing
# run of >=7 zero-default days and exclude it from training and evaluation.
_d = pd.to_datetime(raw["pdate"])
_daily = (1 - raw["label"]).groupby(_d.dt.normalize()).mean().sort_index()
_run = []
for _day in reversed(list(_daily.index)):
    if _daily[_day] == 0: _run.append(_day)
    else: break
censor = None
n_all, dr_all = len(raw), float(1 - raw["label"].mean())
if len(_run) >= 7:
    censor = min(_run)
    raw = raw[_d < censor]
    print(f"label-maturity filter: dropped {n_all-len(raw):,} loans issued on/after {censor.date()} (all label=1)")
y_succ = raw["label"].astype(int)          # 1 = repaid within 5 days
y_def = 1 - y_succ                         # 1 = DEFAULT  (what we want to catch)
clean = clean_raw(raw)
X = build_features(raw)
principal = loan_principal(clean)
groups = raw["msisdn"].astype(str)
print(f"rows={len(X):,} features={X.shape[1]} default-rate={y_def.mean():.3%}")

tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y_def, groups))
Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y_def.iloc[tr], y_def.iloc[te]

# ---------------- model comparison (target = default) ------------------------
cands = {
    "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                          LogisticRegression(max_iter=500, class_weight="balanced")),
    "Random Forest": make_pipeline(SimpleImputer(strategy="median"),
                                   RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=30,
                                                          n_jobs=-1, random_state=7)),
    "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                                  min_samples_leaf=60, l2_regularization=1.0,
                                                                  early_stopping=True, random_state=7),
}
rows, fitted = [], {}
for name, m in cands.items():
    m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; fitted[name] = (m, p)
    rows.append(dict(model=name, roc_auc=roc_auc_score(yte, p), pr_auc=average_precision_score(yte, p),
                     brier=brier_score_loss(yte, p)))
    print(rows[-1])
comp = pd.DataFrame(rows)
best = comp.sort_values("pr_auc", ascending=False).iloc[0]["model"]
model, p_te = fitted[best]
print("best:", best)

# ---------------- economics-driven threshold ---------------------------------
FEE, LGD = 0.20, 1.00     # 5->6 and 10->12 INR  => 20% fee ; assume full loss on default (conservative)
# expected profit per INR lent = (1-pd)*FEE - pd*LGD  >= 0  <=>  pd <= FEE/(FEE+LGD)
pd_cut = FEE / (FEE + LGD)                         # 16.67 %
low_cut = pd_cut / 2

def policy_pnl(pdef, ydef, princ, cut):
    approve = pdef <= cut
    pnl = np.where(ydef == 0, FEE * princ, -LGD * princ) * approve
    return pnl.sum(), approve.mean(), pnl

te_princ = principal[te]
pnl_all = np.where(yte.values == 0, FEE * te_princ, -LGD * te_princ).sum()
pnl_model, appr, _ = policy_pnl(p_te, yte.values, te_princ, pd_cut)
# which cut-off earns the most on the hold-out?
grid = np.round(np.arange(0.02, 0.60, 0.01), 2)
curve = [dict(cut=float(c), pnl=float(policy_pnl(p_te, yte.values, te_princ, c)[0]),
              approve_rate=float(policy_pnl(p_te, yte.values, te_princ, c)[1]),
              defaulters_caught=float(((p_te > c) & (yte.values == 1)).sum() / (yte.values == 1).sum()),
              precision=float(((p_te > c) & (yte.values == 1)).sum() / max((p_te > c).sum(), 1)))
         for c in grid]
best_cut = max(curve, key=lambda d: d["pnl"])["cut"]

pred_flag = (p_te > pd_cut).astype(int)
cls = dict(recall_defaulters=recall_score(yte, pred_flag), precision_defaulters=precision_score(yte, pred_flag),
           f1_defaulters=f1_score(yte, pred_flag), approve_rate=float(appr),
           accuracy=float((pred_flag == yte.values).mean()))

# ---------------- insights for the Home page ---------------------------------
dfi = clean.assign(default=y_def.values, pdef=np.nan)
def seg(col, bins, labels):
    g = dfi.groupby(pd.cut(dfi[col], bins, labels=labels, include_lowest=True), observed=True)["default"].agg(["mean", "size"])
    return [dict(segment=str(i), default_rate=float(r["mean"]), customers=int(r["size"])) for i, r in g.iterrows()]
insights = dict(
    n_rows=int(len(raw)), n_customers=int(groups.nunique()), default_rate=float(y_def.mean()),
    date_from=str(pd.to_datetime(raw.pdate).min().date()), date_to=str(pd.to_datetime(raw.pdate).max().date()),
    by_recharges30=seg("cnt_ma_rech30", [-1, 0, 1, 2, 4, 8, 1e4], ["0", "1", "2", "3-4", "5-8", "9+"]),
    by_loans30=seg("cnt_loans30", [-1, 1, 2, 4, 8, 1e4], ["0-1", "2", "3-4", "5-8", "9+"]),
    by_tenure=seg("aon", [-1, 180, 365, 730, 1e5], ["<6m", "6-12m", "1-2y", "2y+"]),
    by_last_recharge_days=seg("last_rech_date_ma", [-1, 3, 7, 30, 1e4], ["0-3d", "4-7d", "8-30d", ">30d"]),
    by_payback=seg("payback30", [-1, 0, 2, 5, 1e4], ["0d", "1-2d", "3-5d", ">5d"]),
    pct_no_recharge_30d=float((clean.cnt_ma_rech30 == 0).mean()),
    pct_loan_10=float((principal == 10).mean()),
    label_censor=dict(applied=censor is not None, from_date=str(censor.date()) if censor is not None else None,
                      rows_dropped=int(n_all - len(raw)), rows_total=int(n_all), default_rate_all_rows=dr_all,
                      default_rate_matured=float(y_def.mean())),
    data_quality=dict(corrupted_or_missing_cells=int(clean.isna().sum().sum()), total_cells=int(clean.size)),
)


# ---------------- out-of-time check: train on earlier loans, test on later ones ---
dates = pd.to_datetime(raw["pdate"])
otr = (dates < pd.Timestamp("2016-07-08")).values
m_oot = clone(model).fit(X[otr], y_def[otr])
p_oot = m_oot.predict_proba(X[~otr])[:, 1]; y_oot = y_def[~otr].values
oot = dict(split_date="2016-07-08", n_train=int(otr.sum()), n_test=int((~otr).sum()),
           default_rate_train=float(y_def[otr].mean()), default_rate_test=float(y_oot.mean()),
           roc_auc=float(roc_auc_score(y_oot, p_oot)), pr_auc=float(average_precision_score(y_oot, p_oot)),
           recall_defaulters=float(recall_score(y_oot, p_oot > pd_cut)),
           precision_defaulters=float(precision_score(y_oot, p_oot > pd_cut)))
print("out-of-time:", oot)

# permutation importance on a hold-out sample (grouped to raw features via name)
samp = Xte.sample(min(15000, len(Xte)), random_state=1)
pi = permutation_importance(model, samp, yte.loc[samp.index], scoring="roc_auc", n_repeats=3, random_state=1, n_jobs=-1)
imp = (pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean}).sort_values("importance", ascending=False).head(15))

# calibration deciles
q = pd.qcut(p_te, 10, duplicates="drop")
calib = pd.DataFrame({"p": p_te, "y": yte.values}).groupby(q, observed=True).mean().reset_index(drop=True)

metrics = dict(
    best_model=best, comparison=comp.to_dict("records"),
    roc_auc=float(roc_auc_score(yte, p_te)), pr_auc=float(average_precision_score(yte, p_te)),
    classification_at_economic_cut=cls, econ=dict(fee=FEE, lgd=LGD, pd_cut=pd_cut, low_cut=low_cut, best_cut=best_cut),
    holdout=dict(n=int(len(yte)), default_rate=float(yte.mean()),
                 pnl_approve_all=float(pnl_all), pnl_model=float(pnl_model),
                 uplift=float(pnl_model - pnl_all)),
    cut_curve=curve, calibration=calib.rename(columns={"p": "predicted", "y": "actual"}).to_dict("records"),
    importance=imp.to_dict("records"), out_of_time=oot, trained_at=time.strftime("%Y-%m-%d %H:%M"),
)
bundle = dict(model=model, features=list(X.columns), medians=clean.median().to_dict(), metrics_name=best)
joblib.dump(bundle, OUT / "model.joblib", compress=3)
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
(OUT / "insights.json").write_text(json.dumps(insights, indent=2))

# hold-out scores let the Business Simulator re-price the policy instantly
pd.DataFrame({"p_default": p_te.round(4), "default": yte.values, "principal": te_princ}).to_csv(
    "data/holdout_scores.csv", index=False)
# per-transaction predictions in the format the brief asks for (probability + 1/0 label)
pd.DataFrame({"transaction_id": raw.index[te], "pdate": raw["pdate"].iloc[te].values,
              "repay_probability": (1 - p_te).round(4), "predicted_label": (p_te <= pd_cut).astype(int),
              "actual_label": y_succ.iloc[te].values}).to_csv("data/holdout_predictions.csv", index=False)
# small demo file for the Batch Scoring page (identifiers masked)
demo = raw.drop(columns=["label"]).sample(3000, random_state=3).copy()
demo["msisdn"] = ["CUST%05d" % i for i in range(len(demo))]
demo.to_csv("data/sample_customers.csv", index=False)
print(json.dumps({k: metrics[k] for k in ["best_model", "roc_auc", "pr_auc", "classification_at_economic_cut", "holdout", "econ"]}, indent=2))
print(f"done in {time.time()-t0:.0f}s")
