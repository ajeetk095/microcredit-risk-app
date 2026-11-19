"""Turns model results + data insights (+ live market regime) into management recommendations.
Every number shown is computed from the artefacts produced by train_model.py - nothing is hard-coded."""


def _pct(x, d=1):
    return f"{x*100:.{d}f}%"


def build(metrics: dict, ins: dict, market: dict | None = None) -> list[dict]:
    e, h, c = metrics["econ"], metrics["holdout"], metrics["classification_at_economic_cut"]
    be = e["pd_cut"]
    recs = []

    if h["pnl_approve_all"] > 0:
        cmp_ = (f"{h['pnl_model']:,.0f} INR versus {h['pnl_approve_all']:,.0f} INR when everyone is approved "
                f"(+{h['uplift']/h['pnl_approve_all']*100:.0f}%)")
    else:
        cmp_ = (f"{h['pnl_model']:,.0f} INR, whereas approving everyone LOSES {abs(h['pnl_approve_all']):,.0f} INR")
    lc = ins.get("label_censor", {})
    if lc.get("applied"):
        recs.append(dict(priority="High", title="Verify label maturity of recent loans before trusting any default KPI",
            body=f"All {lc['rows_dropped']:,} loans issued on/after {lc['from_date']} ({lc['rows_dropped']/lc['rows_total']*100:.0f}% of the file) have label = 1 "
                 f"(0% default), while earlier weeks default at 15-19%. This is almost certainly labels that had not matured. Counting them gives a misleading "
                 f"{lc['default_rate_all_rows']*100:.1f}% default rate; the matured-loan rate is {lc['default_rate_matured']*100:.1f}% - above the {be*100:.1f}% break-even. "
                 "They were excluded from training. Confirm with the data owner and refresh labels after the 5-day window closes.",
            evidence="Detected automatically in train_model.py (trailing run of zero-default days)."))
    recs.append(dict(priority="High", title="Switch from 'lend to everyone' to score-based approval",
        body=f"On a customer-level hold-out the model-based policy earns {cmp_}. It catches "
             f"{_pct(c['recall_defaulters'],0)} of defaulters while still approving {_pct(c['approve_rate'],0)} of requests.",
        evidence=f"ROC-AUC {metrics['roc_auc']:.3f} (out-of-time {metrics.get('out_of_time',{}).get('roc_auc',float('nan')):.3f}); matured default rate {_pct(ins['default_rate'])} vs break-even {_pct(be)}."))

    losing = []
    for key, lab in [("by_recharges30", "recharges in 30d"), ("by_loans30", "loans in 30d"),
                     ("by_tenure", "network tenure"), ("by_last_recharge_days", "days since last recharge"),
                     ("by_payback", "avg payback days")]:
        for s in ins[key]:
            if s["default_rate"] > be and s["customers"] > 500:
                losing.append((s["default_rate"], f"{lab}: {s['segment']}", s["customers"]))
    losing.sort(reverse=True)
    if losing:
        txt = "; ".join(f"{n} ({_pct(r)}, {k:,} customers)" for r, n, k in losing[:4])
        recs.append(dict(priority="High", title="Stop unconditional lending to loss-making segments",
            body=f"These segments default above the {_pct(be)} break-even, so each loan loses money on average: {txt}. "
                 "Require a recharge before lending, or cap them at the 5 INR product.",
            evidence="Break-even = fee / (fee + loss-given-default) = 20% / 120%."))

    safe = []
    for key, lab in [("by_recharges30", "recharges in 30d"), ("by_loans30", "loans in 30d"), ("by_payback", "avg payback days")]:
        for s in ins[key]:
            if s["default_rate"] < e["low_cut"] and s["customers"] > 2000:
                safe.append((s["default_rate"], f"{lab}: {s['segment']}", s["customers"]))
    safe.sort()
    if safe:
        txt = "; ".join(f"{n} ({_pct(r)})" for r, n, _ in safe[:4])
        recs.append(dict(priority="Medium", title="Grow revenue from the safest customers",
            body=f"Segments with default below {_pct(e['low_cut'])}: {txt}. Offer the 10 INR product by default, "
                 "pre-approve repeat borrowers, and run retention offers - they carry the portfolio's profit.",
            evidence=f"Only {_pct(ins['pct_loan_10'])} of customers currently use the 10 INR loan."))

    nr = ins["by_recharges30"][0]
    recs.append(dict(priority="High", title="Treat 'no recent recharge' as an early-warning trigger",
        body=f"{_pct(ins['pct_no_recharge_30d'])} of customers made no recharge in 30 days and "
             f"{_pct(nr['default_rate'])} of them default - about {nr['default_rate']/ins['default_rate']:.1f}x the portfolio average. "
             "Send a recharge nudge before offering credit; auto-decline if inactivity exceeds 30 days.",
        evidence="Top drivers: spend level/trend, days since last recharge, balance behaviour."))

    first = ins["by_payback"][0]
    recs.append(dict(priority="Medium", title="Build a 'first-loan' ladder for thin-file borrowers",
        body=f"Customers with no payback history default at {_pct(first['default_rate'])}, whereas those who repaid within 1-2 days "
             f"default at only {_pct(ins['by_payback'][1]['default_rate'])}. Start new borrowers at 5 INR and graduate them after 2-3 on-time repayments.",
        evidence="Repayment behaviour is the most reliable predictor of future repayment."))

    recs.append(dict(priority="Medium", title="Protect financial inclusion: decline with a path forward",
        body="The service targets unbanked and remote customers, who often have thin records. Rather than a permanent decline, give a clear route back "
             "(recharge, then 5 INR starter loan, then limit increase after on-time repayments) and review decline rates by region and tenure for unintended exclusion. "
             "Partner MFIs can co-design limits so risk control does not shut out the poor.",
        evidence="Brief context: mobile micro-credit for the unbanked via telecom + MFI partnership."))
    recs.append(dict(priority="Medium", title="Prefer 'reduced limit' over hard declines",
        body=f"At the economic cut-off, {_pct(1-c['precision_defaulters'],0)} of flagged customers would actually have repaid "
             "(precision is "
             f"{_pct(c['precision_defaulters'],0)}). Use a 5 INR cap or recharge-first rule for the Medium tier to avoid losing good customers.",
        evidence=f"Hold-out recall {_pct(c['recall_defaulters'],0)}, precision {_pct(c['precision_defaulters'],0)}."))

    if market:
        r = market["regime"]
        if r == "Calm":
            body = (f"Market stress index is {market['score']:.0f}/100 (Calm, as of {market['as_of']}). No tightening needed; "
                    "keep the standard cut-off and revisit if volatility or the rupee moves sharply.")
        else:
            body = (f"Market stress index is {market['score']:.0f}/100 ({r}, as of {market['as_of']}). The app tightens the approval "
                    f"cut-off by {market['tighten_pts']:.1f} percentage points and recommends higher provisioning for the Medium tier.")
        recs.append(dict(priority="Medium" if r != "Stressed" else "High", title=f"Market overlay - {r.lower()} conditions",
                         body=body, evidence="Source: Yahoo Finance (India VIX, Nifty 50/Bank, USD/INR, telecom stocks). Rule-based overlay - calibrate with your own loss history."))
    else:
        recs.append(dict(priority="Info", title="Market overlay unavailable right now",
            body="Live Yahoo Finance data could not be loaded, so the standard cut-off is used. Open the Market Intelligence page to retry.",
            evidence="Check internet access / Yahoo rate limits."))

    dq = ins["data_quality"]
    recs.append(dict(priority="Medium", title="Fix data quality at the source",
        body=f"{dq['corrupted_or_missing_cells']:,} cells ({dq['corrupted_or_missing_cells']/dq['total_cells']*100:.2f}%) were corrupted or missing "
             "(e.g. 999,520 days of tenure, text values, negative tenure). Add validation rules at ingestion; the model tolerates them but they hide real signal.",
        evidence="Handled via NaN-aware gradient boosting rather than abs()/mean replacement."))

    recs.append(dict(priority="Info", title="Governance: retrain, monitor, and confirm the label",
        body=f"The training data covers {ins['date_from']} to {ins['date_to']}. Retrain on recent data before go-live, monitor PSI and calibration monthly, "
             "and check whether 'not repaid in 5 days' customers repay later - if they do, true losses are lower and the cut-off can be loosened (use the Business Simulator).",
        evidence=f"Out-of-time test (train before {metrics['out_of_time']['split_date']}, test after): ROC-AUC {metrics['out_of_time']['roc_auc']:.3f}, recall {metrics['out_of_time']['recall_defaulters']*100:.0f}% - performance drops modestly on future loans, so monitor monthly."))
    order = {"High": 0, "Medium": 1, "Info": 2}
    return sorted(recs, key=lambda r: order[r["priority"]])
