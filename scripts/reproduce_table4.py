#!/usr/bin/env python
"""
reproduce_table4.py -- reproduces Table 4 and Table 5 of
"Discovering Piano Performance Evaluation Topology".

Run from the directory holding shaptemp.csv, general_dag.json, shap_v1/,
overallscore_v1/, result_v1_b/, result_v1_c/:

    python reproduce_table4.py

Writes table4.csv and table5.csv and prints the reported numbers.

Protocol
--------
Ours       : selected feature-measure pairs are adjusted along the direction and
             magnitude given by the evaluation topology (DAG weight x segment
             importance x eta). The pre-adjustment value is the normalised
             feature value stored in the selection file, so the result stays on
             the 1-7 scale of the feature sheet.
Baseline15 : selected pairs are reassigned a value drawn uniformly from [1, 7],
             following the protocol of the baseline framework. The procedure is
             stochastic, so each performance is run N_DRAWS times and averaged.
             ALL selected pairs are modified, so the reported coverage matches
             what is actually changed.
Control    : same number of pairs as the arm it controls, drawn uniformly at
             random over the whole performance, modified under that arm's own
             correction rule.
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

VERSION = "v1"
ETA = 20.0
PERCENTILES = (90, 95)
N_DRAWS = 200
N_BOOT = 10000
SMOOTH_SIGMA = 1.0
PELT_PENALTY = 0.05
MIN_SEG_LEN, MAX_SEG_LEN = 2, 20
TARGET = "Unsatisfactory - Convincing Interpretation"


def train_evaluator():
    df = pd.read_csv("shaptemp.csv").dropna(subset=[TARGET])
    X, y = df.iloc[:, 3:21], df[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=1)
    model = XGBRegressor(learning_rate=0.098, n_estimators=300,
                         max_depth=6, random_state=1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("evaluator: MSE %.4f  R2 %.4f  (%d train / %d test records)"
          % (mean_squared_error(yte, pred), r2_score(yte, pred), len(Xtr), len(Xte)))
    return model


def load_dag():
    dag = json.load(open("general_dag.json", encoding="utf-8"))
    l1w = {i["from"]: i["norm_weight"] for i in dag if i["level"] == 1}
    l2 = {i["feature_id"]: i for i in dag if i["level"] == 2}
    w_dag = {i["feature_id"]: i["norm_weight"] for i in dag if i["level"] == 2}
    return l1w, l2, w_dag


def pid_list(code):
    files = glob.glob("result_%s_%s/*_result0.csv" % (VERSION, code))
    return sorted([os.path.basename(f).split("_")[0] for f in files], key=int)


def hits(code, pid, w_dag, iters=5):
    dp = pd.read_csv("result_%s_%s/%s_inter1.csv" % (VERSION, code, pid))
    dg = pd.read_csv("result_%s_%s/%s_inter4.csv" % (VERSION, code, pid))
    measures = sorted(dp["measure"].unique())
    features = sorted(dp["feature"].unique())
    mi = {m: i for i, m in enumerate(measures)}
    fi = {f: i for i, f in enumerate(features)}
    crit = {(int(r["measure"]), int(r["feature"])): r["critical_score"]
            for _, r in dg.iterrows()}
    M = np.zeros((len(measures), len(features)))
    for _, r in dp.iterrows():
        m, f = int(r["measure"]), int(r["feature"])
        M[mi[m], fi[f]] = abs(r["feature_value"]) * w_dag.get(f, 0.0) * (1 + crit.get((m, f), 0.0))
    h, a = np.ones(len(measures)), np.ones(len(features))
    for _ in range(iters):
        a = M.T @ h
        if np.linalg.norm(a) > 0:
            a /= np.linalg.norm(a, 2)
        h = M @ a
        if np.linalg.norm(h) > 0:
            h /= np.linalg.norm(h, 2)
    return pd.Series(h, index=measures), pd.Series(a, index=features), dp, dg


def select_cells(code, pid, pct, w_dag, cache):
    if (code, pid) not in cache:
        cache[(code, pid)] = hits(code, pid, w_dag)
    h, a, dp, dg = cache[(code, pid)]
    out = []
    for f in sorted(dp["feature"].unique()):
        fp = dp[dp["feature"] == f].sort_values("measure")
        fg = dg[dg["feature"] == f].sort_values("measure")
        mg = pd.merge(fp, fg, on="measure", suffixes=("_p", "_g"))
        if len(mg) < 10:
            continue
        imp = (h.loc[mg["measure"]].values
               * (abs(mg["feature_value_p"]) * w_dag.get(f, 0.0) * (1 + mg["critical_score"]))
               * a.loc[f])
        sig = gaussian_filter1d(np.log1p(100 * imp), sigma=SMOOTH_SIGMA)
        cps = rpt.Pelt(model="rbf").fit(sig.reshape(-1, 1)).predict(pen=PELT_PENALTY)
        segs, s0 = [], 0
        for cp in cps:
            if s0 < cp and MIN_SEG_LEN <= cp - s0 <= MAX_SEG_LEN:
                segs.append((s0, cp, float(np.mean(sig[s0:cp]))))
            s0 = cp
        if not segs:
            continue
        thr = np.percentile([s[2] for s in segs], pct)
        for s in segs:
            if s[2] >= thr:
                for i in range(s[0], s[1]):
                    r = mg.iloc[i]
                    out.append(dict(measure=int(r["measure"]), feature=int(f),
                                    feature_value=r["feature_value_p"],
                                    importance=s[2]))
    return pd.DataFrame(out)


def sheet(pid):
    feat = pd.read_csv("shap_%s/shap_%s.csv" % (VERSION, pid))
    S = feat.iloc[:, 1:20].values.astype(float)
    midx = {m: i for i, m in enumerate(feat["measure"].values)}
    return S, midx


def apply_ours(S, midx, cells, l1w, l2, eta=ETA):
    M = S.copy()
    for _, r in cells.iterrows():
        f, m = int(r["feature"]), int(r["measure"])
        if f in l2 and m in midx:
            w = l2[f]["norm_weight"] * l1w.get(l2[f]["to"], 1.0)
            M[midx[m], f - 1] = np.clip(
                (r["feature_value"] + l2[f]["direction"] * w * r["importance"] * eta) * 3 + 4, 1, 7)
    return M


def control_ours(S, k, avg_imp, l1w, l2, rs, eta=ETA):
    N = (S - 4.0) / 3.0
    adj = np.zeros(19)
    for f in range(1, 20):
        if f in l2:
            adj[f - 1] = (l2[f]["direction"] * l2[f]["norm_weight"]
                          * l1w.get(l2[f]["to"], 1.0) * avg_imp * eta)
    cap = S.shape[0] * 19
    flat = rs.choice(cap, k, replace=False)
    rr, cc = flat // 19, flat % 19
    M = S.copy()
    M[rr, cc] = np.clip((N[rr, cc] + adj[cc]) * 3 + 4, 1, 7)
    return M


def apply_baseline(S, midx, cells, rs):
    M = S.copy()
    for (m, f) in cells:
        if m in midx:
            M[midx[m], f - 1] = rs.uniform(1.0, 7.0)
    return M


def control_baseline(S, k, rs):
    cap = S.shape[0] * 19
    M = S.copy()
    for flat in rs.choice(cap, k, replace=False):
        M[flat // 19, flat % 19] = rs.uniform(1.0, 7.0)
    return M


def boot_ci(x, rs):
    x = np.asarray(x, float)
    means = np.array([np.mean(rs.choice(x, len(x), replace=True)) for _ in range(N_BOOT)])
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    model = train_evaluator()
    l1w, l2, w_dag = load_dag()
    cache = {}
    rows = []
    for code, composer in (("b", "Beethoven"), ("c", "Chopin")):
        for pid in pid_list(code):
            S, midx = sheet(pid)
            cap = S.shape[0] * 19
            orig = pd.read_csv("overallscore_%s/shap_%s.csv" % (VERSION, pid)).iloc[:, 0].mean()
            rec = dict(pid=int(pid), composer=composer, orig=orig)
            for pct in PERCENTILES:
                cells = select_cells(code, pid, pct, w_dag, cache)
                k, avg_imp = len(cells), cells["importance"].mean()
                rec["ours_%d" % pct] = model.predict(
                    apply_ours(S, midx, cells, l1w, l2)[:, :18]).mean()
                rec["cov_%d" % pct] = k / cap * 100
                rs = np.random.RandomState(1000 + int(pid))
                v = [model.predict(control_ours(S, k, avg_imp, l1w, l2, rs)[:, :18]).mean()
                     for _ in range(N_DRAWS)]
                rec["ctrl_%d" % pct] = float(np.mean(v))
            b = pd.read_csv("result_%s_%s/%s_result0.csv" % (VERSION, code, pid))
            bcells = [(int(r["measure"]), int(r["feature"])) for _, r in b.iterrows()]
            kb = len(bcells)
            rs = np.random.RandomState(2000 + int(pid))
            bv = np.array([model.predict(apply_baseline(S, midx, bcells, rs)[:, :18]).mean()
                           for _ in range(N_DRAWS)])
            cv = [model.predict(control_baseline(S, kb, rs)[:, :18]).mean()
                  for _ in range(N_DRAWS)]
            rec.update(base=bv.mean(), cov_base=kb / cap * 100, ctrl_base=float(np.mean(cv)),
                       p_vs_base=(np.sum(bv >= rec["ours_90"]) + 1) / (N_DRAWS + 1.0),
                       z_vs_base=(rec["ours_90"] - bv.mean()) / bv.std(ddof=1))
            rows.append(rec)
            print("  %s %s done" % (composer, pid))
    T = pd.DataFrame(rows)
    for c in ("ours_90", "ctrl_90", "ours_95", "ctrl_95", "base", "ctrl_base"):
        T["chg_" + c] = (T[c] - T["orig"]) / T["orig"] * 100
    T.to_csv("table4.csv", index=False)

    rs = np.random.RandomState(0)
    print("")
    print("=== Table 4 ===")
    for label, col, cov in (("Ours (p=90)", "chg_ours_90", "cov_90"),
                            ("  control", "chg_ctrl_90", "cov_90"),
                            ("Ours (p=95)", "chg_ours_95", "cov_95"),
                            ("  control", "chg_ctrl_95", "cov_95"),
                            ("Baseline15", "chg_base", "cov_base"),
                            ("  control", "chg_ctrl_base", "cov_base")):
        m, lo, hi = boot_ci(T[col], rs)
        ch = T[T.composer == "Chopin"][col].mean()
        be = T[T.composer == "Beethoven"][col].mean()
        print("%-13s cov=%5.2f%%  Chopin %+5.2f%%  Beethoven %+5.2f%%  All %+5.2f%% [%+.2f, %+.2f]"
              % (label, T[cov].mean(), ch, be, m, lo, hi))
    d = T["chg_ours_90"].values - T["chg_base"].values
    m, lo, hi = boot_ci(d, rs)
    chi = -2 * np.log(T["p_vs_base"].values).sum()
    print("")
    print("paired ours(p=90) - baseline: %+.2f pp [%+.2f, %+.2f], wins %d/%d, "
          "Wilcoxon p=%.2e, mean z=%+.2f, Fisher p=%.2e"
          % (m, lo, hi, int((d > 0).sum()), len(d), stats.wilcoxon(d)[1],
             T["z_vs_base"].mean(), stats.chi2.sf(chi, 2 * len(T))))

    print("")
    print("=== Table 5 (robustness to eta, p=90) ===")
    out = []
    for eta in (5.0, 10.0, 20.0, 40.0, 60.0):
        vals = []
        for code, composer in (("b", "Beethoven"), ("c", "Chopin")):
            for pid in pid_list(code):
                S, midx = sheet(pid)
                orig = pd.read_csv("overallscore_%s/shap_%s.csv"
                                   % (VERSION, pid)).iloc[:, 0].mean()
                cells = select_cells(code, pid, 90, w_dag, cache)
                o = model.predict(apply_ours(S, midx, cells, l1w, l2, eta)[:, :18]).mean()
                vals.append((o - orig) / orig * 100)
        out.append(dict(eta=eta, ours=float(np.mean(vals))))
        print("eta=%4.0f  ours %+5.2f%%" % (eta, np.mean(vals)))
    pd.DataFrame(out).to_csv("table5.csv", index=False)
    print("")
    print("wrote table4.csv and table5.csv")


if __name__ == "__main__":
    main()
