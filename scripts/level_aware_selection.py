#!/usr/bin/env python
"""
level_aware_selection.py -- Section VI-E of
"Interpretable Piano Performance Feedback From Hierarchical and Semantic
Synergies in Features and Segments".

Three-level weakness attribution, diffuseness, and level-aware feedback
selection (greedy maximization of the conveyed weakness F under a budget B),
compared with selection restricted to one level, with alternative definitions
of the within-category split (ablation), and with expert annotation behaviour.

Run from the directory that holds reproduce_table4.py and its inputs:

    python level_aware_selection.py          # full run
    python level_aware_selection.py test     # worked example of Section V-D only

Outputs: la_per_performance.csv, la_coverage.csv, la_beta.csv, la_ablation.csv,
la_examples.json, and a printed report.

Definitions (Section V of the paper)
------------------------------------
critical pair     a (measure, descriptor) pair selected by mutual reinforcement +
                  PELT at percentile PCT (reproduce_table4.select_cells)
passage s         maximal run of consecutive measures holding a critical pair
gap(m, f)         |x_fm - g_fm| / 2  (x, g on [-1, 1]; 0 where no guideline value)
attr(m, f)        I(m, f) * W_L1(parent(f)) * gap(m, f)   (I: importance, Eq. 8)
phi1(s, c)        sum of attr over the critical pairs of s whose descriptor is in c
D(s, f)           sum of gap(m, f) over all measures of s   (deviation profile)
phi2(s, f)        phi1(s, c) * D(s, f) / sum_{f' in c} D(s, f')        [definition D]
H(s, c)           normalized entropy of phi2(s, .) over the descriptors of c
H(s, root)        normalized entropy of phi1(s, .) over the categories
fidelity          L2 item 1 on its leaf; L1 item H(s,c)^beta on the leaves of c;
                  L0 item (H(s,root) H(s,c))^beta on every leaf of s
F(S)              sum over leaves of weight * max fidelity over the selected items

Ablation definitions of the split (all other steps identical):
  original  phi2 = sum of attr over critical pairs (topology weights in the split)
  topo      phi2 = W_DAG(f) * D(s, f)               (path weight x deviation)
  wsplit    phi2 = phi1(s,c) * w(f) D(s,f) / sum w D, diffuseness from D only
  topoleaf  leaf weights W_DAG(f) * D(s, f), diffuseness (category and root) from D only

Descriptor 19 (Competency) is the surrogate of the overall score and not a leaf.
"""
import sys
import json
import numpy as np
import pandas as pd

PCT = 90
BUDGETS = (1, 2, 3, 5, 10)
BETAS = (0.5, 1.0, 2.0, 4.0)
SURROGATE = 19
R_WIDTH = 2.0
N_BOOT = 10000
POLICIES = ("L0", "L1", "L2", "aware")
DEFINITIONS = ("D", "original", "topo", "wsplit", "topoleaf")
MAX_STEPS = 10
EXPERT_FILES = ("beethoven1.csv", "beethoven2.csv", "chopin1.csv", "chopin2.csv")
EXCLUDED_FROM_EXPERT = ("Tempo", "Touch")   # Tempo: one descriptor; Touch: interface grouping differs


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------
def build_tree(l2):
    tree = {}
    for f in sorted(l2):
        if f == SURROGATE:
            continue
        tree.setdefault(l2[f]["to"], []).append(f)
    return tree


def norm_entropy(v, k):
    v = np.asarray(v, float)
    tot = v.sum()
    if tot <= 0:
        return 0.0
    if k == 1:
        return 1.0
    p = v[v > 0] / tot
    return max(0.0, float(-(p * np.log(p)).sum() / np.log(k)))


def build_items(leaf, hsrc, rsrc, tree, beta):
    """
    leaf : list over passages of {descriptor: leaf weight >= 0}
    hsrc : list over passages of {descriptor: value}; within-category diffuseness
    rsrc : list over passages of {category: value}; root diffuseness
    """
    feats = [f for c in tree for f in tree[c]]
    cats = list(tree)
    index, w = {}, []
    for s, d in enumerate(leaf):
        for f in feats:
            index[(s, f)] = len(w)
            w.append(max(float(d.get(f, 0.0)), 0.0))
    w = np.asarray(w, float)
    if w.sum() > 0:
        w = w / w.sum()
    items = {"L2": [], "L1": [], "L0": []}
    info = []
    for s, d in enumerate(leaf):
        phi1 = {c: sum(max(d.get(f, 0.0), 0.0) for f in tree[c]) for c in cats}
        Hc = {c: (norm_entropy([max(hsrc[s].get(f, 0.0), 0.0) for f in tree[c]], len(tree[c]))
                  if phi1[c] > 0 else 0.0) for c in cats}
        H0 = norm_entropy([max(rsrc[s].get(c, 0.0), 0.0) for c in cats], len(cats))
        info.append(dict(phi1=phi1, Hc=Hc, H0=H0, phi0=sum(phi1.values())))
        for c in cats:
            for f in tree[c]:
                if d.get(f, 0.0) > 0:
                    items["L2"].append(("L2", s, f, np.array([index[(s, f)]]), np.array([1.0])))
            if phi1[c] > 0 and Hc[c] > 0:
                idx = np.array([index[(s, f)] for f in tree[c]])
                items["L1"].append(("L1", s, c, idx, np.full(len(idx), Hc[c] ** beta)))
        if H0 > 0:
            idx, fid = [], []
            for c in cats:
                for f in tree[c]:
                    idx.append(index[(s, f)])
                    fid.append((H0 * Hc[c]) ** beta if Hc[c] > 0 else 0.0)
            items["L0"].append(("L0", s, "root", np.array(idx), np.array(fid)))
    return w, items, info


def greedy(w, cand, max_steps):
    conveyed = np.zeros(len(w))
    picked, curve = [], []
    used = np.zeros(len(cand), bool)
    for _ in range(max_steps):
        best, best_gain = -1, 0.0
        for i, (_, _, _, idx, fid) in enumerate(cand):
            if used[i]:
                continue
            g = float((w[idx] * np.maximum(0.0, fid - conveyed[idx])).sum())
            if g > best_gain + 1e-15:
                best, best_gain = i, g
        if best < 0:
            break
        used[best] = True
        _, _, _, idx, fid = cand[best]
        conveyed[idx] = np.maximum(conveyed[idx], fid)
        picked.append(cand[best][:3])
        curve.append(float((w * conveyed).sum()))
    return picked, curve


def candidates(items, policy):
    if policy == "aware":
        return items["L2"] + items["L1"] + items["L0"]
    return items[policy]


def f_at(curve, b):
    if not curve:
        return 0.0
    return curve[min(b, len(curve)) - 1]


def self_test():
    """Worked example of Section V-D (shares given directly)."""
    parents = {1: "Tempo", 2: "Dynamic", 3: "Dynamic", 4: "Touch", 5: "Touch", 6: "Touch",
               7: "Tone", 8: "Tone", 9: "Tone", 10: "Pedal", 11: "Pedal",
               12: "Musicality", 13: "Musicality", 14: "Musicality", 15: "Musicality",
               16: "Emotion", 17: "Emotion", 18: "Emotion", 19: "Competency"}
    tree = build_tree({f: {"to": c} for f, c in parents.items()})
    leaf = [{10: 0.45, 11: 0.05, 7: 0.15, 8: 0.14, 9: 0.13, 2: 0.05, 14: 0.03}]
    root = [{c: sum(leaf[0].get(f, 0.0) for f in tree[c]) for c in tree}]
    w, items, info = build_items(leaf, leaf, root, tree, 1.0)
    got = {p: f_at(greedy(w, candidates(items, p), 2)[1], 2) for p in POLICIES}
    want = {"aware": 0.869, "L1": 0.654, "L2": 0.600, "L0": 0.325}
    for p in POLICIES:
        assert abs(got[p] - want[p]) < 1e-3, (p, got[p], want[p])
    picks = greedy(w, candidates(items, "aware"), 2)[0]
    assert picks[0][:3:2] == ("L2", 10) and picks[1][:3:2] == ("L1", "Tone"), picks
    assert abs(info[0]["Hc"]["Pedal"] - 0.469) < 1e-3 and abs(info[0]["H0"] - 0.496) < 1e-3
    return got


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def cell_table(R, code, pid, l1w, l2, w_dag, cache):
    if (code, pid) not in cache:
        cache[(code, pid)] = R.hits(code, pid, w_dag)
    h, a, dp, dg = cache[(code, pid)]
    mg = pd.merge(dp[["measure", "feature", "feature_value"]],
                  dg[["measure", "feature", "feature_value", "critical_score"]],
                  on=["measure", "feature"], suffixes=("_p", "_g"))
    mg["measure"] = mg["measure"].astype(int)
    mg["feature"] = mg["feature"].astype(int)
    mg = mg[mg["feature"] != SURROGATE].copy()
    wl2 = mg["feature"].map(lambda f: w_dag.get(f, 0.0)).values
    wl1 = mg["feature"].map(lambda f: l1w.get(l2[f]["to"], 1.0) if f in l2 else 0.0).values
    imp = (h.reindex(mg["measure"]).fillna(0.0).values
           * (mg["feature_value_p"].abs().values * wl2 * (1 + mg["critical_score"].fillna(0.0).values))
           * a.reindex(mg["feature"]).fillna(0.0).values)
    mg["gap"] = (mg["feature_value_p"] - mg["feature_value_g"]).abs().fillna(0.0).values / R_WIDTH
    mg["attr"] = imp * wl1 * mg["gap"].values
    mg["dirn"] = np.sign((mg["feature_value_g"] - mg["feature_value_p"]).fillna(0.0).values)
    return mg


def passages(R, code, pid, cells, w_dag, cache):
    sel = R.select_cells(code, pid, PCT, w_dag, cache)
    sel = sel[sel["feature"] != SURROGATE]
    key = set(zip(sel["measure"].astype(int), sel["feature"].astype(int)))
    crit = cells[[(m, f) in key for m, f in zip(cells["measure"], cells["feature"])]]
    if len(crit) == 0:
        return [], crit
    ms = sorted(set(crit["measure"].astype(int)))
    runs, s0, prev = [], ms[0], ms[0]
    for m in ms[1:]:
        if m != prev + 1:
            runs.append((s0, prev))
            s0 = m
        prev = m
    runs.append((s0, prev))
    return runs, crit


def passage_inputs(defn, ranges, crit, cells, tree, w_dag, WD):
    """Return leaf weights, within-category source, root source per passage."""
    cats = list(tree)
    leaf, hsrc, rsrc, kept = [], [], [], []
    for r, (lo, hi) in enumerate(ranges):
        cr = crit[(crit["measure"] >= lo) & (crit["measure"] <= hi)]
        al = cells[(cells["measure"] >= lo) & (cells["measure"] <= hi)]
        attr = cr.groupby("feature")["attr"].sum().to_dict()
        D = al.groupby("feature")["gap"].sum().to_dict()
        phi1 = {c: sum(attr.get(f, 0.0) for f in tree[c]) for c in cats}
        if defn == "original":
            lf = {f: v for f, v in attr.items() if v > 0}
            hs = lf
        elif defn == "topo":
            lf = {f: WD[f] * D.get(f, 0.0) for f in WD if D.get(f, 0.0) > 0}
            hs = lf
        elif defn == "topoleaf":
            lf = {f: WD[f] * D.get(f, 0.0) for f in WD if D.get(f, 0.0) > 0}
            hs = D
        else:   # "D" or "wsplit"
            lf = {}
            for c in cats:
                if phi1[c] <= 0:
                    continue
                sc = {f: (w_dag[f] if defn == "wsplit" else 1.0) * D.get(f, 0.0) for f in tree[c]}
                tot = sum(sc.values())
                for f in tree[c]:
                    if tot > 0 and sc[f] > 0:
                        lf[f] = phi1[c] * sc[f] / tot
            hs = D if defn == "wsplit" else lf
        if sum(lf.values()) <= 0:
            continue
        if defn == "topoleaf":
            rs = {c: sum(D.get(f, 0.0) for f in tree[c]) for c in cats}
        else:
            rs = {c: sum(lf.get(f, 0.0) for f in tree[c]) for c in cats}
        leaf.append(lf)
        hsrc.append(hs)
        rsrc.append(rs)
        kept.append(r)
    return leaf, hsrc, rsrc, [ranges[i] for i in kept]


def addressed_pairs(picks, b, f2c):
    """(passage, category) pairs addressed by the first b items and whether each is
    treated as multi-faceted (an L1 item, or two or more descriptors)."""
    addr = {}
    for lvl, s, node in picks[:b]:
        if lvl == "L1":
            addr.setdefault((s, node), []).append("L1")
        elif lvl == "L2":
            addr.setdefault((s, f2c[node]), []).append("L2")
    out = []
    for (s, c), v in addr.items():
        if c in EXCLUDED_FROM_EXPERT:
            continue
        out.append(int(("L1" in v) or v.count("L2") >= 2))
    return out


def expert_rate(tree, rs):
    """Share of (annotation, category) pairs an expert treats as multi-faceted:
    two or more descriptors of the category rated, or the category rated with no
    descriptor. Multi-descriptor categories only; Tempo and Touch excluded."""
    import os
    if not all(os.path.exists(f) for f in EXPERT_FILES + ("feature_mapping.csv",)):
        return None
    A = pd.concat([pd.read_csv(f) for f in EXPERT_FILES], ignore_index=True)
    A = A.drop_duplicates("annotation_item_id")
    FM = pd.read_csv("feature_mapping.csv")
    item2feat = dict(zip(FM["new_feature"].astype(int), FM["feature"].astype(int)))
    f2c = {f: c for c in tree for f in tree[c]}
    l1_name = {2: "Dynamic", 5: "Tone", 6: "Pedal", 7: "Musicality", 8: "Emotion"}
    L2 = A[A["level"] == 2].copy()
    L2["cat"] = L2["item"].map(lambda i: f2c.get(item2feat.get(int(i), -1)))
    L1 = A[A["level"] == 1].copy()
    L1["cat"] = L1["item"].map(lambda i: l1_name.get(int(i)))
    k = L2.dropna(subset=["cat"]).groupby(["annotation_id", "cat"])["item"].nunique().rename("k").reset_index()
    l1 = L1.dropna(subset=["cat"])[["annotation_id", "cat"]].drop_duplicates().assign(hasL1=1)
    M = pd.merge(k, l1, on=["annotation_id", "cat"], how="outer").fillna({"k": 0, "hasL1": 0})
    M = M[M["cat"].map(lambda c: c in tree and len(tree[c]) > 1 and c not in EXCLUDED_FROM_EXPERT)]
    y = ((M["k"] >= 2) | ((M["hasL1"] == 1) & (M["k"] == 0))).values.astype(float)
    bs = [y[rs.randint(0, len(y), len(y))].mean() for _ in range(N_BOOT)]
    return dict(rate=float(y.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                n_pairs=int(len(y)), n_annotations=int(A["annotation_id"].nunique()),
                k2=float((M["k"] >= 2).mean()), l1only=float(((M["hasL1"] == 1) & (M["k"] == 0)).mean()))


def boot_ci(x, rs):
    x = np.asarray(x, float)
    means = np.array([np.mean(rs.choice(x, len(x), replace=True)) for _ in range(N_BOOT)])
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def boot_ci_grouped(groups, rs):
    """CI for a pooled share, resampling performances."""
    groups = [np.asarray(g, float) for g in groups if len(g)]
    if not groups:
        return float("nan"), float("nan"), float("nan")
    pooled = np.concatenate(groups)
    vals = []
    for _ in range(N_BOOT):
        pick = rs.randint(0, len(groups), len(groups))
        v = np.concatenate([groups[i] for i in pick])
        vals.append(v.mean())
    return pooled.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    from scipy import stats
    import reproduce_table4 as R
    print("self-test:", self_test())
    l1w, l2, w_dag = R.load_dag()
    tree = build_tree(l2)
    cats = list(tree)
    f2c = {f: c for c in cats for f in tree[c]}
    WD = {f: w_dag[f] * l1w.get(l2[f]["to"], 1.0) for f in f2c}
    top = {c: max(tree[c], key=lambda f: w_dag[f]) for c in cats}
    cache = {}
    rows, beta_rows, abl, examples = [], [], [], []
    multi = {d: {b: [] for b in (3, 5, 10)} for d in DEFINITIONS}
    for code, composer in (("b", "Beethoven"), ("c", "Chopin")):
        for pid in R.pid_list(code):
            cells = cell_table(R, code, pid, l1w, l2, w_dag, cache)
            ranges, crit = passages(R, code, pid, cells, w_dag, cache)
            for defn in DEFINITIONS:
                leaf, hsrc, rsrc, rng = passage_inputs(defn, ranges, crit, cells, tree, w_dag, WD)
                for beta in BETAS:
                    w, items, info = build_items(leaf, hsrc, rsrc, tree, beta)
                    picks, curve = greedy(w, candidates(items, "aware"), MAX_STEPS)
                    lv = [p[0] for p in picks[:3]]
                    beta_rows.append(dict(defn=defn, pid=int(pid), composer=composer, beta=beta,
                                          F3=f_at(curve, 3), n=len(lv),
                                          L0=lv.count("L0"), L1=lv.count("L1"), L2=lv.count("L2")))
                    if beta != 1.0:
                        continue
                    rec = dict(defn=defn, pid=int(pid), composer=composer, passages=len(rng),
                               mean_len=float(np.mean([hi - lo + 1 for lo, hi in rng])) if rng else 0.0)
                    rec["H_L1"] = float(np.mean([i["Hc"][c] for i in info for c in cats
                                                 if i["phi1"][c] > 0 and len(tree[c]) > 1] or [np.nan]))
                    rec["H_root"] = float(np.mean([i["H0"] for i in info] or [np.nan]))
                    for b in (3, 5, 10):
                        lvb = [p[0] for p in picks[:b]]
                        for L in ("L0", "L1", "L2"):
                            rec["n%s_B%d" % (L, b)] = lvb.count(L)
                        rec["n_B%d" % b] = len(lvb)
                        multi[defn][b].append(addressed_pairs(picks, b, f2c))
                    named = [node for lvl, s, node in picks[:5] if lvl == "L2" and len(tree[f2c[node]]) > 1]
                    rec["named_multi"] = len(named)
                    rec["named_top"] = sum(1 for f in named if f == top[f2c[f]])
                    rec["named_w"] = float(np.sum([w_dag[f] for f in named]))
                    for pol in POLICIES:
                        cv = curve if pol == "aware" else greedy(w, candidates(items, pol), MAX_STEPS)[1]
                        for b in BUDGETS:
                            rec["F_%s_B%d" % (pol, b)] = f_at(cv, b)
                    rows.append(rec)
                    if defn == "D":
                        for j, (lvl, s, node) in enumerate(picks[:3]):
                            if lvl != "L1":
                                continue
                            lo, hi = rng[s]
                            sh = {l2[f]["from"]: round(leaf[s].get(f, 0.0) / info[s]["phi1"][node], 3)
                                  for f in tree[node] if info[s]["phi1"][node] > 0}
                            examples.append(dict(pid=int(pid), composer=composer, rank=j + 1,
                                                 measures=[int(lo), int(hi)], category=node,
                                                 H=round(info[s]["Hc"][node], 3), shares=sh,
                                                 other_items=[(p[0], (l2[p[2]]["from"] if p[0] == "L2" else p[2]),
                                                               [int(x) for x in rng[p[1]]]) for p in picks[:3]]))
            print("  %s %s done" % (composer, pid))

    T = pd.DataFrame(rows)
    Bt = pd.DataFrame(beta_rows)
    T.to_csv("la_per_performance.csv", index=False)
    Bt.to_csv("la_beta.csv", index=False)
    json.dump(examples, open("la_examples.json", "w"), indent=1, ensure_ascii=False)
    rs = np.random.RandomState(0)
    D = T[T.defn == "D"]

    print("")
    print("=== passages (p=%d) ===" % PCT)
    print(D.groupby("composer")[["passages", "mean_len"]].mean().round(2).to_string())

    print("")
    print("=== [Table level usage] definition D, beta=1 ===")
    for b in (3, 5, 10):
        for name, sub in (("Beethoven", D[D.composer == "Beethoven"]), ("Chopin", D[D.composer == "Chopin"]), ("All", D)):
            tot = sub["n_B%d" % b].sum()
            print("B=%2d %-9s L0 %5.1f%%  L1 %5.1f%%  L2 %5.1f%%  (items %d)"
                  % (b, name, 100.0 * sub["nL0_B%d" % b].sum() / tot, 100.0 * sub["nL1_B%d" % b].sum() / tot,
                     100.0 * sub["nL2_B%d" % b].sum() / tot, tot))
    for name, sub in (("Beethoven", D[D.composer == "Beethoven"]), ("Chopin", D[D.composer == "Chopin"]), ("All", D)):
        print("mean H %-9s L1 %.3f  root %.3f" % (name, sub["H_L1"].mean(), sub["H_root"].mean()))

    print("")
    print("=== [Table coverage] conveyed weakness F, definition D, beta=1 (mean [95% CI]) ===")
    cov = []
    for pol in POLICIES:
        line = "%-6s" % pol
        for b in BUDGETS:
            m, lo, hi = boot_ci(D["F_%s_B%d" % (pol, b)], rs)
            cov.append(dict(policy=pol, B=b, mean=m, lo=lo, hi=hi))
            line += "  B=%d %.3f [%.3f, %.3f]" % (b, m, lo, hi)
        print(line)
    pd.DataFrame(cov).to_csv("la_coverage.csv", index=False)
    print("paired: level-aware minus best single level")
    for b in BUDGETS:
        means = {p: D["F_%s_B%d" % (p, b)].mean() for p in ("L0", "L1", "L2")}
        best = max(means, key=means.get)
        d = D["F_aware_B%d" % b].values - D["F_%s_B%d" % (best, b)].values
        m, lo, hi = boot_ci(d, rs)
        try:
            pv = stats.wilcoxon(d)[1] if np.any(np.abs(d) > 1e-12) else float("nan")
        except ValueError:
            pv = float("nan")
        print("  B=%2d vs %s: %+.4f [%+.4f, %+.4f] (%+.1f%% rel.), improved %d/%d, Wilcoxon p=%.2e"
              % (b, best, m, lo, hi, 100.0 * m / means[best], int((d > 1e-12).sum()), len(d), pv))

    print("")
    print("=== [beta] definition D, B=3 ===")
    for beta in BETAS:
        sub = Bt[(Bt.defn == "D") & (Bt.beta == beta)]
        tot = sub["n"].sum()
        print("beta=%.1f  L0 %5.1f%%  L1 %5.1f%%  L2 %5.1f%%  F %.3f"
              % (beta, 100.0 * sub["L0"].sum() / tot, 100.0 * sub["L1"].sum() / tot,
                 100.0 * sub["L2"].sum() / tot, sub["F3"].mean()))

    print("")
    ex = expert_rate(tree, np.random.RandomState(1))
    print("=== experts ===")
    print(ex)
    print("")
    print("=== [Table ablation] beta=1 ===")
    for defn in DEFINITIONS:
        S = T[T.defn == defn]
        r = dict(defn=defn, H_L1=S["H_L1"].mean())
        for b in (3, 5, 10):
            m, lo, hi = boot_ci_grouped(multi[defn][b], np.random.RandomState(2))
            r["multi_B%d" % b], r["multi_lo_B%d" % b], r["multi_hi_B%d" % b] = m, lo, hi
            tot = S["n_B%d" % b].sum()
            r["L0_B%d" % b] = S["nL0_B%d" % b].sum() / tot
            r["L1_B%d" % b] = S["nL1_B%d" % b].sum() / tot
        for b in (3, 5):
            means = {p: S["F_%s_B%d" % (p, b)].mean() for p in ("L0", "L1", "L2")}
            best = max(means, key=means.get)
            d = S["F_aware_B%d" % b].values - S["F_%s_B%d" % (best, b)].values
            r["gain_B%d" % b] = 100.0 * d.mean() / means[best]
            r["improved_B%d" % b] = int((d > 1e-12).sum())
        r["named_top"] = S["named_top"].sum() / max(S["named_multi"].sum(), 1)
        r["named_w"] = S["named_w"].sum() / max(S["named_multi"].sum(), 1)
        abl.append(r)
        print("%-9s H_L1 %.3f | multi-faceted B3 %.3f [%.3f,%.3f] B5 %.3f [%.3f,%.3f] B10 %.3f [%.3f,%.3f]"
              " | L1/L0 share B5 %.3f/%.3f | gain B3 %+.1f%% (%d) B5 %+.1f%% (%d) | named top %.3f w %.3f"
              % (defn, r["H_L1"], r["multi_B3"], r["multi_lo_B3"], r["multi_hi_B3"],
                 r["multi_B5"], r["multi_lo_B5"], r["multi_hi_B5"], r["multi_B10"], r["multi_lo_B10"], r["multi_hi_B10"],
                 r["L1_B5"], r["L0_B5"], r["gain_B3"], r["improved_B3"], r["gain_B5"], r["improved_B5"],
                 r["named_top"], r["named_w"]))
    pd.DataFrame(abl).to_csv("la_ablation.csv", index=False)
    print("")
    print("examples (definition D, L1 items among the first three):", len(examples))
    for e in examples[:6]:
        print(json.dumps(e, ensure_ascii=False))
    print("wrote la_per_performance.csv, la_coverage.csv, la_beta.csv, la_ablation.csv, la_examples.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print(self_test())
    else:
        main()
