#!/usr/bin/env python
"""
case_study.py -- per-performance feedback items with the reasons for their selection
(Section VI-F of the IEEE Access manuscript).

For every analyzed performance it runs the level-aware selector (definition D,
beta = 1, budget B) and records, for each selected item, the segment, the level,
the node, why the segment and the level were chosen, and the direction of every
deviation involved. Writes case_items.json.

    python case_study.py [B]
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import reproduce_table4 as R
import level_aware_selection as LA

B = int(sys.argv[1]) if len(sys.argv) > 1 else 3
BETA = 1.0
PIECE = {"b": "Beethoven, Thirty-Two Variations in C minor, WoO 80",
         "c": "Chopin, Etude Op. 10 No. 1 in C major"}
SCORE = {"c": "Chopin.musicxml"}


def anchors(name):
    a, b = [t.strip() for t in name.split(" - ", 1)]
    return a, b


def score_context(code, lo, hi, cache={}):
    """Notated directives in measures lo..hi (music21); None when no score is available."""
    if code not in SCORE or not os.path.exists(SCORE[code]):
        return None
    import music21
    if code not in cache:
        cache[code] = music21.converter.parse(SCORE[code])
    r = cache[code].measures(lo, hi)
    out = {"measures": "%d~%d" % (lo, hi), "dynamics": [], "expressions": [], "hairpins": 0, "articulations": set()}
    for el in r.recurse():
        cl = el.classes
        if "Dynamic" in cl:
            out["dynamics"].append("%s (m. %s)" % (el.value, el.measureNumber))
        if "TextExpression" in cl and el.content.strip() and all(ord(ch) < 128 for ch in el.content):
            out["expressions"].append("%s (m. %s)" % (el.content.strip(), el.measureNumber))
        if "Crescendo" in cl or "Diminuendo" in cl:
            out["hairpins"] += 1
        if "GeneralNote" in cl and hasattr(el, "articulations"):
            for a in el.articulations:
                out["articulations"].add(a.name)
    out["articulations"] = sorted(out["articulations"])
    return out


def deviation(cells, f, lo, hi, l2):
    d = cells[(cells.measure >= lo) & (cells.measure <= hi) & (cells.feature == f)]
    g = d[d.feature_value_g.notna()]
    if not len(g):
        return None
    x, gm = float(g.feature_value_p.mean()), float(g.feature_value_g.mean())
    a, b = anchors(l2[f]["from"])
    return {"name": l2[f]["from"], "anchors": [a, b], "performance": round(x, 3), "expert_norm": round(gm, 3),
            "toward": b if gm > x else a, "measures_with_guideline": int(len(g))}


def main():
    l1w, l2, w_dag = R.load_dag()
    tree = LA.build_tree(l2)
    cats = list(tree)
    f2c = {f: c for c in cats for f in tree[c]}
    WD = {f: w_dag[f] * l1w.get(l2[f]["to"], 1.0) for f in f2c}
    names = dict(pd.read_csv("pidname.csv").values)
    cache, perf = {}, []
    for code in ("b", "c"):
        for pid in R.pid_list(code):
            cells = LA.cell_table(R, code, pid, l1w, l2, w_dag, cache)
            ranges, crit = LA.passages(R, code, pid, cells, w_dag, cache)
            leaf, hsrc, rsrc, rng = LA.passage_inputs("D", ranges, crit, cells, tree, w_dag, WD)
            w, items, info = LA.build_items(leaf, hsrc, rsrc, tree, BETA)
            picks, curve = LA.greedy(w, LA.candidates(items, "aware"), B)
            total = sum(i["phi0"] for i in info)
            order = sorted(range(len(info)), key=lambda s: -info[s]["phi0"])
            seg_rank = {s: k + 1 for k, s in enumerate(order)}
            rec = {"pid": int(pid), "piece": PIECE[code], "maestro_file": names.get(int(pid)),
                   "n_measures": int(cells.measure.nunique()),
                   "segments": [{"measures": [int(lo), int(hi)], "share_of_weakness": round(info[s]["phi0"] / total, 3),
                                 "rank": seg_rank[s]} for s, (lo, hi) in enumerate(rng)],
                   "items": []}
            for k, (lvl, s, node) in enumerate(picks):
                lo, hi = rng[s]
                i = info[s]
                item = {"rank": k + 1, "level": lvl, "measures": [int(lo), int(hi)],
                        "segment_share_of_weakness": round(i["phi0"] / total, 3), "segment_rank": seg_rank[s],
                        "conveyed_after": round(curve[k], 3),
                        "category_shares": {c: round(i["phi1"][c] / i["phi0"], 3) for c in cats if i["phi1"][c] > 0}}
                if lvl == "L2":
                    c = f2c[node]
                    item.update(node=l2[node]["from"], category=c, H_category=round(i["Hc"][c], 3),
                                share_in_category=round(leaf[s].get(node, 0.0) / i["phi1"][c], 3),
                                single_descriptor=len(tree[c]) == 1,
                                deviations=[deviation(cells, node, lo, hi, l2)])
                    item["why_level"] = ("single descriptor in its category" if len(tree[c]) == 1 else
                                         "concentrated: share %.2f >= diffuseness %.2f" % (item["share_in_category"], item["H_category"]))
                elif lvl == "L1":
                    c = node
                    sh = {l2[f]["from"]: round(leaf[s].get(f, 0.0) / i["phi1"][c], 3) for f in tree[c] if leaf[s].get(f, 0.0) > 0}
                    item.update(node=c, category=c, H_category=round(i["Hc"][c], 3), descriptor_shares=sh,
                                deviations=[d for d in (deviation(cells, f, lo, hi, l2) for f in tree[c] if leaf[s].get(f, 0.0) > 0) if d])
                    item["why_level"] = "diffuse: diffuseness %.2f > largest share %.2f" % (i["Hc"][c], max(sh.values()))
                else:
                    item.update(node="overall", category=None, H_root=round(i["H0"], 3),
                                deviations=[d for d in (deviation(cells, f, lo, hi, l2) for c in cats for f in tree[c]
                                                         if leaf[s].get(f, 0.0) > 0) if d])
                    item["why_level"] = "weakness spread over categories: root diffuseness %.2f" % i["H0"]
                item["score_context"] = score_context(code, lo, hi)
                rec["items"].append(item)
            perf.append(rec)
            print("%s %s: %s" % (code, pid, [(it["level"], it["node"], it["measures"]) for it in rec["items"]]))
    json.dump({"budget": B, "beta": BETA, "definition": "D (Section V-A)", "performances": perf},
              open("case_items.json", "w"), indent=1, ensure_ascii=False)
    print("wrote case_items.json")


if __name__ == "__main__":
    main()
