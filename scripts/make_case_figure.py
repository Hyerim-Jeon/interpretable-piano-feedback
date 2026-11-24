#!/usr/bin/env python
"""
make_case_figure.py -- case-study figure: critical segments and the selected feedback
items of every performance, one row per performer, on a common measure axis.

    python make_case_figure.py case_items.json fig_case_timeline
"""
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "hatch.linewidth": 0.7})
COLOR = {("L2", "Strict tempo - Freely"): "#E39B4F",
         ("L2", "Even - Colourful"): "#4F83C2",
         ("L2", "Sophisticated - Raw"): "#A9C6EA",
         ("L2", "Honest - Imaginative"): "#5DAE6B",
         ("L1", "Musicality"): "#8E6BBF",
         ("L1", "Pedal"): "#2E9C97"}
FALLBACK = "#888888"


def label(level, node):
    if level == "L1":
        return "%s (category, L1)" % node
    if level == "L0":
        return "whole passage (L0)"
    return "%s (descriptor, L2)" % node.replace(" - ", "–")


def panel(ax, perfs, title, xmax):
    n = len(perfs)
    for row, p in enumerate(perfs):
        y = n - 1 - row
        ax.add_patch(Rectangle((0.5, y - 0.03), p["n_measures"], 0.06, color="#cfcfcf", lw=0))
        for s in p["segments"]:
            lo, hi = s["measures"]
            ax.add_patch(Rectangle((lo - 0.5, y - 0.3), hi - lo + 1, 0.6, facecolor="#ececec", edgecolor="#bcbcbc", lw=0.4))
        groups = {}
        for it in p["items"]:
            groups.setdefault(tuple(it["measures"]), []).append(it)
        for (lo, hi), its in groups.items():
            h = 0.6 / len(its)
            for k, it in enumerate(sorted(its, key=lambda i: i["rank"])):
                y0 = y + 0.3 - (k + 1) * h
                ax.add_patch(Rectangle((lo - 0.5, y0), hi - lo + 1, h, facecolor=COLOR.get((it["level"], it["node"]), FALLBACK),
                                       edgecolor="black", lw=0.45, hatch="////" if it["level"] == "L1" else None))
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_xlim(0, xmax + 2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(p["pid"]) for p in perfs[::-1]], fontsize=6.5)
    ax.set_title(title, fontsize=8, loc="left", pad=3)
    ax.tick_params(axis="x", labelsize=7, length=2, pad=1)
    ax.tick_params(axis="y", length=0, pad=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def main(path, stem):
    D = json.load(open(path, encoding="utf-8"))
    P = D["performances"]
    beet = [p for p in P if p["piece"].startswith("Beethoven")]
    chop = [p for p in P if p["piece"].startswith("Chopin")]
    fig = plt.figure(figsize=(7.16, 3.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[len(beet) + 0.6, len(chop) + 1.6], width_ratios=[0.42, 0.58],
                          hspace=0.38, wspace=0.05, left=0.055, right=0.995, top=0.955, bottom=0.08)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    axl = fig.add_subplot(gs[1, 1])
    axl.axis("off")
    panel(ax1, beet, "(a) Beethoven, Thirty-Two Variations in C minor, WoO 80", max(p["n_measures"] for p in beet))
    panel(ax2, chop, "(b) Chopin, Étude Op. 10 No. 1", max(p["n_measures"] for p in chop))
    ax1.set_ylabel("performance", fontsize=7, labelpad=2)
    ax2.set_ylabel("performance", fontsize=7, labelpad=2)
    ax2.set_xlabel("measure", fontsize=7, labelpad=1)
    used = []
    for p in P:
        for it in p["items"]:
            k = (it["level"], it["node"])
            if k not in used:
                used.append(k)
    used.sort(key=lambda k: (k[0], k[1]))
    handles = [Patch(facecolor=COLOR.get(k, FALLBACK), edgecolor="black", lw=0.45,
                     hatch="////" if k[0] == "L1" else None, label=label(*k)) for k in used]
    handles.append(Patch(facecolor="#ececec", edgecolor="#bcbcbc", lw=0.4, label="critical segment, not selected"))
    axl.legend(handles=handles, loc="center left", fontsize=7, frameon=False, handlelength=2.0,
               title="Selected feedback items (budget B = 3)", title_fontsize=7.5, alignment="left",
               bbox_to_anchor=(0.04, 0.5))
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    print("wrote", stem)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
