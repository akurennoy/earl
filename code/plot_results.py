"""Generate the paper figure from results/rmse.csv.

Usage: python plot_results.py [--results results/rmse.csv] [--out ../figures]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

GRAPHS = [("synthetic", "Synthetic"), ("amazon", "Amazon"), ("movielens", "MovieLens")]
# distinct linestyle + marker per estimator so the lines stay readable in
# black-and-white print and for colour-blind readers
STYLE = {
    "EARL": dict(color="#0072B2", marker="o", ls="-", zorder=5),
    "EARL-centred": dict(color="#56B4E9", marker="s", ls=(0, (4, 1.5)), markerfacecolor="none"),
    "ERL-drop": dict(color="#D55E00", marker="^", ls="-."),
    "IPW-assign": dict(color="#009E73", marker="D", ls=(0, (5, 1.5, 1, 1.5)), markerfacecolor="none"),
    "IPW-alloc": dict(color="#CC79A7", marker="v", ls=(0, (2, 1))),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/rmse.csv")
    ap.add_argument("--out", default=os.path.join("..", "figures"))
    ap.add_argument("--scenario", default="S1")
    ap.add_argument("--gamma-u", type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df = df[(df.scenario == args.scenario) & (df.gamma_u == args.gamma_u)]
    os.makedirs(args.out, exist_ok=True)

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.2,
    })
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 4.4), sharex=True, layout="constrained")

    for j, (gkey, gtitle) in enumerate(GRAPHS):
        sub = df[df.graph == gkey]
        gate = sub.gate.iloc[0]
        ax_r, ax_b = axes[0, j], axes[1, j]
        for ename, style in STYLE.items():
            e = sub[sub.estimator == ename].sort_values("q")
            ax_r.plot(e.q, e.rmse, label=ename, **style)
            ax_b.plot(e.q, e.bias, label=ename, **style)
        qs = sorted(sub.q.unique())
        ax_b.plot(qs, [-(1 - q) * gate for q in qs], ls=":", color="black",
                  label=r"$-(1-q)\,GATE$", zorder=1)
        ax_b.axhline(0.0, color="black", lw=0.6, alpha=0.5)
        ax_r.set_yscale("log")
        ax_r.set_title(gtitle)
        ax_b.set_xlabel(r"allocation rate $q$")
        if j == 0:
            ax_r.set_ylabel("RMSE")
            ax_b.set_ylabel("bias")
        for ax in (ax_r, ax_b):
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.set_xlim(0.05, 1.05)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=6, frameon=False)
    out = os.path.join(args.out, "fig_rmse_bias.pdf")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
