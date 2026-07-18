"""Run the simulation study and write aggregated results to results/.

Usage: python run_simulations.py [--reps 1000] [--data-dir data]
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import dgp
import estimators as est
import graphs

P_TREAT = 0.5
Q_GRID = [round(0.1 * k, 2) for k in range(1, 11)]
RMSE_CONFIGS = [("S1", 0.0), ("S1", 1.0), ("S2", 0.0), ("S2", 1.0), ("S3", 1.0)]
RI_CONFIGS = [("S2null", 0.0), ("S2null", 1.0)]
RI_Q = [0.3, 0.7]
SEED = 20260718


def build_graphs(data_dir, rng):
    return {
        "synthetic": graphs.synthetic_graph(rng=rng),
        "amazon": graphs.amazon_graph(os.path.join(data_dir, "amazon_mi.csv"), rng=rng),
        "movielens": graphs.movielens_graph(os.path.join(data_dir, "ml-100k", "u.data")),
    }


def draw_experiment(B, deg, model, q, reps, rng):
    M = B.shape[1]
    S = rng.binomial(1, q, size=(M, reps)).astype(np.float64)
    Z = rng.binomial(1, P_TREAT, size=(M, reps)).astype(np.float64)
    G = (B @ S) / deg[:, None]
    F = (B @ (S * Z)) / deg[:, None]
    eps = rng.normal(0.0, np.sqrt(dgp.SIGMA2_EPS), size=G.shape)
    Y = model.expected_outcome(G, F) + eps
    return S, Z, Y


def rmse_study(graph_dict, reps, rng):
    rows = []
    for gname, B in graph_dict.items():
        deg = np.asarray(B.sum(axis=1)).ravel()
        for scenario, gamma_u in RMSE_CONFIGS:
            model = dgp.make_model(scenario, gamma_u, B.shape[0], rng)
            for q in Q_GRID:
                S, Z, Y = draw_experiment(B, deg, model, q, reps, rng)
                for ename, efun in est.ESTIMATORS.items():
                    taus = efun(B, deg, S, Z, Y, P_TREAT, q)
                    bias = float(np.mean(taus) - model.gate)
                    sd = float(np.std(taus))
                    rows.append({
                        "graph": gname, "scenario": scenario, "gamma_u": gamma_u,
                        "q": q, "estimator": ename, "gate": model.gate,
                        "bias": bias, "sd": sd,
                        "rmse": float(np.sqrt(bias ** 2 + sd ** 2)), "reps": reps,
                    })
                print(f"rmse: {gname} {scenario} gamma={gamma_u} q={q} done", flush=True)
    return pd.DataFrame(rows)


def ri_study(graph_dict, reps, n_rerand, rng):
    z_crit = stats.norm.ppf(0.975)
    rows = []
    for gname in ("synthetic", "amazon"):
        B = graph_dict[gname]
        deg = np.asarray(B.sum(axis=1)).ravel()
        for scenario, gamma_u in RI_CONFIGS:
            model = dgp.make_model(scenario, gamma_u, B.shape[0], rng)
            for q in RI_Q:
                S, Z, Y = draw_experiment(B, deg, model, q, reps, rng)
                taus = est.earl(B, deg, S, Z, Y, P_TREAT, q)
                v_hats = np.array([
                    est.v_ri(B, deg, Y[:, j], P_TREAT, q, n_rerand, rng)
                    for j in range(reps)
                ])
                reject = np.abs(taus) > z_crit * np.sqrt(v_hats)
                rows.append({
                    "graph": gname, "scenario": scenario, "gamma_u": gamma_u,
                    "q": q, "size": float(np.mean(reject)),
                    "true_var": float(np.var(taus)),
                    "mean_v_ri": float(np.mean(v_hats)),
                    "reps": reps, "n_rerand": n_rerand,
                })
                print(f"ri: {gname} gamma={gamma_u} q={q} done", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--ri-reps", type=int, default=500)
    ap.add_argument("--rerandomisations", type=int, default=200)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    graph_dict = build_graphs(args.data_dir, rng)
    os.makedirs(args.out_dir, exist_ok=True)

    for gname, B in graph_dict.items():
        deg = np.asarray(B.sum(axis=1)).ravel()
        print(f"{gname}: N={B.shape[0]} M={B.shape[1]} edges={int(B.sum())} "
              f"d_A={int(deg.max())} d_R={int((B.sum(axis=0)).max())}", flush=True)

    rmse = rmse_study(graph_dict, args.reps, rng)
    rmse.to_csv(os.path.join(args.out_dir, "rmse.csv"), index=False)

    ri = ri_study(graph_dict, args.ri_reps, args.rerandomisations, rng)
    ri.to_csv(os.path.join(args.out_dir, "ri_size.csv"), index=False)


if __name__ == "__main__":
    main()
