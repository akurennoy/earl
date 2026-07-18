"""GATE estimators for bipartite experiments with partial assignment.

All estimators accept treatment data in vectorised form: S and Z are
(M, n_reps) matrices of allocation and assignment indicators, Y is the
(N, n_reps) matrix of outcomes; a (n_reps,) vector of estimates is returned.
"""

import numpy as np


def earl(B, deg, S, Z, Y, p, q):
    """Proposed estimator: each participating connection is weighted by the
    inverse allocation propensity 1/q and by the centred assignment."""
    w = (B @ (S * (Z - p))) / (q * p * (1.0 - p))
    return np.mean(w * Y, axis=0)


def earl_centred(B, deg, S, Z, Y, p, q):
    """Variant with the product of standardised allocation coverage and
    standardised exposure as the weight; requires would-be assignments of
    unallocated units. Coincides with ERL on the full graph at q=1."""
    wH = (B @ (Z - p)) / (p * (1.0 - p))
    if q >= 1.0:
        return np.mean(wH * Y, axis=0)
    wG = (B @ (S - q)) / (q * (1.0 - q))
    return np.mean(wG * wH * Y, axis=0)


def erl_drop(B, deg, S, Z, Y, p, q):
    """Standard ERL on the reduced graph: unallocated units are dropped."""
    w = (B @ (S * (Z - p))) / (p * (1.0 - p))
    return np.mean(w * Y, axis=0)


def ipw_participating(B, deg, S, Z, Y, p, q):
    """Horvitz-Thompson full-/empty-exposure contrast on the reduced graph;
    propensities reflect assignment only, as if the participating units were
    the whole graph."""
    k = B @ S
    t = B @ (S * Z)
    w_t = np.where(t == k, np.power(p, -k), 0.0)
    w_c = np.where(t == 0.0, np.power(1.0 - p, -k), 0.0)
    return np.mean((w_t - w_c) * Y, axis=0)


def ipw_allocation_aware(B, deg, S, Z, Y, p, q):
    """Horvitz-Thompson full-/empty-exposure contrast on the full graph;
    propensities account for both allocation and assignment."""
    d = deg[:, None]
    t = B @ (S * Z)
    c = B @ (S * (1.0 - Z))
    log_pt = -d * np.log(q * p)
    log_pc = -d * np.log(q * (1.0 - p))
    w_t = np.where(t == d, np.exp(np.minimum(log_pt, 700.0)), 0.0)
    w_c = np.where(c == d, np.exp(np.minimum(log_pc, 700.0)), 0.0)
    return np.mean((w_t - w_c) * Y, axis=0)


ESTIMATORS = {
    "EARL": earl,
    "EARL-centred": earl_centred,
    "ERL-drop": erl_drop,
    "IPW-assign": ipw_participating,
    "IPW-alloc": ipw_allocation_aware,
}


def v_ri(B, deg, Y, p, q, n_rerandomisations, rng):
    """Randomisation-inference variance estimate of the EARL estimator:
    empirical variance of EARL recomputed on re-randomised (S, Z) with the
    observed outcomes held fixed. Y is a single outcome vector (N,)."""
    M = B.shape[1]
    S_t = rng.binomial(1, q, size=(M, n_rerandomisations)).astype(np.float64)
    Z_t = rng.binomial(1, p, size=(M, n_rerandomisations)).astype(np.float64)
    w = (B @ (S_t * (Z_t - p))) / (q * p * (1.0 - p))
    taus = np.mean(w * Y[:, None], axis=0)
    return np.var(taus)
