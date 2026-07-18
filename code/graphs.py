"""Bipartite experiment graphs: a synthetic generator and two public datasets."""

import numpy as np
import pandas as pd
from scipy import sparse


def _to_csr(user_idx, item_idx, n_users, n_items):
    data = np.ones(len(user_idx), dtype=np.float64)
    B = sparse.csr_matrix((data, (user_idx, item_idx)), shape=(n_users, n_items))
    B.data[:] = 1.0
    return B


def synthetic_graph(n_analysis=1000, n_randomisation=100, deg_min=1, deg_max=10, rng=None):
    """Graph of Doudchenko et al. (2020), Section 6.1: each analysis unit is
    connected to m_a ~ U{deg_min..deg_max} randomisation units chosen at random."""
    rng = np.random.default_rng(rng)
    rows, cols = [], []
    for a in range(n_analysis):
        m = rng.integers(deg_min, deg_max + 1)
        for r in rng.choice(n_randomisation, size=m, replace=False):
            rows.append(a)
            cols.append(r)
    return _to_csr(np.array(rows), np.array(cols), n_analysis, n_randomisation)


def amazon_graph(path, n_analysis=1000, deg_min=2, deg_max=50, rng=None):
    """User-item graph from the Amazon review data of He & McAuley (2016),
    subsampled to n_analysis users as in Doudchenko et al. (2020), Section 6.2.
    Users are analysis units, items are randomisation units."""
    rng = np.random.default_rng(rng)
    df = pd.read_csv(path, header=None, names=["user", "item", "rating", "ts"])
    deg = df.groupby("user").size()
    eligible = deg[(deg >= deg_min) & (deg <= deg_max)].index.to_numpy()
    users = rng.choice(eligible, size=n_analysis, replace=False)
    sub = df[df["user"].isin(set(users))]
    user_codes = pd.Categorical(sub["user"]).codes
    item_codes = pd.Categorical(sub["item"]).codes
    return _to_csr(user_codes, item_codes, n_analysis, item_codes.max() + 1)


def movielens_graph(path):
    """Full MovieLens-100K user-movie graph (Harper & Konstan, 2015).
    Users are analysis units, movies are randomisation units."""
    df = pd.read_csv(path, sep="\t", header=None, names=["user", "item", "rating", "ts"])
    user_codes = pd.Categorical(df["user"]).codes
    item_codes = pd.Categorical(df["item"]).codes
    return _to_csr(user_codes, item_codes, user_codes.max() + 1, item_codes.max() + 1)
