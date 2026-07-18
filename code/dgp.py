"""Semi-synthetic outcome models on a fixed bipartite graph.

Outcomes follow Y_a = e_a + eps_a with
    e_a = alpha_a + response(F_a) + gamma_u * (1 - G_a),
where G_a is the fraction of a's connections allocated to the experiment,
F_a is the fraction of a's connections that are allocated *and* treated,
and gamma_u is the effect of unallocated connections (e.g. from a
concurrent experiment they participate in).

Scenario coefficients follow Harshaw et al. (2023), Section 8, adapted to
the partial-assignment setting; the noise level follows Doudchenko et al.
(2020), Section 6.1. Coefficients are drawn once per scenario and held
fixed across replications.
"""

import numpy as np

SIGMA2_EPS = 0.5


class OutcomeModel:
    def __init__(self, name, alpha, beta, gamma_u, nonlinear=False):
        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.gamma_u = gamma_u
        self.nonlinear = nonlinear
        self.gate = 0.0 if nonlinear else float(np.mean(beta))

    def expected_outcome(self, G, F):
        alpha = self.alpha if G.ndim == 1 else self.alpha[:, None]
        if self.nonlinear:
            resp = 4.0 * F * (F - 1.0)
        else:
            beta = self.beta if G.ndim == 1 else self.beta[:, None]
            resp = beta * F
        return alpha + resp + self.gamma_u * (1.0 - G)

    def draw_outcomes(self, G, F, rng):
        eps = rng.normal(0.0, np.sqrt(SIGMA2_EPS), size=len(self.alpha))
        return self.expected_outcome(G, F) + eps


def make_model(scenario, gamma_u, n_analysis, rng):
    rng = np.random.default_rng(rng)
    if scenario == "S1":  # positive heterogeneous effects
        alpha = rng.normal(-1.0, np.sqrt(3.0 / 8.0), n_analysis)
        beta = rng.normal(2.0, 1.0, n_analysis)
        return OutcomeModel("S1", alpha, beta, gamma_u)
    if scenario == "S2":  # near-zero effects
        alpha = rng.normal(2.0, np.sqrt(3.0 / 8.0), n_analysis)
        beta = rng.normal(0.0, np.sqrt(0.5), n_analysis)
        return OutcomeModel("S2", alpha, beta, gamma_u)
    if scenario == "S2null":  # exact strong null (per-unit zero effects)
        alpha = rng.normal(2.0, np.sqrt(3.0 / 8.0), n_analysis)
        beta = np.zeros(n_analysis)
        return OutcomeModel("S2null", alpha, beta, gamma_u)
    if scenario == "S3":  # non-linear response, zero GATE
        alpha = rng.normal(0.0, np.sqrt(1.0 / 8.0), n_analysis)
        return OutcomeModel("S3", alpha, None, gamma_u, nonlinear=True)
    raise ValueError(scenario)
