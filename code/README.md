# Simulation study for the EARL estimator

Semi-synthetic simulation study accompanying the paper *EARL: Exposure- and
Allocation-Reweighted Linear Estimator for Bipartite Experiments with Partial
Assignment*.

## Layout

- `graphs.py` — bipartite experiment graphs: the synthetic generator of
  Doudchenko et al. (2020), the Amazon user–item review graph of
  He & McAuley (2016), and the MovieLens-100K user–movie graph.
- `dgp.py` — semi-synthetic outcome models (scenarios S1–S3 with an optional
  effect of unallocated units, e.g. from a concurrent experiment).
- `estimators.py` — EARL, its centred-weight variant, ERL on the reduced
  graph, two Horvitz–Thompson (IPW) benchmarks, and the
  randomisation-inference variance estimator.
- `run_simulations.py` — runs the full study and writes `results/rmse.csv`
  and `results/ri_size.csv`.
- `plot_results.py` — produces the figures used in the paper.

## Reproducing the results

1. Install dependencies (Python 3.11+):

   ```
   python3 -m venv venv && . venv/bin/activate
   pip install -r requirements.txt
   ```

2. Download the two public datasets (~30 MB):

   ```
   ./fetch_data.sh
   ```

3. Run the study (about 10–20 minutes on a laptop) and build the figures:

   ```
   python run_simulations.py
   python plot_results.py
   ```

All randomness is controlled by a single seed set in `run_simulations.py`;
re-running reproduces the reported numbers exactly.
