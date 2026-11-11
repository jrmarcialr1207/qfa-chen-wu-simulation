# Exact probabilistic simulation of 1gQFAs: implementation and experiments

Code, data and figures for the technical report *Exact Probabilistic Simulation of Quantum Finite
Automata: implementation, verification, and numerical analysis of the n²+1 construction of Chen and Wu*.

The report studies the construction of Chen and Wu, *The State Cost of Classical Simulation of
One-Way General Quantum Finite Automata*, arXiv:2604.07058 (current version): every n-state 1gQFA
is simulated exactly, under strict cutpoint, by a PFA with n²+1 states (Proposition 3.1 and
Theorem 3.2 of that paper), with the identity

    f_P(w) - 1/(n²+1) = δ · ε^|w| · (f_Q(w) - λ).

## Layout

| Path | Content |
|---|---|
| `implementation/qfa_n2plus1.py` | **Main module.** n²+1 construction, generic over float64 / mpmath / interval backends; orthogonal and naive choices of the isomorphism T; tight and per-symbol ε |
| `implementation/qfa_simulation.py` | Random 1gQFAs (Stinespring), Hilbert–Schmidt linearization (GFA), direct simulation |
| `implementation/qfa_interval.py` | Interval-arithmetic building blocks (basis, channels, GFA in `mpmath.iv`) |
| `implementation/qfa_lower_bound.py` | Dimension-counting separation identity (supplementary, our own argument) |
| `implementation/qfa_improved.py` | Legacy: refinements of a classical-style 2k+6 normalization (comparison baseline) |
| `experiments/generate_n2plus1_tables.py` | Tables 1–3 of the report |
| `experiments/run_n2plus1_interval.py` | Table 4 (interval certification, precision in bits) |
| `experiments/check_universal_bound.py` | Numerical check of κ ≤ √n (Proposition 3) |
| `experiments/generate_table5.py` | Table 5 (separation identity) |
| `experiments/make_n2plus1_figures.py` | Figures 1–3 |
| `experiments/generate_table1_table2.py`, `generate_table3.py`, `run_interval_experiment.py` | Legacy experiments for the 2k+6 baseline (Table 2, last column) |
| `data/` | JSON outputs (`*_n2plus1*.json` are the current ones) |
| `figures/` | PNG figures (`n2plus1_*_en.png` are the current ones) |

## Requirements

    pip install numpy mpmath matplotlib

## Reproducing the tables

Fixed seeds `default_rng(100 + n)`, λ = 0.45, alphabet {a, b}.

    cd experiments
    python3 generate_n2plus1_tables.py     # Tables 1-3 (about 1 minute)
    python3 run_n2plus1_interval.py        # Table 4 (about 30 seconds)
    python3 check_universal_bound.py       # κ <= sqrt(n) over 300 random automata per n
    python3 make_n2plus1_figures.py        # Figures 1-3

Float64 failures at other cutpoints (n = 6, words up to length 5, refined 2k+6 baseline not used):

    cd implementation
    python3 -c "
    from qfa_simulation import run_experiment
    for lam in (0.3, 0.45, 0.5, 0.7):
        r = run_experiment(n=6, lam=lam, max_word_len=5, seed=106, hp_dps=100)
        print(lam, len(r['double_precision_mismatches']), '/', r['num_words_tested'])"

(That last command exercises the legacy 2k+6 baseline: 48/63, 26/63, 16/63, 20/63 for λ = 0.3, 0.45, 0.5, 0.7.)

## Note on units

Interval precision is `mpmath.iv.prec`, which is measured in **bits** (not decimal digits).
