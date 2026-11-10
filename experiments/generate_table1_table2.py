"""
Genera los datos de la Tabla 1 (verificacion de estados y comportamiento de
cutpoint) y la Tabla 2 (constantes C, N de la construccion del Teorema 3.2),
para n=2,...,6, tal como se reportan en el informe.

Requiere: qfa_simulation.py en el mismo directorio (o en el PYTHONPATH).
"""

import json
import numpy as np

from qfa_simulation import run_experiment, build_random_qfa, qfa_to_gfa

LAMBDA = 0.45


def generate_table1():
    rows = []
    for n in range(2, 7):
        result = run_experiment(n=n, lam=LAMBDA, max_word_len=5, seed=100 + n, hp_dps=100)
        row = (
            n,
            result['k_gfa_states'],
            result['n_pfa_total_states'],
            result['expected_pfa_bound_2k_plus_6'],
            result['num_words_tested'],
            len(result['double_precision_mismatches']),
            len(result['high_precision_mismatches']),
            result['max_gfa_reconstruction_error'],
        )
        rows.append(row)
        print(row)
    with open('results_table.json', 'w') as f:
        json.dump(rows, f)
    return rows


def compute_C_N(n, lam, seed):
    """Recalcula C y N (Teorema 3.2, construccion original) para un 1gQFA
    aleatorio dado, exactamente como en la Seccion 5 del informe."""
    rng = np.random.default_rng(seed)
    qfa = build_random_qfa(n, ('a', 'b'), rng)
    gfa = qfa_to_gfa(qfa)
    u, A_dict, v = gfa.u, gfa.A_dict, gfa.v
    k = gfa.k
    alphabet = gfa.alphabet()
    m = k + 1
    tildeA = {}
    for s in alphabet:
        A = A_dict[s]
        TA = np.zeros((m, m))
        TA[:k, :k] = A
        TA[k, k] = 1.0
        tildeA[s] = TA
    d = 2 * m
    hatA = {}
    for s in alphabet:
        TA = tildeA[s]
        P = np.maximum(TA, 0.0)
        Nn = np.maximum(-TA, 0.0)
        HA = np.zeros((d, d))
        HA[:m, :m] = P
        HA[:m, m:] = Nn
        HA[m:, :m] = Nn
        HA[m:, m:] = P
        hatA[s] = HA
    N = d + 2
    Bmats = {}
    for s in alphabet:
        HA = hatA[s]
        r = HA.sum(axis=1)
        c = HA.sum(axis=0)
        T = r.sum()
        B = np.zeros((N, N))
        B[1:1 + d, 0] = -r
        B[1:1 + d, 1:1 + d] = HA
        B[1 + d, 0] = T
        B[1 + d, 1:1 + d] = -c
        Bmats[s] = B
    max_entry = max(np.max(np.abs(B)) for B in Bmats.values())
    C = max_entry + 1.0
    return C, N, k


def generate_table2():
    results = []
    for n in range(2, 7):
        C, N, k = compute_C_N(n, LAMBDA, 100 + n)
        results.append((n, k, N, C, C * N))
        print(f'n={n}: k={k}, N={N}, C={C:.2f}, C*N={C*N:.2f}, log10(C*N)={np.log10(C*N):.3f}')
    with open('cn_table.json', 'w') as f:
        json.dump(results, f)
    return results


if __name__ == '__main__':
    print("=== Tabla 1: verificación de estados y cutpoint ===")
    generate_table1()
    print("\n=== Tabla 2: constantes C, N ===")
    generate_table2()
