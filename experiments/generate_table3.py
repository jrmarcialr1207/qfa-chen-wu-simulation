"""
Genera los datos de la Tabla 3: comparacion entre la construccion original
del Teorema 3.2 y la refinada (Proposiciones 1 y 2 -- cota ajustada y
normalizacion por simbolo), para n=2,...,6.

Requiere: qfa_simulation.py y qfa_improved.py en el mismo directorio.
"""

import json
import numpy as np

from qfa_simulation import build_random_qfa, qfa_to_gfa, gfa_to_pfa
from qfa_improved import gfa_to_pfa_improved

LAMBDA = 0.45


def find_breaking_length(n, lam, seed, constructor, max_len=16, alphabet=('a', 'b')):
    """Encuentra la primera longitud de palabra en la que la construccion deja
    de coincidir con el 1gQFA original en aritmetica float64."""
    rng = np.random.default_rng(seed)
    qfa = build_random_qfa(n, alphabet, rng)
    gfa = qfa_to_gfa(qfa)
    pfa = constructor(gfa, lam)
    for L in range(1, max_len + 1):
        rng2 = np.random.default_rng(1000 + L)
        test_words = [''.join(rng2.choice(list(alphabet), size=L)) for _ in range(20)]
        for w in test_words:
            fQ = qfa.run(w)
            fP = pfa.accept_probability(w)
            if (fQ > lam) != (fP > 0.5):
                return L
    return None


def compute_CN_old_new(n, lam, seed):
    rng = np.random.default_rng(seed)
    qfa = build_random_qfa(n, ('a', 'b'), rng)
    gfa = qfa_to_gfa(qfa)
    # 'original' == cota conservadora (max valor absoluto) + C compartido,
    # obtenido aqui reutilizando gfa_to_pfa_improved con las banderas que
    # replican exactamente la construccion original del Teorema 3.2.
    pfa_orig_equiv = gfa_to_pfa_improved(gfa, lam, tight_bound=False, per_symbol_C=False)
    pfa_new = gfa_to_pfa_improved(gfa, lam, tight_bound=True, per_symbol_C=True)
    CN_old = max(pfa_orig_equiv.C_dict[s] * pfa_orig_equiv.N for s in pfa_orig_equiv.C_dict)
    CN_new = max(pfa_new.C_dict[s] * pfa_new.N for s in pfa_new.C_dict)
    return CN_old, CN_new, pfa_new.N + 2


def generate_table3():
    rows = []
    for n in range(2, 7):
        CN_old, CN_new, total_states = compute_CN_old_new(n, LAMBDA, 100 + n)
        L_old = find_breaking_length(n, LAMBDA, 100 + n, lambda g, l: gfa_to_pfa(g, l))
        L_new = find_breaking_length(
            n, LAMBDA, 100 + n,
            lambda g, l: gfa_to_pfa_improved(g, l, tight_bound=True, per_symbol_C=True)
        )
        mejora = CN_old / CN_new
        row = (n, total_states, round(CN_old, 1), round(CN_new, 1), round(mejora, 1), L_old, L_new)
        rows.append(row)
        print(row)
    with open('table3_data.json', 'w') as f:
        json.dump(rows, f)
    return rows


if __name__ == '__main__':
    generate_table3()
