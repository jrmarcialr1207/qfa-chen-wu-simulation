"""
Genera los datos de la Tabla 5: certificacion por intervalos del argumento
de separacion del Teorema 4.1 (cota inferior de Chen y Wu), para n=2,...,6.

Requiere: qfa_lower_bound.py en el mismo directorio.

Nota: para n=2,3 se enumeran TODOS los patrones de signo (exhaustivo).
Para n=4 se enumera exhaustivamente por defecto en este script (32768
patrones), lo cual toma bastante (~90s); se puede pasar exhaustive_n4=False
para usar muestreo y acelerar la ejecucion. Para n=5,6 siempre se usa
muestreo aleatorio (300 patrones), ya que la enumeracion exhaustiva
(2^24 y 2^35 patrones) no es factible.
"""

import json
import itertools
import numpy as np

from qfa_lower_bound import (
    traceless_orthonormal_basis_np, max_valid_epsilon, certify_shattering,
)


def generate_table5(exhaustive_n4=True):
    results = []

    for n in (2, 3):
        basis = traceless_orthonormal_basis_np(n)
        eps = max_valid_epsilon(n, basis)
        d = len(basis)
        sample = list(itertools.product([1, -1], repeat=d))
        min_prec_found = None
        for prec_try in (2, 3, 4, 5, 6, 7, 8, 10, 15):
            res = certify_shattering(n, eps, prec_start=prec_try, prec_max=prec_try,
                                      max_rounds=1, sign_patterns=sample)
            if res['all_certified']:
                min_prec_found = prec_try
                break
        row = {'n': n, 'dim': d, 'eps': round(eps, 4), 'patrones_probados': len(sample),
               'exhaustivo': True, 'precision_minima_necesaria': min_prec_found}
        results.append(row)
        print(row)

    n = 4
    basis = traceless_orthonormal_basis_np(n)
    eps = max_valid_epsilon(n, basis)
    d = len(basis)
    if exhaustive_n4:
        sample = list(itertools.product([1, -1], repeat=d))
        exhaustive = True
    else:
        rng = np.random.default_rng(7)
        sample = [tuple(rng.choice([1, -1], size=d)) for _ in range(300)]
        exhaustive = False
    min_prec_found = None
    for prec_try in (2, 3, 4, 5, 6, 7, 8, 10, 15):
        res = certify_shattering(n, eps, prec_start=prec_try, prec_max=prec_try,
                                  max_rounds=1, sign_patterns=sample)
        if res['all_certified']:
            min_prec_found = prec_try
            break
    row = {'n': n, 'dim': d, 'eps': round(eps, 4), 'patrones_probados': len(sample),
           'exhaustivo': exhaustive, 'precision_minima_necesaria': min_prec_found}
    results.append(row)
    print(row)

    rng = np.random.default_rng(7)
    for n in (5, 6):
        basis = traceless_orthonormal_basis_np(n)
        eps = max_valid_epsilon(n, basis)
        d = len(basis)
        sample = [tuple(rng.choice([1, -1], size=d)) for _ in range(300)]
        min_prec_found = None
        for prec_try in (2, 3, 4, 5, 6, 7, 8, 10, 15):
            res = certify_shattering(n, eps, prec_start=prec_try, prec_max=prec_try,
                                      max_rounds=1, sign_patterns=sample)
            if res['all_certified']:
                min_prec_found = prec_try
                break
        row = {'n': n, 'dim': d, 'eps': round(eps, 4), 'patrones_probados': len(sample),
               'exhaustivo': False, 'precision_minima_necesaria': min_prec_found}
        results.append(row)
        print(row)

    with open('table5_lowerbound_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == '__main__':
    generate_table5(exhaustive_n4=True)
