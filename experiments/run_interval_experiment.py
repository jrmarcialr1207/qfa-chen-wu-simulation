import json
import time
import itertools

import numpy as np
import mpmath as mp

from qfa_simulation import build_random_qfa
from qfa_interval import qfa_to_gfa_iv, certify_word, fQ_interval, qfa_to_iv_data

ALPHABET = ('a', 'b')
LAMBDA = 0.45

VARIANT_ORIGINAL = dict(tight_bound=False, per_symbol_C=False)
VARIANT_REFINED = dict(tight_bound=True, per_symbol_C=True)


def sample_words(alphabet, seed=7):
    """Palabras cortas exhaustivas (longitud <= 3) + muestra aleatoria de
    palabras mas largas (longitud 5 y 7), para mantener el costo
    computacional razonable."""
    words = ['']
    for L in range(1, 4):
        words += [''.join(c) for c in itertools.product(alphabet, repeat=L)]

    rng = np.random.default_rng(seed)
    for L in (5, 7):
        for _ in range(4):
            words.append(''.join(rng.choice(list(alphabet), size=L)))
    return words


def ground_truth_decision(qfa, word, lam, prec=100):
    old_prec = mp.iv.prec
    try:
        mp.iv.prec = prec
        rho0_iv, channels_iv, Pacc_iv = qfa_to_iv_data(qfa)
        fq = fQ_interval(qfa, word, rho0_iv, channels_iv, Pacc_iv)
        lam_iv = mp.iv.mpf(str(lam))
        diff = fq - lam_iv
        if diff.a > 0:
            return 'accept'
        elif diff.b < 0:
            return 'reject'
        else:
            return 'inconclusive'
    finally:
        mp.iv.prec = old_prec


def run_for_n(n, seed_qfa, words):
    rng = np.random.default_rng(seed_qfa)
    qfa = build_random_qfa(n, ALPHABET, rng)

    def gfa_fn():
        return qfa_to_gfa_iv(qfa)

    results = []
    for w in words:
        gt = ground_truth_decision(qfa, w, LAMBDA)

        t0 = time.time()
        r_orig = certify_word(gfa_fn, LAMBDA, w, VARIANT_ORIGINAL, prec_start=20)
        t_orig = time.time() - t0

        t0 = time.time()
        r_ref = certify_word(gfa_fn, LAMBDA, w, VARIANT_REFINED, prec_start=20)
        t_ref = time.time() - t0

        results.append({
            'n': n, 'word': w, 'len': len(w), 'ground_truth': gt,
            'orig_decision': r_orig['decision'], 'orig_prec': r_orig['prec'],
            'orig_rounds': r_orig['rounds'], 'orig_time': round(t_orig, 3),
            'ref_decision': r_ref['decision'], 'ref_prec': r_ref['prec'],
            'ref_rounds': r_ref['rounds'], 'ref_time': round(t_ref, 3),
        })
    return results


if __name__ == '__main__':
    all_results = []
    words = sample_words(ALPHABET, seed=7)
    print(f'Total de palabras por valor de n: {len(words)}')

    for n in (2, 3, 4, 5):
        print(f'\n=== n={n} ===')
        t0 = time.time()
        res_n = run_for_n(n, seed_qfa=100 + n, words=words)
        all_results.extend(res_n)
        print(f'  tiempo total: {time.time()-t0:.1f}s')

        mismatches_orig = [r for r in res_n if r['orig_decision'] != r['ground_truth']]
        mismatches_ref = [r for r in res_n if r['ref_decision'] != r['ground_truth']]
        print(f'  discrepancias (original vs verdad): {len(mismatches_orig)}')
        print(f'  discrepancias (refinada vs verdad): {len(mismatches_ref)}')
        print(f'  precision media (original): {np.mean([r["orig_prec"] for r in res_n]):.1f}')
        print(f'  precision media (refinada): {np.mean([r["ref_prec"] for r in res_n]):.1f}')
        print(f'  precision maxima (original): {max(r["orig_prec"] for r in res_n)}')
        print(f'  precision maxima (refinada): {max(r["ref_prec"] for r in res_n)}')

    with open('/home/claude/interval_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\nResultados guardados en interval_results.json')
