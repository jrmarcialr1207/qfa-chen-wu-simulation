"""
Certificacion por aritmetica de intervalos con precision adaptativa de la
construccion n^2+1 (Chen y Wu, version actual, Prop. 3.1 + Teorema 3.2).

La precision se mide en BITS (mp.iv.prec) y se duplica en cada ronda:
16, 32, 64, ... Para cada palabra w se certifica el signo de
   s(w) = f_P(w) - 1/m      y      f_Q(w) - lambda   (simulacion directa)
y se verifica que coinciden. Una palabra queda certificada en la primera
ronda en que ambos intervalos excluyen el cero.

Uso: python3 run_n2plus1_interval.py   (desde experiments/)
"""
import json
import os
import sys
import time

import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation'))
from qfa_simulation import build_random_qfa
import qfa_interval as QI
import qfa_n2plus1 as N

LAM = 0.45
ALPHABET = ('a', 'b')
PREC_LADDER = [16, 32, 64, 128, 256, 512, 1024]

VARIANTS = {
    'V0_naive_paper':    dict(T_mode='naive',      eps_rule='paper', per_symbol=False),
    'V3_orth_tight_sym': dict(T_mode='orthogonal', eps_rule='tight', per_symbol=True),
}


def sample_words(seed=7, long_lengths=(12, 20), per_len=4):
    import itertools
    words = ['']
    for L in range(1, 4):
        words += [''.join(c) for c in itertools.product(ALPHABET, repeat=L)]
    rng = np.random.default_rng(seed)
    for L in (5, 7) + tuple(long_lengths):
        for _ in range(per_len):
            words.append(''.join(rng.choice(list(ALPHABET), size=L)))
    return words


def certify_all(qfa, words, kw, ladder=PREC_LADDER):
    """Devuelve {palabra: dict(prec, rounds, agree)} y tiempos."""
    ops = N.OPS['iv']
    n = qfa.n
    pending = list(words)
    out = {}
    old = mp.iv.prec
    try:
        for rnd, prec in enumerate(ladder, start=1):
            if not pending:
                break
            mp.iv.prec = prec
            u, A, v = QI.qfa_to_gfa_iv(qfa)
            lam_iv = mp.iv.mpf(str(LAM))
            v = N.shift_final_vector(v, lam_iv, n, ops)
            pfa = N.gfa_to_pfa_n2plus1(u, A, v, ops, **kw)
            rho0_iv, ch_iv, Pacc_iv = QI.qfa_to_iv_data(qfa)
            still = []
            for w in pending:
                s = N.signal(pfa, w)
                fq = QI.fQ_interval(qfa, w, rho0_iv, ch_iv, Pacc_iv) - lam_iv
                ok_p = (s.a > 0) or (s.b < 0)
                ok_q = (fq.a > 0) or (fq.b < 0)
                if ok_p and ok_q:
                    out[w] = dict(prec=prec, rounds=rnd,
                                  agree=bool((s.a > 0) == (fq.a > 0)))
                else:
                    still.append(w)
            pending = still
        for w in pending:
            out[w] = dict(prec=None, rounds=None, agree=None)
    finally:
        mp.iv.prec = old
    return out


def main(ns=(2, 3, 4, 5)):
    words = sample_words()
    print('palabras por n:', len(words))
    table = []
    raw = {}
    for n in ns:
        rng = np.random.default_rng(100 + n)
        qfa = build_random_qfa(n, ALPHABET, rng)
        row = [n, len(words)]
        raw[n] = {}
        for name, kw in VARIANTS.items():
            t0 = time.time()
            res = certify_all(qfa, words, kw)
            raw[n][name] = res
            precs = [r['prec'] for r in res.values() if r['prec'] is not None]
            unresolved = sum(1 for r in res.values() if r['prec'] is None)
            disagree = sum(1 for r in res.values() if r['agree'] is False)
            row += [max(precs), max(r['rounds'] for r in res.values() if r['rounds']),
                    round(float(np.mean(precs)), 1), disagree, unresolved]
            print(n, name, 'tiempo %.1fs' % (time.time() - t0), row[-5:])
        table.append(row)
    d = os.path.join(os.path.dirname(__file__), '..', 'data')
    json.dump(table, open(os.path.join(d, 'table4_n2plus1_interval.json'), 'w'))
    json.dump(raw, open(os.path.join(d, 'interval_n2plus1_raw.json'), 'w'))
    return table


if __name__ == '__main__':
    main()
