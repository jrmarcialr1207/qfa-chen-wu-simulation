"""
Tablas de la construccion de n^2+1 estados (Chen y Wu, version actual,
Prop. 3.1 + Teorema 3.2):

  Tabla 1: estados, discrepancias de cutpoint en float64 vs alta precision.
  Tabla 2: constante de decaimiento por simbolo 1/eps y cota universal.
  Tabla 3: variantes de la construccion y longitud de ruptura en float64.

Semillas: default_rng(100+n), lambda=0.45, alfabeto {a,b}, como en el resto.
Uso:  python3 generate_n2plus1_tables.py   (desde experiments/, con
      implementation/ en el PYTHONPATH)
"""
import json
import math
import sys
import os
import random

import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation'))
from qfa_simulation import build_random_qfa, qfa_to_gfa, all_words_up_to
import qfa_n2plus1 as N

LAM = 0.45
HP_DPS = 150

VARIANTS = {
    'V0_naive_paper':      dict(T_mode='naive',      eps_rule='paper', per_symbol=False),
    'V1_orth_paper':       dict(T_mode='orthogonal', eps_rule='paper', per_symbol=False),
    'V2_orth_tight':       dict(T_mode='orthogonal', eps_rule='tight', per_symbol=False),
    'V3_orth_tight_sym':   dict(T_mode='orthogonal', eps_rule='tight', per_symbol=True),
}


def make(n):
    rng = np.random.default_rng(100 + n)
    qfa = build_random_qfa(n, ('a', 'b'), rng)
    return qfa, qfa_to_gfa(qfa)


def worst_inv_eps(pfa):
    return max(1.0 / e for e in pfa['eps_f'].values())


def sample_words(L, count=64, seed=0):
    if 2 ** L <= count:
        import itertools
        return [''.join(c) for c in itertools.product('ab', repeat=L)]
    r = random.Random(seed * 1000 + L)
    return [''.join(r.choice('ab') for _ in range(L)) for _ in range(count)]


def break_length(pf, ph, max_len=40):
    """Primera longitud L en que alguna palabra muestreada tiene el signo
    de float64 distinto del de la referencia en alta precision."""
    inv_m_f = pf['inv_m']
    for L in range(1, max_len + 1):
        for w in sample_words(L):
            sf = N.accept_probability(pf, w) > inv_m_f
            sh = N.signal(ph, w) > 0
            if sf != sh:
                return L
    return None


def table1():
    rows = []
    for n in range(2, 7):
        qfa, gfa = make(n)
        words = all_words_up_to(('a', 'b'), 5)
        kw = VARIANTS['V3_orth_tight_sym']
        pf = N.pfa_from_gfa(gfa, LAM, 'float', **kw)
        ph = N.pfa_from_gfa(gfa, LAM, 'mp', dps=HP_DPS, **kw)
        bad_f = bad_h = 0
        mp.mp.dps = HP_DPS
        max_id = mp.mpf(0)
        for w in words:
            truth = qfa.run(w) - LAM > 0
            sf = N.signal(pf, w) > 0
            sh = N.signal(ph, w)
            bad_f += (sf != truth)
            bad_h += ((sh > 0) != truth)
            fg = N.fG_shifted(gfa, LAM, w, 'mp', HP_DPS)
            mp.mp.dps = HP_DPS
            rel = abs(sh - N.decay_factor(ph, w) * fg) / (abs(sh) + mp.mpf('1e-120'))
            max_id = max(max_id, rel)
        row = [n, n * n, pf['m'], n * n + 1, len(words), bad_f, bad_h, float(max_id)]
        rows.append(row)
        print('T1', row)
    return rows


def table2():
    old = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data',
                                      'table2_CN_constants.json')))
    rows = []
    for n in range(2, 7):
        qfa, gfa = make(n)
        out = [n, n * n + 1]
        for name in ('V0_naive_paper', 'V1_orth_paper', 'V3_orth_tight_sym'):
            pf = N.pfa_from_gfa(gfa, LAM, 'float', **VARIANTS[name])
            out.append(worst_inv_eps(pf))
        kappa_orth = max(N.pfa_from_gfa(gfa, LAM, 'float', **VARIANTS['V1_orth_paper'])['kappa'].values())
        out.append(kappa_orth)
        out.append(math.sqrt(n))
        out.append((n * n + 1) * math.sqrt(n))
        out.append(old[n - 2][4])           # C*N de la construccion 2k+6
        assert kappa_orth <= math.sqrt(n) + 1e-9
        rows.append(out)
        print('T2', [round(x, 3) for x in out])
    return rows


def table3():
    rows = []
    for n in range(2, 7):
        qfa, gfa = make(n)
        out = [n]
        ph_cache = {}
        for name, kw in VARIANTS.items():
            pf = N.pfa_from_gfa(gfa, LAM, 'float', **kw)
            ph = N.pfa_from_gfa(gfa, LAM, 'mp', dps=HP_DPS, **kw)
            bl = break_length(pf, ph, max_len=40)
            out.append(worst_inv_eps(pf))
            out.append(bl)
        rows.append(out)
        print('T3', out)
    return rows


if __name__ == '__main__':
    res = {'table1_n2plus1': table1(), 'table2_n2plus1': table2(),
           'table3_n2plus1': table3()}
    for k, v in res.items():
        json.dump(v, open(os.path.join(os.path.dirname(__file__), '..', 'data',
                                       k + '.json'), 'w'))
