"""Figuras (EN) de la construccion n^2+1. Uso: python3 make_n2plus1_figures.py"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), '..', 'data')
F = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(F, exist_ok=True)
t2 = json.load(open(os.path.join(D, 'table2_n2plus1.json')))
t3 = json.load(open(os.path.join(D, 'table3_n2plus1.json')))
raw = json.load(open(os.path.join(D, 'interval_n2plus1_raw.json')))
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
C = {'naive': '#9a9a9a', 'orth': '#4c78a8', 'ref': '#d1495b', 'bound': '#222222'}

# Fig 1: attenuation per symbol (1/eps) vs n
ns = [r[0] for r in t2]
fig, ax = plt.subplots(figsize=(4.6, 3.1))
ax.plot(ns, [r[2] for r in t2], 'o--', c=C['naive'], label='naive $T$, rule $\\varepsilon\\max|K|\\leq 1/m$')
ax.plot(ns, [r[3] for r in t2], 's-', c=C['orth'], label='orthogonal $T$, same rule')
ax.plot(ns, [r[4] for r in t2], '^-', c=C['ref'], label='orthogonal $T$, tight rule (refined)')
ax.plot(ns, [r[7] for r in t2], 'k:', label='universal bound $(n^2+1)\\sqrt{n}$')
ax.set_yscale('log'); ax.set_xlabel('$n$'); ax.set_ylabel('per-symbol attenuation $1/\\varepsilon$')
ax.set_xticks(ns); ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(F, 'n2plus1_attenuation_en.png'), dpi=200); plt.close(fig)

# Fig 2: signal decay vs word length and observed float64 break lengths
fig, ax = plt.subplots(figsize=(4.8, 3.2))
cols = plt.cm.viridis(np.linspace(0.05, 0.9, 5))
for (n, *_rest), r3, col in zip(t2, t3, cols):
    inv_eps = r3[7]          # V3 worst 1/eps
    L = np.arange(0, 41)
    y = -L * math.log10(inv_eps)
    ax.plot(L, y, c=col, label=f'$n={n}$')
    bl = r3[8]
    ax.plot([bl], [-bl * math.log10(inv_eps)], 'o', c=col, ms=4)
ax.axhline(-16, c='k', ls='--', lw=0.8); ax.text(0.5, -18.2, 'float64 floor $\\approx 10^{-16}$', fontsize=7.5)
ax.set_ylim(-45, 2); ax.set_xlabel('word length $|w|$'); ax.set_ylabel('$\\log_{10}\\,\\varepsilon^{|w|}$')
ax.legend(frameon=False, fontsize=7.5, ncol=2); fig.tight_layout()
fig.savefig(os.path.join(F, 'n2plus1_decay_en.png'), dpi=200); plt.close(fig)

# Fig 3: certified precision (bits) vs word length
fig, ax = plt.subplots(figsize=(4.8, 3.2))
for n, col in zip(('2', '3', '4', '5'), cols):
    for name, ls, lab in (('V0_naive_paper', '--', 'naive'), ('V3_orth_tight_sym', '-', 'refined')):
        by = {}
        for w, r in raw[n][name].items():
            by.setdefault(len(w), []).append(r['prec'])
        Ls = sorted(by)
        ax.plot(Ls, [np.mean(by[l]) for l in Ls], ls, c=col, marker='o', ms=3,
                label=f'$n={n}$ {lab}')
ax.set_xlabel('word length $|w|$'); ax.set_ylabel('certified precision (bits, mean)')
ax.legend(frameon=False, fontsize=6, ncol=2); ax.set_xticks([0,5,10,15,20]); fig.tight_layout()
fig.savefig(os.path.join(F, 'n2plus1_interval_precision_en.png'), dpi=200); plt.close(fig)
print('ok')
