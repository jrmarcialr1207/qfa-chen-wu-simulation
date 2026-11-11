"""
Comprobacion numerica de la cota universal kappa <= sqrt(n) (Proposicion 3 del informe):
para T ortogonal, max_{i,j}|K_sigma[i,j]| <= ||A_sigma||_2 <= sqrt(n) para todo canal CPTP.
Prueba 300 1gQFAs aleatorios por n (env_dim 1..4) y canales extremos (reemplazo, unitario).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation'))
from qfa_simulation import build_random_qfa, qfa_to_gfa, hermitian_basis, apply_channel, coeffs
import qfa_n2plus1 as N

def kappa_of(gfa, lam=0.45):
    pf = N.pfa_from_gfa(gfa, lam, 'float', T_mode='orthogonal', eps_rule='paper', per_symbol=False)
    return max(pf['kappa'].values()), max(np.linalg.norm(A, 2) for A in gfa.A_dict.values())

worst = {}
for n in range(2, 7):
    mx_k = mx_op = 0.0
    for t in range(300):
        rng = np.random.default_rng(10_000 * n + t)
        qfa = build_random_qfa(n, ('a', 'b'), rng, env_dim=1 + t % 4)
        k, op = kappa_of(qfa_to_gfa(qfa))
        mx_k, mx_op = max(mx_k, k), max(mx_op, op)
    # canal de reemplazo rho -> Tr(rho)|0><0| (extremo para la norma HS) y unitario
    qfa = build_random_qfa(n, ('a', 'b'), np.random.default_rng(1))
    basis = hermitian_basis(n)
    rep = [np.outer(np.eye(n)[0], np.eye(n)[j]).astype(complex) for j in range(n)]  # Kraus de reemplazo
    qfa.channels['a'] = rep
    k, op = kappa_of(qfa_to_gfa(qfa))
    worst[n] = (mx_k, mx_op, k, op)
    print(f'n={n}: sqrt(n)={np.sqrt(n):.3f}  aleatorios: max kappa={mx_k:.3f}, max ||A||_2={mx_op:.3f}  | reemplazo: kappa={k:.3f}, ||A||_2={op:.3f}')
    assert mx_k <= np.sqrt(n) + 1e-9 and k <= np.sqrt(n) + 1e-9
print('cota universal verificada')
