"""
NOTA (2026-11): este modulo implementa una normalizacion GFA->PFA de tipo clasico/Turakainen
con 2k+6 estados (B_sigma + C*J). NO es la construccion de la version actual del articulo de
Chen y Wu (arXiv:2604.07058), que usa n^2+1 estados y esta en qfa_n2plus1.py. Se conserva
como linea base de comparacion (Tabla 2 del informe). Las precisiones de intervalos se miden en BITS.

Certificacion del cutpoint de un 1gQFA mediante aritmetica de intervalos con
PRECISION ADAPTATIVA, en vez de fijar de antemano un numero de digitos
decimales "suficiente" (como se hizo en qfa_simulation.py y qfa_improved.py
con dps=80 o dps=150 elegidos a mano).

Idea central
------------
Se recalcula, para una palabra w dada, un ENCIERRO RIGUROSO (intervalo real
que contiene con certeza el valor verdadero) de

    s(w) = pi . P_w . h  -  1/2

usando aritmetica de intervalos (redondeo dirigido) en todas las etapas de
la construccion: la linealizacion 1gQFA->GFA (Proposicion 3.1) y la
conversion GFA->PFA (Teorema 3.2, con los refinamientos de las
Proposiciones 7.1/7.2). Si el intervalo resultante NO contiene el cero,
el signo de s(w) -- y por tanto la decision de cutpoint -- queda
CERTIFICADO matematicamente, sin importar que tan angosto o ancho sea el
intervalo. Si el intervalo SI contiene el cero (no se puede decidir el
signo con la precision de trabajo actual), se DUPLICA la precision y se
repite, hasta un tope maximo.

Esto certifica automaticamente con la precision minima necesaria para cada
palabra -- ni una fijada de antemano que resulte insuficiente (como el
float64 estandar) ni una exageradamente alta "por si acaso" (como los 80 o
150 digitos usados en el trabajo anterior).
"""

import numpy as np
import mpmath as mp

from qfa_simulation import hermitian_basis as hermitian_basis_np


# ---------------------------------------------------------------------
# Utilidades de intervalos
# ---------------------------------------------------------------------

def iv_pos(x):
    """max(x, 0) de forma segura para un intervalo x (real)."""
    if x.a >= 0:
        return x
    elif x.b <= 0:
        return mp.iv.mpf(0)
    else:
        return mp.iv.mpf([0, x.b])


def iv_neg(x):
    """max(-x, 0) de forma segura para un intervalo x (real)."""
    if x.b <= 0:
        return -x
    elif x.a >= 0:
        return mp.iv.mpf(0)
    else:
        return mp.iv.mpf([0, -x.a])


def float_to_iv(x):
    """Convierte un float64 (numero diadico exacto) a un intervalo de ancho
    cero en la precision de trabajo actual."""
    return mp.iv.mpf(float(x))


def complex_to_iv(z):
    return mp.iv.mpc(float_to_iv(z.real), float_to_iv(z.imag))


# ---------------------------------------------------------------------
# Base de Hermitianos en intervalos (Proposicion 3.1)
# ---------------------------------------------------------------------

def hermitian_basis_iv(n):
    """
    Construye la misma base de Hermitianos que hermitian_basis(n) en
    qfa_simulation.py, pero con entradas en aritmetica de intervalos
    (usando iv.sqrt(2) en la precision de trabajo actual, por lo que el
    ancho de estas entradas escala automaticamente con mp.iv.prec).
    """
    basis = []
    inv_sqrt2 = 1 / mp.iv.sqrt(2)

    for i in range(n):
        M = [[mp.iv.mpc(0) for _ in range(n)] for _ in range(n)]
        M[i][i] = mp.iv.mpc(1)
        basis.append(M)

    for i in range(n):
        for j in range(i + 1, n):
            M = [[mp.iv.mpc(0) for _ in range(n)] for _ in range(n)]
            M[i][j] = mp.iv.mpc(inv_sqrt2)
            M[j][i] = mp.iv.mpc(inv_sqrt2)
            basis.append(M)

    for i in range(n):
        for j in range(i + 1, n):
            M = [[mp.iv.mpc(0) for _ in range(n)] for _ in range(n)]
            M[i][j] = mp.iv.mpc(0, inv_sqrt2)
            M[j][i] = mp.iv.mpc(0, -inv_sqrt2)
            basis.append(M)

    assert len(basis) == n * n
    return basis


def mat_mul_iv(A, B):
    n, m, p = len(A), len(B), len(B[0])
    assert len(A[0]) == m
    C = [[mp.iv.mpc(0) for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(m):
            if A[i][k] == 0:
                continue
            aik = A[i][k]
            for j in range(p):
                C[i][j] += aik * B[k][j]
    return C


def conj_transpose_iv(A):
    n, m = len(A), len(A[0])
    return [[mp.iv.mpc(A[j][i].real, -A[j][i].imag) for j in range(n)] for i in range(m)]


def trace_iv(A):
    s = mp.iv.mpc(0)
    for i in range(len(A)):
        s += A[i][i]
    return s


def apply_channel_iv(kraus_list, rho):
    n = len(rho)
    out = [[mp.iv.mpc(0) for _ in range(n)] for _ in range(n)]
    for K in kraus_list:
        Kd = conj_transpose_iv(K)
        term = mat_mul_iv(mat_mul_iv(K, rho), Kd)
        for i in range(n):
            for j in range(n):
                out[i][j] += term[i][j]
    return out


def coeffs_iv(rho, basis):
    """Coeficientes reales x_i = Tr(B_i . rho) para una base Hermitiana."""
    coeffs = []
    for B in basis:
        t = trace_iv(mat_mul_iv(B, rho))
        # t deberia ser (numericamente) real; usamos la parte real como
        # intervalo real riguroso.
        coeffs.append(t.real)
    return coeffs


# ---------------------------------------------------------------------
# Conversion de un 1gQFA (dado en numpy/float64) a datos en intervalos
# ---------------------------------------------------------------------

def qfa_to_iv_data(qfa):
    """Convierte rho0, los canales (listas de Kraus) y P_acc de un
    OneWayGeneralQFA (numpy, float64) a su representacion en intervalos,
    en la precision de trabajo mp.iv.prec ACTUAL."""
    n = qfa.n
    rho0_iv = [[complex_to_iv(qfa.rho0[i, j]) for j in range(n)] for i in range(n)]
    Pacc_iv = [[complex_to_iv(qfa.P_acc[i, j]) for j in range(n)] for i in range(n)]
    channels_iv = {}
    for sigma, kraus_ops in qfa.channels.items():
        channels_iv[sigma] = [
            [[complex_to_iv(K[i, j]) for j in range(n)] for i in range(n)]
            for K in kraus_ops
        ]
    return rho0_iv, channels_iv, Pacc_iv


def fQ_interval(qfa, word, rho0_iv=None, channels_iv=None, Pacc_iv=None):
    """Encierro riguroso de f_Q(w) = Tr(P_acc . rho_w), recalculando la
    simulacion cuantica directa en intervalos (para verificacion cruzada
    independiente de la construccion GFA/PFA)."""
    if rho0_iv is None:
        rho0_iv, channels_iv, Pacc_iv = qfa_to_iv_data(qfa)
    rho = rho0_iv
    for sigma in word:
        rho = apply_channel_iv(channels_iv[sigma], rho)
    t = trace_iv(mat_mul_iv(Pacc_iv, rho))
    return t.real


def qfa_to_gfa_iv(qfa):
    """Version en intervalos de la Proposicion 3.1 (qfa_to_gfa en
    qfa_simulation.py): produce (u, A_dict, v) con entradas mp.iv.mpf,
    en la precision de trabajo mp.iv.prec ACTUAL."""
    n = qfa.n
    basis = hermitian_basis_iv(n)
    n2 = n * n
    rho0_iv, channels_iv, Pacc_iv = qfa_to_iv_data(qfa)

    u = coeffs_iv(rho0_iv, basis)  # lista de n2 intervalos reales

    A_dict = {}
    for sigma, kraus_iv in channels_iv.items():
        M = [[mp.iv.mpf(0) for _ in range(n2)] for _ in range(n2)]
        for j, Bj in enumerate(basis):
            EBj = apply_channel_iv(kraus_iv, Bj)
            col = coeffs_iv(EBj, basis)
            for i in range(n2):
                M[i][j] = col[i]
        # A_sigma = M^T
        A = [[M[j][i] for j in range(n2)] for i in range(n2)]
        A_dict[sigma] = A

    v = coeffs_iv(Pacc_iv, basis)
    return u, A_dict, v


# ---------------------------------------------------------------------
# Vector-matriz en intervalos (para el GFA y para el PFA)
# ---------------------------------------------------------------------

def vec_mat_iv(u, A):
    """u (lista de intervalos, fila) por A (lista de listas, matriz) -> lista."""
    k = len(u)
    m = len(A[0])
    out = [mp.iv.mpf(0) for _ in range(m)]
    for i in range(k):
        ui = u[i]
        if ui == 0:
            continue
        row = A[i]
        for j in range(m):
            out[j] += ui * row[j]
    return out


def dot_iv(u, v):
    s = mp.iv.mpf(0)
    for i in range(len(u)):
        s += u[i] * v[i]
    return s


def fG_interval(gfa_iv, word):
    u, A_dict, v = gfa_iv
    vec = u
    for sigma in word:
        vec = vec_mat_iv(vec, A_dict[sigma])
    return dot_iv(vec, v)


# ---------------------------------------------------------------------
# Construccion GFA -> PFA en intervalos (Teorema 3.2 + Proposiciones 7.1/7.2)
# ---------------------------------------------------------------------

def gfa_to_pfa_iv(gfa_iv, lam, tight_bound=True, per_symbol_C=True,
                   safety_margin_frac='1e-6'):
    u, A_dict, v = gfa_iv
    k = len(u)
    alphabet = list(A_dict.keys())
    lam_iv = mp.iv.mpf(str(lam))
    margin_frac = mp.mpf(safety_margin_frac)

    m = k + 1
    tildeA = {}
    for s in alphabet:
        A = A_dict[s]
        TA = [[mp.iv.mpf(0) for _ in range(m)] for _ in range(m)]
        for i in range(k):
            for j in range(k):
                TA[i][j] = A[i][j]
        TA[k][k] = mp.iv.mpf(1)
        tildeA[s] = TA
    tildeu = list(u) + [-lam_iv]
    tildev = list(v) + [mp.iv.mpf(1)]

    d = 2 * m
    hatA = {}
    for s in alphabet:
        TA = tildeA[s]
        HA = [[mp.iv.mpf(0) for _ in range(d)] for _ in range(d)]
        for i in range(m):
            for j in range(m):
                p_ij = iv_pos(TA[i][j])
                n_ij = iv_neg(TA[i][j])
                HA[i][j] = p_ij
                HA[i][m + j] = n_ij
                HA[m + i][j] = n_ij
                HA[m + i][m + j] = p_ij
        hatA[s] = HA
    up = [iv_pos(x) for x in tildeu]
    un = [iv_neg(x) for x in tildeu]
    hatu = up + un
    hatv = list(tildev) + [-x for x in tildev]

    N = d + 2
    Bmats = {}
    max_entry_by_symbol = {}
    min_entry_by_symbol = {}
    for s in alphabet:
        HA = hatA[s]
        r = [sum((HA[i][j] for j in range(d)), mp.iv.mpf(0)) for i in range(d)]
        c = [sum((HA[i][j] for i in range(d)), mp.iv.mpf(0)) for j in range(d)]
        T = sum(r, mp.iv.mpf(0))
        B = [[mp.iv.mpf(0) for _ in range(N)] for _ in range(N)]
        for i in range(d):
            B[1 + i][0] = -r[i]
            for j in range(d):
                B[1 + i][1 + j] = HA[i][j]
        B[1 + d][0] = T
        for j in range(d):
            B[1 + d][1 + j] = -c[j]
        Bmats[s] = B
        max_e = max(B[i][j].b for i in range(N) for j in range(N))
        min_e = min(B[i][j].a for i in range(N) for j in range(N))
        max_entry_by_symbol[s] = max_e
        min_entry_by_symbol[s] = min_e

    C_per_symbol = {}
    for s in alphabet:
        if tight_bound:
            C_min = max(mp.mpf(0), -min_entry_by_symbol[s])
        else:
            C_min = max(abs(max_entry_by_symbol[s]), abs(min_entry_by_symbol[s]))
        margin = max(margin_frac * max(C_min, mp.mpf(1)), mp.mpf('1e-12'))
        C_per_symbol[s] = mp.iv.mpf(C_min + margin)

    if per_symbol_C:
        C_dict = C_per_symbol
    else:
        C_shared = max(C_per_symbol.values(), key=lambda x: x.a)
        C_dict = {s: C_shared for s in alphabet}

    Pmats = {}
    for s in alphabet:
        C = C_dict[s]
        B = Bmats[s]
        denom = C * N
        P = [[(B[i][j] + C) / denom for j in range(N)] for i in range(N)]
        Pmats[s] = P

    s_norm = sum((abs(x) for x in hatu), mp.iv.mpf(0))
    pi = [mp.iv.mpf(0)] * N
    if s_norm.a > 0 or s_norm.b > 0:
        for i in range(d):
            pi[1 + i] = hatu[i] / s_norm
    else:
        pi[0] = mp.iv.mpf(1)

    g = [mp.iv.mpf(0)] * N
    for i in range(d):
        g[1 + i] = hatv[i]
    M = max((abs(x).b for x in g), default=mp.mpf(0))
    M_iv = mp.iv.mpf(M)
    if M == 0:
        h = [mp.iv.mpf('0.5')] * N
    else:
        h = [mp.iv.mpf('0.5') + g[i] / (2 * M_iv) for i in range(N)]

    return {'N': N, 'Pmats': Pmats, 'pi': pi, 'h': h, 'C_dict': C_dict}


def accept_probability_iv(pfa_iv, word):
    vec = pfa_iv['pi']
    Pmats = pfa_iv['Pmats']
    for sigma in word:
        vec = vec_mat_iv(vec, Pmats[sigma])
    return dot_iv(vec, pfa_iv['h'])


# ---------------------------------------------------------------------
# Certificacion adaptativa de precision
# ---------------------------------------------------------------------

def certify_word(gfa_data_fn, lam, word, variant, prec_start=30, prec_max=2000,
                  max_rounds=12):
    """
    Certifica la decision de cutpoint para la palabra `word` duplicando la
    precision de trabajo hasta que el intervalo de (pi.P_w.h - 1/2) no
    contenga el cero, o hasta agotar `max_rounds`.

    gfa_data_fn: funcion sin argumentos que, LLAMADA CON mp.iv.prec ya fijado
                 a la precision deseada, devuelve gfa_iv=(u,A_dict,v) para
                 esa precision (se recalcula en cada ronda, ya que la propia
                 base de Hermitianos -- via sqrt(2) -- depende de la precision).
    variant: dict con kwargs para gfa_to_pfa_iv (tight_bound, per_symbol_C).

    Devuelve un diccionario con la decision ('accept'/'reject'/'inconclusive'),
    la precision final usada, el numero de rondas, y el ancho final del
    intervalo (como medida de cuanta holgura quedo en la certificacion).
    """
    prec = prec_start
    rounds = 0
    old_prec = mp.iv.prec
    try:
        while rounds < max_rounds and prec <= prec_max:
            mp.iv.prec = prec
            gfa_iv = gfa_data_fn()
            pfa_iv = gfa_to_pfa_iv(gfa_iv, lam, **variant)
            s = accept_probability_iv(pfa_iv, word) - mp.iv.mpf('0.5')
            rounds += 1
            if s.a > 0:
                return {'decision': 'accept', 'prec': prec, 'rounds': rounds,
                        'interval': (float(s.a), float(s.b))}
            if s.b < 0:
                return {'decision': 'reject', 'prec': prec, 'rounds': rounds,
                        'interval': (float(s.a), float(s.b))}
            prec *= 2
        return {'decision': 'inconclusive', 'prec': prec // 2, 'rounds': rounds,
                'interval': (float(s.a), float(s.b))}
    finally:
        mp.iv.prec = old_prec
