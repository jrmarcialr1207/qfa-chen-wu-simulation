"""
NOTA (2026-11): este modulo implementa una normalizacion GFA->PFA de tipo clasico/Turakainen
con 2k+6 estados (B_sigma + C*J). NO es la construccion de la version actual del articulo de
Chen y Wu (arXiv:2604.07058), que usa n^2+1 estados y esta en qfa_n2plus1.py. Se conserva
como linea base de comparacion (Tabla 2 del informe). Las precisiones de intervalos se miden en BITS.

Extensión de qfa_simulation.py: variantes de la construcción del Teorema 3.2
(Chen & Wu, arXiv:2604.07058) que buscan minimizar la constante de
normalización C sin alterar la cota de 2k+6 estados.

Justificación algebraica de por qué esto es válido
----------------------------------------------------
En la construcción original, cada matriz auxiliar B_sigma (de la etapa iii)
se diseña deliberadamente con SUMA DE FILA Y DE COLUMNA EXACTAMENTE CERO.
Esto implica, para la matriz de unos J (N x N):

    B_sigma . J = 0      (porque las filas de B_sigma suman 0)
    J . B_sigma = 0      (porque las columnas de B_sigma suman 0)
    J . J       = N . J

Por lo tanto, si se usa una constante DISTINTA por símbolo, C_sigma, con
P_sigma = (B_sigma + C_sigma . J) / (C_sigma . N), el producto de dos
símbolos telescopía exactamente igual que con una C compartida:

    P_s1 . P_s2 = [B_s1.B_s2] / (C_s1 . C_s2 . N^2)  +  J / N

(los términos cruzados B_s1.J y J.B_s2 se anulan). Por inducción, para
una palabra w = s1...sL:

    P_w = B_w / ( N^L . prod_i C_{s_i} )  +  J / N

Esto generaliza la fórmula del artículo (que corresponde al caso
C_{s_i} = C para todo símbolo) y muestra que CADA SÍMBOLO PUEDE TENER SU
PROPIA CONSTANTE DE NORMALIZACIÓN, siempre que sea estrictamente mayor que
el valor necesario para garantizar que P_sigma sea estocástica y
estrictamente positiva. Esto no cambia el número de estados (sigue siendo
N = 2k+4 estados "principales", más 2 absorbentes = 2k+6 en total): sólo
cambia el valor escalar C_sigma usado en la normalización de cada símbolo.

Esta extensión implementa y verifica dos mejoras independientes:

  (A) Cota ajustada de C: en vez de C > max_{i,j,sigma} |B_sigma[i,j]|,
      se usa la cota mínima necesaria C_sigma > max_{i,j}(-B_sigma[i,j])
      (el valor más negativo, con signo cambiado) -- que puede ser hasta
      la mitad de la cota anterior si los valores positivos de B_sigma son
      más grandes en magnitud que los negativos.

  (B) C específico por símbolo, en vez de un único C compartido por todo
      el alfabeto (el máximo entre todos los símbolos).

Ambas mejoras reducen el producto C_sigma . N efectivo de la construcción
y, por tanto, ralentizan el decaimiento geométrico de la señal de cutpoint
descrito en el informe anterior, permitiendo simular palabras más largas
antes de agotar la precisión de una aritmética de punto flotante estándar.
"""

import numpy as np
import mpmath as mp
from qfa_simulation import GFA, PFA


def gfa_to_pfa_improved(gfa, lam, tight_bound=True, per_symbol_C=True,
                         safety_margin_frac=1e-6):
    """
    Variante de gfa_to_pfa que minimiza la(s) constante(s) de normalización.

    Parámetros
    ----------
    tight_bound : si True, usa C_sigma > max(0, -min_ij B_sigma[i,j])
                  (justo lo necesario para positividad estricta) en vez de
                  la cota conservadora max_ij |B_sigma[i,j]|.
    per_symbol_C : si True, cada símbolo usa su propia constante C_sigma;
                   si False, se usa el máximo de las C_sigma individuales
                   compartido por todo el alfabeto (equivalente al
                   comportamiento del artículo original, pero ya con la
                   posible mejora de tight_bound).
    safety_margin_frac : margen relativo de seguridad sobre la cota mínima
                          (evita entradas exactamente cero en P_sigma).
    """
    u, A_dict, v = gfa.u, gfa.A_dict, gfa.v
    k = gfa.k
    alphabet = gfa.alphabet()

    # --- Paso 1: cutpoint -> 0, dimensión m = k+1 (idéntico al original) ---
    m = k + 1
    tildeA = {}
    for s in alphabet:
        A = A_dict[s]
        TA = np.zeros((m, m))
        TA[:k, :k] = A
        TA[k, k] = 1.0
        tildeA[s] = TA
    tildeu = np.concatenate([u, [-lam]])
    tildev = np.concatenate([v, [1.0]])

    # --- Paso 2: partes no negativas, dimensión d = 2m (idéntico) ---
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
    up = np.maximum(tildeu, 0.0)
    un = np.maximum(-tildeu, 0.0)
    hatu = np.concatenate([up, un])
    hatv = np.concatenate([tildev, -tildev])

    # --- Paso 3: construcción de B_sigma (idéntica) ---
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

    # --- Elección de C_sigma: aquí está la mejora ---
    C_per_symbol = {}
    for s in alphabet:
        B = Bmats[s]
        if tight_bound:
            min_entry = np.min(B)          # el valor más negativo (o >=0)
            C_min = max(0.0, -min_entry)   # cota MÍNIMA necesaria
        else:
            C_min = np.max(np.abs(B))      # cota conservadora (artículo)
        margin = max(safety_margin_frac * max(C_min, 1.0), 1e-9)
        C_per_symbol[s] = C_min + margin

    if per_symbol_C:
        C_dict = C_per_symbol
    else:
        C_shared = max(C_per_symbol.values())
        C_dict = {s: C_shared for s in alphabet}

    J = np.ones((N, N))
    Pmats = {}
    for s in alphabet:
        C = C_dict[s]
        Pmats[s] = (Bmats[s] + C * J) / (C * N)
        assert np.all(Pmats[s] > 0), f"P_{s} no es estrictamente positiva"
        assert np.allclose(Pmats[s].sum(axis=1), 1.0), f"P_{s} no es estocástica"

    # --- distribución inicial y vector de fin de palabra (idénticos) ---
    s_norm = np.sum(np.abs(hatu))
    pi = np.zeros(N)
    if s_norm > 0:
        pi[1:1 + d] = hatu / s_norm
    else:
        pi[0] = 1.0

    g = np.zeros(N)
    g[1:1 + d] = hatv
    M = np.max(np.abs(g))
    if M == 0:
        h = np.full(N, 0.5)
    else:
        h = 0.5 * np.ones(N) + g / (2 * M)
    h = np.clip(h, 0.0, 1.0)

    pfa = PFA(N, Pmats, pi, h)
    pfa._g = g
    pfa.C_dict = C_dict  # guardado para diagnóstico/reporte
    return pfa


def gfa_to_pfa_improved_hp(gfa, lam, tight_bound=True, per_symbol_C=True,
                            safety_margin_frac=mp.mpf('1e-6'), dps=80):
    """
    Versión en aritmética de precisión arbitraria (mpmath) de
    gfa_to_pfa_improved, construida enteramente en alta precisión desde el
    primer paso (por la misma razón explicada en qfa_simulation.py: si las
    matrices se calculan primero en float64, el redondeo ya introducido
    contamina la señal antes de que la alta precisión pueda intervenir).
    """
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        u = mp.matrix([mp.mpf(float(x)) for x in gfa.u])
        A_dict = {s: mp.matrix(gfa.A_dict[s].tolist()) for s in gfa.alphabet()}
        v = mp.matrix([mp.mpf(float(x)) for x in gfa.v])
        lam_mp = mp.mpf(lam)
        k = gfa.k
        alphabet = gfa.alphabet()

        m = k + 1
        tildeA = {}
        for s in alphabet:
            A = A_dict[s]
            TA = mp.zeros(m, m)
            for i in range(k):
                for j in range(k):
                    TA[i, j] = A[i, j]
            TA[k, k] = mp.mpf(1)
            tildeA[s] = TA
        tildeu = mp.matrix([u[i] for i in range(k)] + [-lam_mp])
        tildev = mp.matrix([v[i] for i in range(k)] + [mp.mpf(1)])

        d = 2 * m

        def pos(x):
            return x if x > 0 else mp.mpf(0)

        def neg(x):
            return -x if x < 0 else mp.mpf(0)

        hatA = {}
        for s in alphabet:
            TA = tildeA[s]
            HA = mp.zeros(d, d)
            for i in range(m):
                for j in range(m):
                    p_ij = pos(TA[i, j])
                    n_ij = neg(TA[i, j])
                    HA[i, j] = p_ij
                    HA[i, m + j] = n_ij
                    HA[m + i, j] = n_ij
                    HA[m + i, m + j] = p_ij
            hatA[s] = HA
        up = [pos(tildeu[i]) for i in range(m)]
        un = [neg(tildeu[i]) for i in range(m)]
        hatu = mp.matrix(up + un)
        hatv = mp.matrix([tildev[i] for i in range(m)] + [-tildev[i] for i in range(m)])

        N = d + 2
        Bmats = {}
        for s in alphabet:
            HA = hatA[s]
            r = [sum(HA[i, j] for j in range(d)) for i in range(d)]
            c = [sum(HA[i, j] for i in range(d)) for j in range(d)]
            T = sum(r)
            B = mp.zeros(N, N)
            for i in range(d):
                B[1 + i, 0] = -r[i]
                for j in range(d):
                    B[1 + i, 1 + j] = HA[i, j]
            B[1 + d, 0] = T
            for j in range(d):
                B[1 + d, 1 + j] = -c[j]
            Bmats[s] = B

        C_per_symbol = {}
        for s in alphabet:
            B = Bmats[s]
            if tight_bound:
                min_entry = min(B[i, j] for i in range(N) for j in range(N))
                C_min = max(mp.mpf(0), -min_entry)
            else:
                C_min = max(abs(B[i, j]) for i in range(N) for j in range(N))
            margin = max(safety_margin_frac * max(C_min, mp.mpf(1)), mp.mpf('1e-12'))
            C_per_symbol[s] = C_min + margin

        if per_symbol_C:
            C_dict = C_per_symbol
        else:
            C_shared = max(C_per_symbol.values())
            C_dict = {s: C_shared for s in alphabet}

        Pmats = {}
        for s in alphabet:
            C = C_dict[s]
            B = Bmats[s]
            P = mp.zeros(N, N)
            denom = C * N
            for i in range(N):
                for j in range(N):
                    P[i, j] = (B[i, j] + C) / denom
            Pmats[s] = P

        s_norm = sum(abs(x) for x in hatu)
        pi_list = [mp.mpf(0)] * N
        if s_norm > 0:
            for i in range(d):
                pi_list[1 + i] = hatu[i] / s_norm
        else:
            pi_list[0] = mp.mpf(1)
        pi = mp.matrix([pi_list])

        g_list = [mp.mpf(0)] * N
        for i in range(d):
            g_list[1 + i] = hatv[i]
        M = max(abs(x) for x in g_list)
        if M == 0:
            h_list = [mp.mpf('0.5')] * N
        else:
            h_list = [mp.mpf('0.5') + g_list[i] / (2 * M) for i in range(N)]
        h = mp.matrix(h_list)

        from qfa_simulation import PFA_HP
        pfa = PFA_HP(N, Pmats, pi, h)
        pfa.C_dict = C_dict
        return pfa
    finally:
        mp.mp.dps = old_dps
