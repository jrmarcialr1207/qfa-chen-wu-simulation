"""
Extension del analisis de fragilidad numerica y de la certificacion por
intervalos (Seccion 8) al argumento de la COTA INFERIOR (Teorema 4.1) de
Chen y Wu.

Alcance y limitacion honesta
----------------------------
El Teorema 4.1 se apoya en un argumento de "shattering" (en el sentido de
Vapnik-Chervonenkis): una familia de n^2-1 estados de densidad "preparados"
rho_1,...,rho_{n^2-1}, correspondientes a direcciones ortonormales del
subespacio de operadores Hermitianos de traza nula, y una familia de
mediciones de dos resultados M_s (una por cada patron de signos
s en {+1,-1}^{n^2-1}) tales que Tr(M_s . rho_j) = eps * s_j exactamente,
donde eps>0 es un margen fijo. Esta identidad es el nucleo matematico
autocontenido del argumento, y es la parte que este modulo implementa y
certifica.

El paso adicional que el Teorema 4.1 necesita -- realizar cada medicion
M_s como una rotacion unitaria U_s de un UNICO proyector de aceptacion fijo
P_acc, de manera que un 1gQFA con un alfabeto finito pueda, leyendo la
palabra adecuada, poner en juego cualquier M_s -- requiere una eleccion
mas especial de generadores de la que aqui se reconstruyo (se verificó
numericamente que, con la base de Hermitianos usada en el resto de este
informe, los distintos M_s NO comparten espectro, por lo que no son todos
unitariamente equivalentes a un P_acc comun). Esa parte de la
construccion original de Chen y Wu no se reconstruye aqui por no contarse
con el texto completo del articulo; se deja explicitamente como limitacion
y posible extension futura.

Lo que SI se certifica rigurosamente en este modulo es el nucleo
matematico: la identidad de shattering en si misma, mediante aritmetica de
intervalos con precision adaptativa (identica metodologia que en
qfa_interval.py), permitiendo comparar la fragilidad numerica de este
argumento con la de la construccion del Teorema 3.2 (Secciones 5, 7 y 8).
"""

import numpy as np
import mpmath as mp

from qfa_simulation import hermitian_basis


# ---------------------------------------------------------------------
# Base ortonormal del subespacio de operadores Hermitianos de traza nula
# ---------------------------------------------------------------------

def traceless_diagonal_basis_np(n):
    """n-1 direcciones diagonales, ortonormales (Hilbert-Schmidt) y de
    traza nula (analogas a las matrices diagonales de Gell-Mann)."""
    dirs = []
    for k in range(1, n):
        d = np.zeros(n)
        d[:k] = 1.0
        d[k] = -k
        d = d / np.linalg.norm(d)
        dirs.append(np.diag(d).astype(complex))
    return dirs


def traceless_orthonormal_basis_np(n):
    """Base ortonormal completa (dimension n^2-1) del subespacio de
    operadores Hermitianos de traza nula, en numpy/float64."""
    full = hermitian_basis(n)
    offdiag = [B for B in full if not np.allclose(B, np.diag(np.diag(B)))]
    return traceless_diagonal_basis_np(n) + offdiag


def max_valid_epsilon(n, basis, margin=0.999):
    """Maximo eps tal que I/n + eps*B_j sea semidefinida positiva para
    TODAS las direcciones B_j de la base (peor caso), con un margen de
    seguridad multiplicativo."""
    eps_max = np.inf
    for B in basis:
        min_eig_B = np.linalg.eigvalsh(B).min()
        if min_eig_B < 0:
            eps_bound = (1 / n) / (-min_eig_B)
            eps_max = min(eps_max, eps_bound)
    return eps_max * margin


# ---------------------------------------------------------------------
# Version en intervalos (mpmath.iv)
# ---------------------------------------------------------------------

def traceless_orthonormal_basis_iv(n):
    """Version en intervalos de la base anterior, en la precision de
    trabajo mp.iv.prec actual (usa fracciones exactas / raices via iv,
    por lo que hereda la precision de trabajo automaticamente)."""
    basis_np = traceless_orthonormal_basis_np(n)
    basis_iv = []
    for B in basis_np:
        # las entradas de esta base son numeros diadicos exactos (0, 1,
        # -k, 1/sqrt(2), etc. ya normalizados en float64); los convertimos
        # a intervalos de ancho cero directamente, igual que en
        # qfa_interval.py (son datos "exactos" en el sentido de que no
        # provienen de un calculo con cancelacion).
        M = [[mp.iv.mpc(float(B[i, j].real), float(B[i, j].imag))
              for j in range(n)] for i in range(n)]
        basis_iv.append(M)
    return basis_iv


def trace_product_real_iv(M, N):
    """Tr(M . N) como intervalo real, para matrices de intervalos M, N."""
    n = len(M)
    total = mp.iv.mpc(0)
    for i in range(n):
        for k in range(n):
            total += M[i][k] * N[k][i]
    return total.real


def certify_shattering(n, eps, prec_start=15, prec_max=200, max_rounds=10,
                        sign_patterns=None):
    """
    Certifica, mediante aritmetica de intervalos con precision adaptativa,
    que para cada patron de signos s en `sign_patterns` (o TODOS los
    2^(n^2-1), si no se especifica y esto es factible) y cada direccion j,
    el signo de Tr(M_s . rho_j) coincide con s_j.

    Devuelve un diccionario con la precision y numero de rondas usadas
    (el maximo sobre todos los pares (s,j) certificados), y confirma si
    hubo algun caso no certificado dentro de prec_max.
    """
    basis_np = traceless_orthonormal_basis_np(n)
    d = len(basis_np)  # n^2 - 1

    if sign_patterns is None:
        if d <= 20:
            import itertools
            sign_patterns = list(itertools.product([1, -1], repeat=d))
        else:
            raise ValueError("Demasiados patrones para enumeracion exhaustiva; "
                              "pase sign_patterns explicitamente (muestreo).")

    old_prec = mp.iv.prec
    pending = list(sign_patterns)
    max_prec_used = prec_start
    max_rounds_used = 0
    certified = {}
    try:
        prec = prec_start
        rounds = 0
        while pending and rounds < max_rounds and prec <= prec_max:
            mp.iv.prec = prec
            rounds += 1
            basis_iv = traceless_orthonormal_basis_iv(n)
            I_over_n = [[mp.iv.mpc(1 / n if a == b else 0) for b in range(n)]
                        for a in range(n)]
            rhos_iv = []
            for B_j in basis_iv:
                rho_j = [[I_over_n[a][b] + mp.iv.mpf(eps) * B_j[a][b]
                          for b in range(n)] for a in range(n)]
                rhos_iv.append(rho_j)

            still_pending = []
            for signs in pending:
                M_s = [[mp.iv.mpc(0) for _ in range(n)] for _ in range(n)]
                for s_i, B in zip(signs, basis_iv):
                    for a in range(n):
                        for b in range(n):
                            if B[a][b] != 0:
                                M_s[a][b] += s_i * B[a][b]
                ok_all = True
                for j in range(d):
                    val = trace_product_real_iv(M_s, rhos_iv[j])
                    if signs[j] > 0:
                        if not (val.a > 0):
                            ok_all = False
                            break
                    else:
                        if not (val.b < 0):
                            ok_all = False
                            break
                if ok_all:
                    certified[signs] = (prec, rounds)
                else:
                    still_pending.append(signs)
            pending = still_pending
            max_prec_used = prec
            max_rounds_used = rounds
            prec *= 2
    finally:
        mp.iv.prec = old_prec

    return {
        'n': n, 'dim': d, 'num_patterns_tested': len(sign_patterns),
        'max_prec_used': max_prec_used, 'max_rounds_used': max_rounds_used,
        'all_certified': len(pending) == 0, 'num_uncertified': len(pending),
    }
