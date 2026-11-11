"""
NOTA (2026-11): este modulo implementa una normalizacion GFA->PFA de tipo clasico/Turakainen
con 2k+6 estados (B_sigma + C*J). NO es la construccion de la version actual del articulo de
Chen y Wu (arXiv:2604.07058), que usa n^2+1 estados y esta en qfa_n2plus1.py. Se conserva
como linea base de comparacion (Tabla 2 del informe). Las precisiones de intervalos se miden en BITS.

Implementación de la construcción de:
Zeyu Chen y Junde Wu, "The Quadratic State Cost of Classical Simulation of
One-Way Quantum Finite Automata", arXiv:2604.07058 (2026).

Este script implementa, de forma numérica y verificable:

  1. Proposición 3.1: linealización de un 1gQFA (autómata cuántico general de
     una vía, "measure-once") de n estados en un GFA (autómata generalizado
     finito) de n^2 estados, usando una base ortonormal de operadores
     Hermitianos respecto al producto interno de Hilbert-Schmidt.

  2. Teorema 3.2: conversión de un GFA de k estados con cutpoint estricto
     lambda en un PFA (autómata probabilístico finito) de a lo sumo 2k+6
     estados, con cutpoint normalizado a 1/2.

  3. Corolario 3.3: la composición de ambos pasos simula cualquier 1gQFA de
     n estados mediante un PFA de a lo sumo 2n^2+6 estados, preservando
     exactamente el lenguaje de cutpoint estricto.

El script construye 1gQFAs aleatorios (canales CPTP vía el método de
Stinespring), ejecuta la construcción completa, y verifica sobre muchas
palabras aleatorias que:

    f_P(w) > 1/2   <=>   f_Q(w) > lambda

tal como garantiza el Teorema 3.2, además de verificar que la linealización
intermedia (el GFA) reproduce exactamente la probabilidad de aceptación del
1gQFA original.

HALLAZGO NUMÉRICO: la construcción del Teorema 3.2 satisface exactamente
pi P_w h = (f_G(w) - lambda) / (s * (C*N)^|w|), es decir, la "señal" que
distingue el cutpoint decae GEOMÉTRICAMENTE con la longitud de la palabra
(a tasa 1/(C*N), donde C*N puede ser de cientos o miles para autómatas de
pocos estados). En aritmética de punto flotante de 64 bits esa señal cae
por debajo de la precisión de máquina (~1e-16) después de apenas unas pocas
letras, produciendo discrepancias de cutpoint que NO son un error de la
construcción sino una limitación de la aritmética de precisión fija. Este
script lo demuestra construyendo el PFA dos veces -- una en float64 (para
exhibir el problema) y otra enteramente en aritmética de precisión
arbitraria con mpmath (que sí reproduce el teorema de forma exacta en todos
los casos probados).
"""

import itertools
import numpy as np
import mpmath as mp


# ---------------------------------------------------------------------------
# 1. Utilidades cuánticas: base de Hermitianos, canales CPTP aleatorios
# ---------------------------------------------------------------------------

def hermitian_basis(n):
    """
    Devuelve una base ortonormal (respecto al producto interno de
    Hilbert-Schmidt <A,B> = Tr(A^dagger B)) del espacio real de operadores
    Hermitianos n x n. Tiene dimensión n^2, tal como usa la Proposición 3.1.
    """
    basis = []

    # Elementos diagonales E_ii
    for i in range(n):
        B = np.zeros((n, n), dtype=complex)
        B[i, i] = 1.0
        basis.append(B)

    # Parte simétrica fuera de la diagonal: (E_ij + E_ji)/sqrt(2)
    for i in range(n):
        for j in range(i + 1, n):
            B = np.zeros((n, n), dtype=complex)
            B[i, j] = 1 / np.sqrt(2)
            B[j, i] = 1 / np.sqrt(2)
            basis.append(B)

    # Parte antisimétrica (Hermitiana) fuera de la diagonal:
    # i*(E_ij - E_ji)/sqrt(2)
    for i in range(n):
        for j in range(i + 1, n):
            B = np.zeros((n, n), dtype=complex)
            B[i, j] = 1j / np.sqrt(2)
            B[j, i] = -1j / np.sqrt(2)
            basis.append(B)

    assert len(basis) == n * n
    return basis


def check_orthonormal(basis, tol=1e-9):
    """Verifica que la base sea ortonormal respecto a Tr(A^dagger B)."""
    n2 = len(basis)
    gram = np.zeros((n2, n2), dtype=complex)
    for i, Bi in enumerate(basis):
        for j, Bj in enumerate(basis):
            gram[i, j] = np.trace(Bi.conj().T @ Bj)
    assert np.allclose(gram, np.eye(n2), atol=tol), "La base no es ortonormal"


def coeffs(rho, basis):
    """Coeficientes reales x_i = Tr(B_i rho) de un operador Hermitiano rho."""
    return np.array([np.real(np.trace(B @ rho)) for B in basis])


def random_unitary(dim, rng):
    """Unitaria Haar-aleatoria de dimensión dim, vía QR de una matriz Ginibre."""
    Z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diagonal(R)
    ph = d / np.abs(d)
    return Q * ph


def random_cptp_channel(n, env_dim, rng):
    """
    Genera un canal CPTP aleatorio en dimensión n mediante el método de
    Stinespring: una unitaria aleatoria en el espacio conjunto
    sistema-ambiente, con el ambiente inicializado en |0>, seguida de traza
    parcial sobre el ambiente. Devuelve la lista de operadores de Kraus.
    """
    dim = n * env_dim
    U = random_unitary(dim, rng)
    # índice combinado = sistema * env_dim + ambiente
    U4 = U.reshape(n, env_dim, n, env_dim)
    kraus_ops = [U4[:, i, :, 0] for i in range(env_dim)]

    # sanity check: sum K_i^dagger K_i = I
    total = sum(K.conj().T @ K for K in kraus_ops)
    assert np.allclose(total, np.eye(n), atol=1e-8), "Kraus ops no son CPTP"
    return kraus_ops


def apply_channel(kraus_ops, rho):
    """Aplica un canal CPTP (dado por operadores de Kraus) a rho."""
    return sum(K @ rho @ K.conj().T for K in kraus_ops)


# ---------------------------------------------------------------------------
# 2. El 1gQFA (autómata cuántico general de una vía, measure-once)
# ---------------------------------------------------------------------------

class OneWayGeneralQFA:
    """
    Q = (H, Sigma, rho0, {E_sigma}, P_acc)  según la Sección 2.2 del artículo.
    """

    def __init__(self, n, rho0, channels, P_acc):
        self.n = n
        self.rho0 = rho0
        self.channels = channels  # dict: symbol -> lista de operadores de Kraus
        self.P_acc = P_acc

    def alphabet(self):
        return list(self.channels.keys())

    def run(self, word):
        """Devuelve f_Q(w) = Tr(P_acc rho_w)."""
        rho = self.rho0
        for sigma in word:
            rho = apply_channel(self.channels[sigma], rho)
        return float(np.real(np.trace(self.P_acc @ rho)))


# ---------------------------------------------------------------------------
# 3. Proposición 3.1: 1gQFA -> GFA (linealización de dimensión n^2)
# ---------------------------------------------------------------------------

class GFA:
    """G = (S, Sigma, u, {A_sigma}, v), representado con vectores/matrices reales."""

    def __init__(self, u, A_dict, v):
        self.u = u
        self.A_dict = A_dict
        self.v = v
        self.k = len(u)

    def alphabet(self):
        return list(self.A_dict.keys())

    def run(self, word):
        """Devuelve f_G(w) = u A_{sigma_1} ... A_{sigma_m} v."""
        vec = self.u.copy()
        for sigma in word:
            vec = vec @ self.A_dict[sigma]
        return float(vec @ self.v)


def qfa_to_gfa(qfa):
    """
    Implementa la Proposición 3.1: construye el GFA de n^2 estados que
    reproduce exactamente f_Q(w) mediante la linealización de Hilbert-Schmidt
    de la evolución del 1gQFA.
    """
    n = qfa.n
    basis = hermitian_basis(n)
    check_orthonormal(basis)
    n2 = n * n

    # nu = coeficientes de rho0
    nu = coeffs(qfa.rho0, basis)

    # A_sigma = M_sigma^T, donde M_sigma es la matriz del canal en la base
    A_dict = {}
    for sigma, kraus_ops in qfa.channels.items():
        M = np.zeros((n2, n2))
        for j, Bj in enumerate(basis):
            EBj = apply_channel(kraus_ops, Bj)
            M[:, j] = coeffs(EBj, basis)
        A_dict[sigma] = M.T

    # v_i = Tr(P_acc B_i)
    v = coeffs(qfa.P_acc, basis)

    return GFA(nu, A_dict, v)


# ---------------------------------------------------------------------------
# 4. Teorema 3.2: GFA -> PFA (cutpoint estricto lambda -> cutpoint 1/2)
# ---------------------------------------------------------------------------

class PFA:
    """
    PFA explícito de N+2 estados (N estados "principales" + 2 estados
    absorbentes q_acc, q_rej), construido exactamente según la demostración
    del Teorema 3.2, incluyendo el paso de fin de palabra (end-marker).
    """

    def __init__(self, N, Pmats, pi, h):
        self.N = N          # número de estados "principales" (antes del end-marker)
        self.Pmats = Pmats  # dict: symbol -> matriz estocástica N x N
        self.pi = pi        # distribución inicial sobre los N estados principales
        self.h = h          # prob. de ir a q_acc en el end-marker, por estado
        self.total_states = N + 2  # + q_acc, q_rej

    def accept_probability(self, word):
        """
        Calcula f_P(w) = pi P_w h simulando el autómata completo de N+2
        estados (incluyendo los dos estados absorbentes), letra por letra,
        y aplicando al final la transición de fin de palabra.
        """
        N = self.N
        # distribución extendida sobre los N+2 estados
        dist = np.zeros(N + 2)
        dist[:N] = self.pi

        for sigma in word:
            P = self.Pmats[sigma]
            new_dist = np.zeros(N + 2)
            # las transiciones normales sólo mueven masa entre los N estados
            # principales; los 2 estados absorbentes son puntos fijos.
            new_dist[:N] = dist[:N] @ P
            new_dist[N] = dist[N]
            new_dist[N + 1] = dist[N + 1]
            dist = new_dist

        # transición de fin de palabra: estado i (< N) -> q_acc con prob h[i]
        acc_mass = dist[N] + np.dot(dist[:N], self.h)
        return float(acc_mass)

    def accept_probability_formula(self, word, g):
        """
        Calcula f_P(w) mediante la fórmula compacta pi P_w h del artículo,
        usada como verificación cruzada independiente del recorrido completo.
        """
        vec = self.pi.copy()
        for sigma in word:
            vec = vec @ self.Pmats[sigma]
        return float(vec @ self.h)


class PFA_HP:
    """
    Versión de PFA construida enteramente con aritmética de precisión
    arbitraria (mpmath), desde el primer paso de la demostración del
    Teorema 3.2. Esto es necesario porque, si las matrices P_sigma se
    calculan primero en float64 y sólo se multiplican en alta precisión
    después, el redondeo de ~1e-16 ya introducido en cada entrada de
    P_sigma contamina la señal (f_G(w) - lambda), que decae como
    (C*N)^{-|w|} y puede quedar por debajo de ese redondeo para palabras
    de longitud moderada. Construir todo en alta precisión desde el
    principio evita ese problema.
    """

    def __init__(self, N, Pmats, pi, h):
        self.N = N
        self.Pmats = Pmats  # dict: symbol -> mp.matrix (N x N), entradas mp.mpf
        self.pi = pi        # mp.matrix fila 1xN
        self.h = h          # mp.matrix columna Nx1
        self.total_states = N + 2

    def accept_probability(self, word):
        dist = self.pi
        for sigma in word:
            dist = dist * self.Pmats[sigma]
        return (dist * self.h)[0, 0]


def gfa_to_pfa_hp(gfa, lam, dps=80, safety_margin_frac=mp.mpf('0.1')):
    """
    Misma construcción que gfa_to_pfa (Teorema 3.2), pero realizada
    enteramente con aritmética mpmath de `dps` dígitos decimales.
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

        # --- Paso 1: cutpoint -> 0, dimensión m = k+1 ---
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

        # --- Paso 2: partes no negativas, dimensión d = 2m ---
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

        # --- Paso 3: normalización a matrices estocásticas, dimensión N = d+2 ---
        N = d + 2
        Bmats = {}
        max_entry = mp.mpf(0)
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
            for i in range(N):
                for j in range(N):
                    if abs(B[i, j]) > max_entry:
                        max_entry = abs(B[i, j])

        C = max_entry * (1 + safety_margin_frac) + mp.mpf('1e-6')
        Pmats = {}
        for s in alphabet:
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
        pi = mp.matrix([pi_list])  # fila 1 x N

        g_list = [mp.mpf(0)] * N
        for i in range(d):
            g_list[1 + i] = hatv[i]

        M = max(abs(x) for x in g_list)
        if M == 0:
            h_list = [mp.mpf('0.5')] * N
        else:
            h_list = [mp.mpf('0.5') + g_list[i] / (2 * M) for i in range(N)]
        h = mp.matrix(h_list)  # columna N x 1

        return PFA_HP(N, Pmats, pi, h)
    finally:
        mp.mp.dps = old_dps


def gfa_to_pfa(gfa, lam, safety_margin=1.0):
    """
    Implementa la demostración del Teorema 3.2 paso a paso:

      1. Traslada el cutpoint a 0 añadiendo un estado extra (dimensión m=k+1).
      2. Separa la matriz (posiblemente con signos) en partes no negativas,
         duplicando la dimensión (d = 2m).
      3. Normaliza simultáneamente todas las matrices a estocásticas mediante
         la construcción B_sigma de suma de fila/columna cero más C*J.
      4. Añade el paso de fin de palabra que convierte el vector final en
         probabilidades de aceptar/rechazar.

    Devuelve un objeto PFA con N = 2k+4 estados principales
    (y por lo tanto N+2 = 2k+6 estados en total, tal como en el Teorema 3.2).
    """
    u, A_dict, v = gfa.u, gfa.A_dict, gfa.v
    k = gfa.k
    alphabet = gfa.alphabet()

    # --- Paso 1: cutpoint -> 0, dimensión m = k+1 ---
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

    # --- Paso 2: partes no negativas, dimensión d = 2m ---
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

    # --- Paso 3: normalización a matrices estocásticas, dimensión N = d+2 ---
    N = d + 2
    Bmats = {}
    for s in alphabet:
        HA = hatA[s]
        r = HA.sum(axis=1)   # sumas de fila (d,)
        c = HA.sum(axis=0)   # sumas de columna (d,)
        T = r.sum()          # = c.sum()
        B = np.zeros((N, N))
        B[1:1 + d, 0] = -r
        B[1:1 + d, 1:1 + d] = HA
        B[1 + d, 0] = T
        B[1 + d, 1:1 + d] = -c
        Bmats[s] = B

    max_entry = max(np.max(np.abs(B)) for B in Bmats.values())
    C = max_entry + safety_margin  # C tal que |entradas de B_sigma| < C
    J = np.ones((N, N))
    Pmats = {s: (Bmats[s] + C * J) / (C * N) for s in alphabet}

    # sanity check: cada P_sigma debe ser estocástica por filas y positiva
    for s in alphabet:
        P = Pmats[s]
        assert np.all(P > 0), f"P_{s} no es estrictamente positiva"
        assert np.allclose(P.sum(axis=1), 1.0), f"P_{s} no es estocástica"

    # --- distribución inicial y vector de fin de palabra ---
    s_norm = np.sum(np.abs(hatu))
    pi = np.zeros(N)
    if s_norm > 0:
        pi[1:1 + d] = hatu / s_norm
    else:
        pi[0] = 1.0  # caso degenerado: lenguaje vacío

    g = np.zeros(N)
    g[1:1 + d] = hatv

    M = np.max(np.abs(g))
    if M == 0:
        h = np.full(N, 0.5)
    else:
        h = 0.5 * np.ones(N) + g / (2 * M)
    h = np.clip(h, 0.0, 1.0)

    pfa = PFA(N, Pmats, pi, h)
    pfa._g = g  # guardado para la verificación cruzada con la fórmula compacta
    return pfa


# ---------------------------------------------------------------------------
# 5. Experimento de verificación
# ---------------------------------------------------------------------------

def build_random_qfa(n, alphabet, rng, env_dim=2, acc_rank=None):
    """Construye un 1gQFA aleatorio de n estados sobre el alfabeto dado."""
    rho0 = np.zeros((n, n), dtype=complex)
    rho0[0, 0] = 1.0  # estado inicial puro |0><0|

    channels = {sigma: random_cptp_channel(n, env_dim, rng) for sigma in alphabet}

    if acc_rank is None:
        acc_rank = max(1, n // 2)
    P_acc = np.zeros((n, n), dtype=complex)
    for i in range(acc_rank):
        P_acc[i, i] = 1.0

    return OneWayGeneralQFA(n, rho0, channels, P_acc)


def all_words_up_to(alphabet, max_len):
    words = ['']
    for length in range(1, max_len + 1):
        for combo in itertools.product(alphabet, repeat=length):
            words.append(''.join(combo))
    return words


def run_experiment(n, lam, alphabet=('a', 'b'), max_word_len=5, seed=0, hp_dps=80):
    rng = np.random.default_rng(seed)
    qfa = build_random_qfa(n, alphabet, rng)
    gfa = qfa_to_gfa(qfa)

    # PFA en punto flotante estándar (para mostrar la fragilidad numérica)
    pfa_double = gfa_to_pfa(gfa, lam)
    # PFA construido enteramente en alta precisión (referencia matemática)
    pfa_hp = gfa_to_pfa_hp(gfa, lam, dps=hp_dps)

    words = all_words_up_to(alphabet, max_word_len)

    max_gfa_error = 0.0
    double_precision_mismatches = []
    high_precision_mismatches = []

    old_dps = mp.mp.dps
    mp.mp.dps = hp_dps
    try:
        for w in words:
            fQ = qfa.run(w)
            fG = gfa.run(w)
            max_gfa_error = max(max_gfa_error, abs(fQ - fG))

            fP_double = pfa_double.accept_probability(w)
            fP_hp = pfa_hp.accept_probability(w)

            cutpoint_qfa = fQ > lam

            if cutpoint_qfa != (fP_double > 0.5):
                double_precision_mismatches.append((w, fQ, fP_double))

            if cutpoint_qfa != (fP_hp > mp.mpf('0.5')):
                high_precision_mismatches.append((w, fQ, str(fP_hp)))
    finally:
        mp.mp.dps = old_dps

    k_gfa = gfa.k
    n_pfa_states = pfa_double.total_states
    expected_k = n * n
    expected_pfa_bound = 2 * expected_k + 6

    return {
        'n': n,
        'lam': lam,
        'num_words_tested': len(words),
        'max_gfa_reconstruction_error': max_gfa_error,
        'double_precision_mismatches': double_precision_mismatches,
        'high_precision_mismatches': high_precision_mismatches,
        'k_gfa_states': k_gfa,
        'expected_k_gfa_states': expected_k,
        'n_pfa_total_states': n_pfa_states,
        'expected_pfa_bound_2k_plus_6': expected_pfa_bound,
    }


if __name__ == '__main__':
    print(__doc__)
    print("=" * 78)
    print("VERIFICACIÓN NUMÉRICA DE LA CONSTRUCCIÓN 1gQFA -> GFA -> PFA")
    print("=" * 78)

    for n in (2, 3, 4):
        for lam in (0.3, 0.5, 0.7):
            result = run_experiment(n=n, lam=lam, max_word_len=5, seed=42 + n)
            n_dp_mismatch = len(result['double_precision_mismatches'])
            n_hp_mismatch = len(result['high_precision_mismatches'])
            status_hp = "OK (exacto)" if n_hp_mismatch == 0 else "FALLÓ"

            print(f"\n--- n={result['n']}, lambda={result['lam']} ---")
            print(f"  Palabras probadas:                       {result['num_words_tested']}")
            print(f"  Error máx. reconstrucción GFA (1gQFA):    {result['max_gfa_reconstruction_error']:.2e}")
            print(f"  Estados del GFA (Prop. 3.1):              {result['k_gfa_states']} "
                  f"(esperado n^2 = {result['expected_k_gfa_states']})")
            print(f"  Estados totales del PFA (Teo. 3.2):       {result['n_pfa_total_states']} "
                  f"(cota 2k+6 = {result['expected_pfa_bound_2k_plus_6']})")
            print(f"  Discrepancias en punto flotante (float64): {n_dp_mismatch} / {result['num_words_tested']}"
                  f"  <- artefacto numérico esperado para palabras largas")
            print(f"  Discrepancias en precisión arbitraria:     {n_hp_mismatch} / {result['num_words_tested']}"
                  f"  <- {status_hp}")
            if result['high_precision_mismatches']:
                print(f"  Discrepancias reales (alta precisión): {result['high_precision_mismatches'][:5]}")
