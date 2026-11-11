"""
Implementacion de la construccion de n^2+1 estados de:

    Zeyu Chen y Junde Wu, "The State Cost of Classical Simulation of One-Way
    General Quantum Finite Automata", arXiv:2604.07058 (version actual).

Pasos (numeracion de esa version):
  * Teorema 3.2, paso 1: linealizacion 1gQFA -> GFA de k = n^2 estados con
    cutpoint 0, con vector final v_i = Tr((P_acc - lambda I) B_i).
  * Proposicion 3.1: GFA de k estados (cutpoint 0) -> PFA de m = k+1 estados
    con cutpoint 1/m, mediante
        S_sigma = Q + eps * K_sigma,     p_0 = q + delta * T(u),
    de modo que, para toda palabra w,
        f_P(w) - 1/m = delta * eps^{|w|} * f_G(w).

La construccion deja libre el isomorfismo T: R^{1xk} -> W (con T(r) e_1 = r v).
Aqui se ofrecen dos elecciones ('orthogonal' y 'naive') y tres reglas para eps
('paper': eps*max|K| <= 1/m, 'tight': solo entradas negativas, y por simbolo).

Una unica implementacion generica (listas de Python) sirve para tres
backends numericos: float64 ('float'), precision arbitraria ('mp') e
intervalos con redondeo dirigido ('iv').
"""

import math
import mpmath as mp

from qfa_simulation import hermitian_basis

# ---------------------------------------------------------------------
# Backends numericos
# ---------------------------------------------------------------------


class _FloatOps:
    name = 'float'

    def from_float(self, x):
        return float(x)

    def from_str(self, s):
        return float(s)

    def sqrt(self, x):
        return math.sqrt(x)

    def to_float(self, x):
        return float(x)

    def is_zero(self, x):
        return x == 0


class _MPOps:
    name = 'mp'

    def from_float(self, x):
        return mp.mpf(float(x))

    def from_str(self, s):
        return mp.mpf(s)

    def sqrt(self, x):
        return mp.sqrt(x)

    def to_float(self, x):
        return float(x)

    def is_zero(self, x):
        return x == 0


class _IVOps:
    name = 'iv'

    def from_float(self, x):
        return mp.iv.mpf(float(x))

    def from_str(self, s):
        return mp.iv.mpf(s)

    def sqrt(self, x):
        return mp.iv.sqrt(x)

    def to_float(self, x):
        return float(x.mid)

    def is_zero(self, x):
        return x == 0


OPS = {'float': _FloatOps(), 'mp': _MPOps(), 'iv': _IVOps()}


# ---------------------------------------------------------------------
# Algebra lineal generica sobre listas
# ---------------------------------------------------------------------

def _zeros(r, c, ops):
    z = ops.from_float(0.0)
    return [[z for _ in range(c)] for _ in range(r)]


def _mm(A, B, ops):
    """Producto de matrices (listas de listas); omite ceros exactos."""
    r, inner, c = len(A), len(B), len(B[0])
    out = _zeros(r, c, ops)
    for i in range(r):
        Ai = A[i]
        oi = out[i]
        for t in range(inner):
            a = Ai[t]
            if ops.is_zero(a):
                continue
            Bt = B[t]
            for j in range(c):
                oi[j] = oi[j] + a * Bt[j]
    return out


def _vm(u, A, ops):
    return _mm([u], A, ops)[0]


def _transpose(A):
    return [list(col) for col in zip(*A)]


# ---------------------------------------------------------------------
# Piezas de la construccion
# ---------------------------------------------------------------------

def helmert_rows(k, ops):
    """Base ortonormal Z (k x m, m=k+1) del subespacio W = {z : z.1 = 0}."""
    m = k + 1
    Z = _zeros(k, m, ops)
    one = ops.from_float(1.0)
    for r in range(k):
        j = r + 1
        d = ops.sqrt(ops.from_float(j * (j + 1)))
        for c in range(j):
            Z[r][c] = one / d
        Z[r][j] = ops.from_float(-j) / d
    return Z


def _householder(a, b, ops):
    """Reflexion H (simetrica, ortogonal) con H a = b, para a, b unitarios."""
    k = len(a)
    w = [a[i] - b[i] for i in range(k)]
    ww = ops.from_float(0.0)
    for x in w:
        ww = ww + x * x
    two = ops.from_float(2.0)
    one = ops.from_float(1.0)
    zero = ops.from_float(0.0)
    H = _zeros(k, k, ops)
    for i in range(k):
        for j in range(k):
            H[i][j] = (one if i == j else zero) - two * w[i] * w[j] / ww
    return H


def _norm(x, ops):
    s = ops.from_float(0.0)
    for t in x:
        s = s + t * t
    return ops.sqrt(s)


def build_T(v, ops, mode='orthogonal'):
    """
    Devuelve (Z, G, Ginv) con T(r) = r G Z, T(r) e_1 = r v.
    'orthogonal': G = s R con R reflexion ortogonal y s = |v|/|g|.
    'naive'     : G = I + (v-g) g^T/(g.g)  (cambio minimo en la direccion g).
    """
    k = len(v)
    Z = helmert_rows(k, ops)
    g = [Z[r][0] for r in range(k)]
    one = ops.from_float(1.0)
    zero = ops.from_float(0.0)
    if mode == 'orthogonal':
        ng, nv = _norm(g, ops), _norm(v, ops)
        a = [x / ng for x in g]
        b = [x / nv for x in v]
        R = _householder(a, b, ops)
        s = nv / ng
        G = [[s * R[i][j] for j in range(k)] for i in range(k)]
        Ginv = [[R[j][i] / s for j in range(k)] for i in range(k)]  # R simetrica
        return Z, G, Ginv
    elif mode == 'naive':
        gg = zero
        gv = zero
        for i in range(k):
            gg = gg + g[i] * g[i]
            gv = gv + g[i] * v[i]
        # G = I + (v-g) g^T / gg  (G g = v);  G^{-1} = I - (v-g) g^T / gv
        G = [[(one if i == j else zero) + (v[i] - g[i]) * g[j] / gg
              for j in range(k)] for i in range(k)]
        Ginv = [[(one if i == j else zero) - (v[i] - g[i]) * g[j] / gv
                 for j in range(k)] for i in range(k)]
        return Z, G, Ginv
    raise ValueError(mode)


def shift_final_vector(v, lam, n, ops):
    """v_i = Tr((P_acc - lam I) B_i) = Tr(P_acc B_i) - lam Tr(B_i), con
    Tr(B_i)=1 para los n elementos diagonales y 0 en el resto."""
    out = list(v)
    for i in range(n):
        out[i] = out[i] - lam
    return out


def gfa_to_pfa_n2plus1(u, A_dict, v, ops, T_mode='orthogonal',
                       eps_rule='tight', per_symbol=True, frac=0.99):
    """
    Proposicion 3.1 de Chen y Wu (version actual) para un GFA (u, {A_s}, v)
    de k estados con cutpoint 0. Devuelve un dict con el PFA de m=k+1 estados.

    eps_rule: 'paper' -> eps*max|K| <= 1/m ;  'tight' -> eps*max(-K) <= 1/m.
    per_symbol: eps_sigma distinto por simbolo (la identidad
        f_P - 1/m = delta * prod eps_{sigma_i} * f_G  se conserva).
    frac: factor de holgura (<1) aplicado a las cotas de eps y delta.
    """
    k = len(u)
    m = k + 1
    Z, G, Ginv = build_T(v, ops, T_mode)

    Zt = _transpose(Z)
    K = {}
    kappa = {}
    for s, A in A_dict.items():
        X = _mm(_mm(Ginv, A, ops), G, ops)       # G^{-1} A G  (k x k)
        Ks = _mm(_mm(Zt, X, ops), Z, ops)        # Z^T X Z     (m x m)
        K[s] = Ks
        vals = [ops.to_float(x) for row in Ks for x in row]
        kappa[s] = max(abs(x) for x in vals) if eps_rule == 'paper' \
            else max(-x for x in vals)

    if per_symbol:
        eps_f = {s: frac / (m * kappa[s]) for s in K}
    else:
        e = frac / (m * max(kappa.values()))
        eps_f = {s: e for s in K}
    eps = {s: ops.from_float(eps_f[s]) for s in K}

    inv_m = ops.from_float(1.0) / ops.from_float(m)
    S = {}
    for s in K:
        S[s] = [[inv_m + eps[s] * K[s][i][j] for j in range(m)] for i in range(m)]

    # p0 = q + delta T(u),  T(u) = u G Z
    Tu = _vm(_vm(u, G, ops), Z, ops)
    mx = max(abs(ops.to_float(x)) for x in Tu)
    delta_f = frac / (m * mx) if mx > 0 else 1.0
    delta = ops.from_float(delta_f)
    p0 = [inv_m + delta * x for x in Tu]
    return {'m': m, 'k': k, 'S': S, 'p0': p0, 'eps': eps, 'eps_f': eps_f,
            'delta': delta, 'delta_f': delta_f, 'kappa': kappa,
            'inv_m': inv_m, 'ops': ops}


def accept_probability(pfa, word):
    """f_P(w) = (p_0 S_w)_1  (estado aceptor 1, marcador final = identidad)."""
    ops = pfa['ops']
    vec = pfa['p0']
    for sigma in word:
        vec = _vm(vec, pfa['S'][sigma], ops)
    return vec[0]


def signal(pfa, word):
    """f_P(w) - 1/m  (su signo decide la pertenencia, Prop. 3.1)."""
    return accept_probability(pfa, word) - pfa['inv_m']


def decay_factor(pfa, word):
    """delta * prod eps_{sigma_i}: factor exacto de la identidad de la Prop. 3.1."""
    f = pfa['delta']
    for s in word:
        f = f * pfa['eps'][s]
    return f


# ---------------------------------------------------------------------
# Conveniencia: construir desde un GFA de numpy (qfa_simulation.GFA)
# ---------------------------------------------------------------------

def pfa_from_gfa(gfa, lam, backend='float', dps=80, **kw):
    """Construye el PFA n^2+1 desde el GFA de qfa_simulation (float64).
    Para 'mp' fija mp.mp.dps = dps y deja ese valor activo."""
    ops = OPS[backend]
    n = int(round(math.sqrt(gfa.k)))
    if backend == 'mp':
        mp.mp.dps = dps
    lam_t = ops.from_str(str(lam))
    u = [ops.from_float(x) for x in gfa.u]
    A = {s: [[ops.from_float(x) for x in row] for row in gfa.A_dict[s]]
         for s in gfa.alphabet()}
    v = [ops.from_float(x) for x in gfa.v]
    v = shift_final_vector(v, lam_t, n, ops)
    return gfa_to_pfa_n2plus1(u, A, v, ops, **kw)


def fG_shifted(gfa, lam, word, backend='mp', dps=80):
    """f_G(w) - lam = f_Q(w) - lam, evaluado en el backend dado."""
    ops = OPS[backend]
    if backend == 'mp':
        mp.mp.dps = dps
    n = int(round(math.sqrt(gfa.k)))
    lam_t = ops.from_str(str(lam))
    vec = [ops.from_float(x) for x in gfa.u]
    for s in word:
        A = [[ops.from_float(x) for x in row] for row in gfa.A_dict[s]]
        vec = _vm(vec, A, ops)
    v = shift_final_vector([ops.from_float(x) for x in gfa.v], lam_t, n, ops)
    tot = ops.from_float(0.0)
    for a, b in zip(vec, v):
        tot = tot + a * b
    return tot
