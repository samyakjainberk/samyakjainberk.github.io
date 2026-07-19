"""
spectral_lab.py — independent Python reference implementation of the
spectral-estimation widget's numerics (spectral-numerics.js).

Purpose: check the JavaScript implementation independently. The random
number generator (mulberry32 + Box-Muller) is ported bit-for-bit from the
JS, so the SAME seed produces the SAME matrices, probes and Lanczos runs
in both languages. Ground-truth eigendecompositions here use numpy.linalg
(a completely different eigensolver than the JS tred2/tqli), so agreement
between the two implementations checks both the algorithms and the
eigensolvers.

Run the self test:
    python3 spectral_lab.py

Every public function mirrors its JS counterpart by name and draw order:
RNG, orthonormalize, build_goe / build_wishart / build_spiked /
build_clustered / build_log_spaced, MLP (loss/grad/hvp/train/full_hessian),
power_deflation, lanczos, ritz_from_lanczos, block_lanczos,
ritz_from_block_lanczos, slq, edge_plus_slq, kpm, scree_from_density,
smooth_density, slq_density, trapz, l1_density_error.
"""

import math
import numpy as np

M32 = 0xFFFFFFFF


def _imul(a, b):
    """32-bit signed multiply, like JS Math.imul."""
    r = (a & M32) * (b & M32) & M32
    return r - 0x100000000 if r >= 0x80000000 else r


def _u32(x):
    return x & M32


class RNG:
    """Port of the JS mulberry32 + cached Box-Muller gaussian."""

    def __init__(self, seed=12345):
        self.a = _u32(int(seed))
        self._gauss_cache = None

    def uniform(self):
        self.a = _u32(self.a + 0x6D2B79F5)
        t = self.a
        t = _u32(_imul(t ^ (t >> 15), (1 | t) & M32))
        t = _u32((_u32(t + _imul(t ^ (t >> 7), (61 | t) & M32))) ^ t)
        return _u32(t ^ (t >> 14)) / 4294967296.0

    def gauss(self):
        if self._gauss_cache is not None:
            g, self._gauss_cache = self._gauss_cache, None
            return g
        u = 0.0
        while u == 0.0:
            u = self.uniform()
        v = self.uniform()
        r = math.sqrt(-2.0 * math.log(u))
        self._gauss_cache = r * math.sin(2.0 * math.pi * v)
        return r * math.cos(2.0 * math.pi * v)

    def rademacher(self):
        return -1.0 if self.uniform() < 0.5 else 1.0

    def vec(self, p, dist='gaussian'):
        if dist == 'rademacher':
            return np.array([self.rademacher() for _ in range(p)])
        return np.array([self.gauss() for _ in range(p)])

    def int(self, n):
        return int(self.uniform() * n)


# ------------------------- linear algebra helpers -------------------------

def orthonormalize(vecs, against=None, rng=None, tol=1e-10):
    """Two-pass modified Gram-Schmidt; mirrors the JS draw order for
    rank-deficient replacement. Returns (list_of_vecs, replaced_count)."""
    out, replaced = [], 0
    prior = list(against) if against else []
    for v0 in vecs:
        v = v0.copy()
        for _ in range(2):
            for u in prior:
                v -= (u @ v) * u
            for u in out:
                v -= (u @ v) * u
        nv = np.linalg.norm(v)
        if nv < tol:
            if rng is None:
                continue
            replaced += 1
            tries = 0
            while nv < tol and tries < 20:
                v = rng.vec(len(v0), 'gaussian')
                for _ in range(2):
                    for u in prior:
                        v -= (u @ v) * u
                    for u in out:
                        v -= (u @ v) * u
                nv = np.linalg.norm(v)
                tries += 1
            if nv < tol:
                continue
        out.append(v / nv)
    return out, replaced


def eigen_sym(A):
    """Ground truth via numpy (independent of the JS tred2/tqli).
    Returns eigenvalues ascending and eigenvectors as columns."""
    vals, vecs = np.linalg.eigh(A)
    return vals, vecs


# ---------------------------- matrix ensembles ----------------------------

def build_goe(p, rng, scale=1.0):
    A = np.empty((p, p))
    for i in range(p):
        for j in range(p):
            A[i, j] = rng.gauss()
    s = scale / math.sqrt(2.0 * p)
    A = (A + A.T) * s
    return A


def build_bernoulli(p, rng, q=0.5):
    """Symmetric matrix, every element iid Bernoulli(q) in {0,1}
    (upper triangle sampled, mirrored). Mirrors the JS draw order."""
    q = 0.5 if not np.isfinite(q) else min(1.0, max(0.0, q))
    A = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            v = 1.0 if rng.uniform() < q else 0.0
            A[i, j] = v
            A[j, i] = v
    return A


def build_wishart(p, rng, aspect):
    nw = max(2, round(aspect * p))
    X = np.empty(p * nw)
    for i in range(p * nw):
        X[i] = rng.gauss()
    X = X.reshape(p, nw)
    return X @ X.T / nw


def build_spiked(p, rng, num_spikes=3, spike_max=8.0, spike_min=4.0, neg_frac=0.0):
    A = build_goe(p, rng, 1.0)
    for s_idx in range(num_spikes):
        u = rng.vec(p, 'gaussian')
        u /= np.linalg.norm(u)
        lam = spike_max if num_spikes == 1 else spike_min + (spike_max - spike_min) * s_idx / (num_spikes - 1)
        if rng.uniform() < neg_frac:
            lam = -lam
        A += lam * np.outer(u, u)
    return A


def rotate_diag(lam, p, rng):
    cols = [rng.vec(p, 'gaussian') for _ in range(p)]
    Q, _ = orthonormalize(cols, None, rng)
    A = np.zeros((p, p))
    for k in range(p):
        A += lam[k] * np.outer(Q[k], Q[k])
    return A


def build_clustered(p, rng, num_clusters=4, spread=1e-4, lo=-1.0, hi=5.0):
    centers = [hi if num_clusters == 1 else lo + (hi - lo) * c / (num_clusters - 1)
               for c in range(num_clusters)]
    lam = np.array([centers[i % num_clusters] + spread * rng.gauss() for i in range(p)])
    return rotate_diag(lam, p, rng)


def build_log_spaced(p, rng, decades=3, hi=10.0):
    lam = np.array([hi * 10.0 ** (-decades * i / (p - 1 if p > 1 else 1)) for i in range(p)])
    return rotate_diag(lam, p, rng)


# ------------------------- MLP with exact HVP (R-op) -------------------------

class MLP:
    """One-hidden-layer MLP, MSE loss L = 1/(2n) sum ||y - t||^2.
    Parameter layout: W1 (h x dIn), b1 (h), W2 (dOut x h), b2 (dOut)."""

    def __init__(self, d_in, hidden, d_out, n, seed=7, activation='tanh',
                 target_mode='teacher', noise=0.1):
        self.d_in, self.h, self.d_out, self.n = d_in, hidden, d_out, n
        self.act = activation
        self.P = hidden * d_in + hidden + d_out * hidden + d_out
        rng = RNG(seed)
        self.rng = rng
        self.X = np.array([rng.gauss() for _ in range(n * d_in)]).reshape(n, d_in)
        self.T = np.empty((n, d_out))
        if target_mode == 'teacher':
            tw = self._init_w(RNG(seed + 1000))
            for s in range(n):
                y = self._forward_one(tw, s)['y']
                for o in range(d_out):
                    self.T[s, o] = y[o] + noise * rng.gauss()
        else:
            for i in range(n * d_out):
                self.T.flat[i] = rng.gauss()
        self.w = self._init_w(rng)

    def _init_w(self, rng, gain=1.0):
        w = np.zeros(self.P)
        s1, s2 = gain / math.sqrt(self.d_in), gain / math.sqrt(self.h)
        o = 0
        for i in range(self.h * self.d_in):
            w[o] = s1 * rng.gauss(); o += 1
        o += self.h                       # b1 zeros
        for i in range(self.d_out * self.h):
            w[o] = s2 * rng.gauss(); o += 1
        return w

    def _unpack(self, w):
        dI, h, dO = self.d_in, self.h, self.d_out
        o = 0
        W1 = w[o:o + h * dI].reshape(h, dI); o += h * dI
        b1 = w[o:o + h]; o += h
        W2 = w[o:o + dO * h].reshape(dO, h); o += dO * h
        b2 = w[o:o + dO]
        return W1, b1, W2, b2

    def _phi(self, a):
        if self.act == 'relu':
            return (np.maximum(a, 0.0), (a > 0).astype(float), np.zeros_like(a))
        t = np.tanh(a)
        d1 = 1.0 - t * t
        return (t, d1, -2.0 * t * d1)

    def _forward_one(self, w, s):
        W1, b1, W2, b2 = self._unpack(w)
        x = self.X[s]
        a1 = W1 @ x + b1
        h1, d1, dd1 = self._phi(a1)
        y = W2 @ h1 + b2
        return dict(x=x, a1=a1, h1=h1, d1=d1, dd1=dd1, y=y)

    def loss(self, w=None):
        w = self.w if w is None else w
        L = 0.0
        for s in range(self.n):
            r = self._forward_one(w, s)['y'] - self.T[s]
            L += 0.5 * float(r @ r)
        return L / self.n

    def grad(self, w=None):
        w = self.w if w is None else w
        W1, b1, W2, b2 = self._unpack(w)
        g = np.zeros(self.P)
        gW1, gb1, gW2, gb2 = self._unpack(g)
        for s in range(self.n):
            f = self._forward_one(w, s)
            del2 = (f['y'] - self.T[s]) / self.n
            gb2 += del2
            gW2 += np.outer(del2, f['h1'])
            del1 = (W2.T @ del2) * f['d1']
            gb1 += del1
            gW1 += np.outer(del1, f['x'])
        return g

    def hvp(self, v, w=None):
        """Exact Hessian-vector product, forward-over-reverse (R-op)."""
        w = self.w if w is None else w
        W1, b1, W2, b2 = self._unpack(w)
        V1, vb1, V2, vb2 = self._unpack(np.asarray(v, dtype=float))
        Hv = np.zeros(self.P)
        HW1, Hb1, HW2, Hb2 = self._unpack(Hv)
        for s in range(self.n):
            f = self._forward_one(w, s)
            da1 = V1 @ f['x'] + vb1
            dh1 = f['d1'] * da1
            dy = V2 @ f['h1'] + vb2 + W2 @ dh1
            del2 = (f['y'] - self.T[s]) / self.n
            del2dot = dy / self.n
            Hb2 += del2dot
            HW2 += np.outer(del2dot, f['h1']) + np.outer(del2, dh1)
            w2d = W2.T @ del2
            tang = V2.T @ del2 + W2.T @ del2dot
            del1dot = tang * f['d1'] + w2d * f['dd1'] * da1
            Hb1 += del1dot
            HW1 += np.outer(del1dot, f['x'])
        return Hv

    def train(self, steps, lr):
        for _ in range(steps):
            self.w -= lr * self.grad()

    def full_hessian(self):
        H = np.empty((self.P, self.P))
        e = np.zeros(self.P)
        for j in range(self.P):
            e[:] = 0.0; e[j] = 1.0
            H[:, j] = self.hvp(e)
        return 0.5 * (H + H.T)


# ------------------------------ operators ------------------------------

class MatOp:
    def __init__(self, A):
        self.A, self.p, self.count = A, A.shape[0], 0

    def mv(self, v):
        self.count += 1
        return self.A @ v


class HvpOp:
    def __init__(self, mlp):
        self.mlp, self.p, self.count = mlp, mlp.P, 0

    def mv(self, v):
        self.count += 1
        return self.mlp.hvp(v)


class DeflatedOp:
    """(I - UU^T) A (I - UU^T); projections are not counted as matvecs."""

    def __init__(self, op, U):
        self.op, self.U, self.p = op, U, op.p

    def project(self, v):
        w = v.copy()
        for u in self.U:
            w -= (u @ w) * u
        return w

    def mv(self, v):
        return self.project(self.op.mv(self.project(v)))

    @property
    def count(self):
        return self.op.count


# ------------------------- power iteration + deflation -------------------------

def power_deflation(op, num_eigs=3, iters=100, shift=0.0, seed=1):
    p = op.p
    num_eigs = min(num_eigs, p)
    rng = RNG(seed)
    found, U = [], []
    for _ in range(num_eigs):
        x = rng.vec(p, 'gaussian')
        for u in U:
            x -= (u @ x) * u
        x /= np.linalg.norm(x)
        history = []
        for _ in range(iters):
            y = op.mv(x)
            if shift != 0.0:
                y = y + shift * x
            for u in U:
                y -= (u @ y) * u
            history.append(float(x @ y) - shift)   # exact signed Rayleigh quotient
            ny = np.linalg.norm(y)
            if ny < 1e-300:
                break
            x = y / ny
        rq = float(x @ op.mv(x))
        found.append(dict(value=rq, vector=x, history=history))
        U.append(x)
    return dict(eigs=found, matvecs=op.count)


# ------------------------------- Lanczos -------------------------------

def lanczos(op, k, seed=2, reorth=True, start_vec=None, against=None, dist='gaussian'):
    p = op.p
    k = min(k, p)
    against = list(against) if against else []
    rng = RNG(seed)
    v = start_vec.copy() if start_vec is not None else rng.vec(p, dist)
    for u in against:
        v -= (u @ v) * u
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return dict(alpha=[], beta=[], V=[], breakdowns=0, exhausted=True, beta_final=0.0)
    v = v / nv
    V, alpha, beta = [v], [], []
    breakdowns, exhausted, beta_final = 0, False, 0.0
    v_prev, beta_prev = None, 0.0
    for j in range(k):
        vj = V[j]
        w = op.mv(vj)
        a = float(vj @ w)
        alpha.append(a)
        w -= a * vj
        if v_prev is not None:
            w -= beta_prev * v_prev
        if reorth:
            for _ in range(2):
                for u in against:
                    w -= (u @ w) * u
                for u in V:
                    w -= (u @ w) * u
        else:
            for u in against:
                w -= (u @ w) * u
        b = float(np.linalg.norm(w))
        if j == k - 1:
            beta_final = b
            break
        if b < 1e-10 * (abs(a) + 1.0):
            breakdowns += 1
            res, _ = orthonormalize([rng.vec(p, 'gaussian')], against + V, rng)
            if not res:
                exhausted = True
                break
            V.append(res[0])
            beta.append(0.0)
            v_prev, beta_prev = vj, 0.0
        else:
            w = w / b
            V.append(w)
            beta.append(b)
            v_prev, beta_prev = vj, b
    return dict(alpha=alpha, beta=beta, V=V, breakdowns=breakdowns,
                exhausted=exhausted, beta_final=beta_final)


def ritz_from_lanczos(lz, want_vectors=False):
    k = len(lz['alpha'])
    if k == 0:
        return dict(values=[], weights=[], residuals=[], vectors=[])
    T = np.diag(lz['alpha'])
    for i, b in enumerate(lz['beta']):
        T[i, i + 1] = T[i + 1, i] = b
    vals, S = np.linalg.eigh(T)
    weights = S[0, :] ** 2
    residuals = abs(lz['beta_final'] * S[k - 1, :])
    vectors = None
    if want_vectors:
        Vm = np.stack(lz['V'], axis=1)      # p x k
        vectors = [Vm @ S[:, j] for j in range(k)]
    return dict(values=list(vals), weights=list(weights),
                residuals=list(residuals), vectors=vectors)


# ----------------------------- block Lanczos -----------------------------

def block_lanczos(op, k, b, seed=3, reorth=True, start_block=None, against=None, dist='gaussian'):
    p = op.p
    b = max(1, b)
    k = max(1, min(k, p // b))
    against = list(against) if against else []
    rng = RNG(seed)
    dim = k * b
    T = np.zeros((dim, dim))
    start = [q.copy() for q in start_block] if start_block is not None \
        else [rng.vec(p, dist) for _ in range(b)]
    q0, replaced = orthonormalize(start, against, rng)
    if len(q0) < b:
        return dict(T=None, exhausted=True)
    blocks = [q0]
    B_prev = None
    exhausted = False
    for j in range(k):
        Qj = blocks[j]
        W = [op.mv(q) for q in Qj]
        Aj = np.array([[float(Qj[r] @ W[c]) for c in range(b)] for r in range(b)])
        Aj = 0.5 * (Aj + Aj.T)
        T[j * b:(j + 1) * b, j * b:(j + 1) * b] = Aj
        if j == k - 1:
            break
        for c in range(b):
            for r in range(b):
                W[c] -= Aj[r, c] * Qj[r]
            if B_prev is not None:
                Qm = blocks[j - 1]
                for r in range(b):
                    W[c] -= B_prev[c, r] * Qm[r]
        prior_all = against + [v for blk in blocks for v in blk] if reorth \
            else against + blocks[j] + (blocks[j - 1] if j > 0 else [])
        for _ in range(2):
            for w in W:
                for u in prior_all:
                    w -= (u @ w) * u
        Bj = np.zeros((b, b))
        Qn = []
        for c in range(b):
            wv = W[c]
            for r in range(len(Qn)):
                proj = float(Qn[r] @ wv)
                Bj[c, r] = proj
                wv -= proj * Qn[r]
            nw = float(np.linalg.norm(wv))
            if nw < 1e-10:
                res, _ = orthonormalize([rng.vec(p, 'gaussian')],
                                        against + [v for blk in blocks for v in blk] + Qn, rng)
                if not res:
                    exhausted = True
                    break
                Qn.append(res[0])
                Bj[c, c] = 0.0
            else:
                Bj[c, c] = nw
                Qn.append(wv / nw)
        if exhausted or len(Qn) < b:
            exhausted = True
            break
        for r in range(b):
            for c in range(b):
                T[(j + 1) * b + r, j * b + c] = Bj[c, r]
                T[j * b + c, (j + 1) * b + r] = Bj[c, r]
        blocks.append(Qn)
        B_prev = Bj
    used_dim = len(blocks) * b
    return dict(T=T[:used_dim, :used_dim], dim=used_dim, b=b, blocks=blocks, exhausted=exhausted)


def ritz_from_block_lanczos(bl, want_vectors=False):
    if bl['T'] is None:
        return dict(values=[], weights=[], vectors=[])
    dim, b = bl['dim'], bl['b']
    vals, S = np.linalg.eigh(bl['T'])
    weights = np.sum(S[0:b, :] ** 2, axis=0)
    vectors = None
    if want_vectors:
        allv = np.stack([v for blk in bl['blocks'] for v in blk], axis=1)
        vectors = [allv @ S[:, j] for j in range(dim)]
    return dict(values=list(vals), weights=list(weights), vectors=vectors)


# -------------------------------- SLQ family --------------------------------

def slq(op, probes=8, k=30, dist='rademacher', seed=4, reorth=True,
        orth_probes=False, block=1, deflate_u=None):
    p = op.p
    deflate_u = list(deflate_u) if deflate_u else []
    rng = RNG(seed)
    base_op = DeflatedOp(op, deflate_u) if deflate_u else op
    per_probe, krylov_bank = [], []
    exhausted = False
    for _ in range(probes):
        against = deflate_u + (krylov_bank if orth_probes else [])
        if block == 1:
            z = rng.vec(p, dist)
            res, _ = orthonormalize([z], against, None)
            if not res:
                exhausted = True
                break
            lz = lanczos(base_op, k, seed=rng.int(10 ** 9), reorth=reorth,
                         start_vec=res[0], against=against)
            if not lz['alpha']:
                exhausted = True
                break
            rz = ritz_from_lanczos(lz)
            per_probe.append(dict(nodes=rz['values'], weights=rz['weights']))
            if orth_probes:
                krylov_bank.extend(lz['V'])
        else:
            start = [rng.vec(p, dist) for _ in range(block)]
            sres, _ = orthonormalize(start, against, None)
            if len(sres) < block:
                exhausted = True
                break
            bl = block_lanczos(base_op, k, block, seed=rng.int(10 ** 9),
                               reorth=reorth, start_block=sres, against=against)
            if bl['T'] is None:
                exhausted = True
                break
            rz = ritz_from_block_lanczos(bl)
            per_probe.append(dict(nodes=rz['values'],
                                  weights=[w / block for w in rz['weights']]))
            if orth_probes:
                for blk in bl['blocks']:
                    krylov_bank.extend(blk)
    return dict(per_probe=per_probe, matvecs=op.count, exhausted=exhausted,
                probes_run=len(per_probe))


def edge_plus_slq(op, cycles=2, k_edge=30, m_per_cycle=3, edge_which='both',
                  probes=8, k_slq=30, dist='rademacher', seed=5, orth_probes=False, block=1):
    p = op.p
    rng = RNG(seed)
    U, deflated = [], []
    for _ in range(cycles):
        dop = DeflatedOp(op, U) if U else op
        lz = lanczos(dop, k_edge, seed=rng.int(10 ** 9), reorth=True, against=U)
        if not lz['alpha']:
            break
        rz = ritz_from_lanczos(lz, want_vectors=True)
        idx = sorted(range(len(rz['values'])), key=lambda i: -rz['values'][i])
        if edge_which == 'top':
            picked = idx[:m_per_cycle]
        else:
            n_top = (m_per_cycle + 1) // 2
            n_bot = m_per_cycle // 2
            picked = idx[:n_top] + (idx[len(idx) - n_bot:] if n_bot else [])
        for j in picked:
            res, _ = orthonormalize([rz['vectors'][j]], U, None)
            if not res:
                continue
            U.append(res[0])
            deflated.append(dict(value=rz['values'][j], residual=rz['residuals'][j]))
            if len(U) >= p - 2:
                break
        if len(U) >= p - 2:
            break
    s = slq(op, probes=probes, k=k_slq, dist=dist, seed=rng.int(10 ** 9),
            deflate_u=U, orth_probes=orth_probes, block=block)
    return dict(deflated=deflated, U=U, slq=s, matvecs=op.count)


# --------------------------------- KPM ---------------------------------

def kpm(op, degree=80, probes=8, dist='rademacher', seed=6, jackson=True,
        lmin=None, lmax=None):
    p = op.p
    K = degree
    rng = RNG(seed)
    if lmin is None or lmax is None:
        lz = lanczos(op, min(40, p), seed=rng.int(10 ** 9), reorth=True)
        rz = ritz_from_lanczos(lz)
        vmin, vmax = min(rz['values']), max(rz['values'])
        pad = 0.05 * (vmax - vmin) + 1e-10
        lmin = vmin - pad if lmin is None else lmin
        lmax = vmax + pad if lmax is None else lmax
    c_scale, c_shift = (lmax - lmin) / 2.0, (lmax + lmin) / 2.0
    if not c_scale > 1e-8 * max(1.0, abs(c_shift)):
        raise ValueError('KPM: degenerate spectral range')

    def mv_t(v):
        return (op.mv(v) - c_shift * v) / c_scale

    mu_sum = np.zeros(K + 1)
    for _ in range(probes):
        z = rng.vec(p, dist)
        nz2 = float(z @ z)
        w0 = z.copy()
        w1 = mv_t(z)
        mu_sum[0] += float(z @ w0) / nz2
        if K >= 1:
            mu_sum[1] += float(z @ w1) / nz2
        for kk in range(2, K + 1):
            w2 = 2.0 * mv_t(w1) - w0
            mu_sum[kk] += float(z @ w2) / nz2
            w0, w1 = w1, w2
    mu_raw = mu_sum / probes
    if np.max(np.abs(mu_raw)) > 2 * max(1e-12, abs(mu_raw[0])):
        raise ValueError('KPM: moments diverged; [lmin, lmax] does not bracket the spectrum')
    mu = mu_raw.copy()
    if jackson:
        q = math.pi / (K + 2)
        for kk in range(K + 1):
            mu[kk] *= ((K + 2 - kk) * math.sin(q) * math.cos(kk * q)
                       + math.cos(q) * math.sin(kk * q)) / ((K + 2) * math.sin(q))

    def _eval_mu(mu_arr, t_grid):
        out = np.zeros(len(t_grid))
        for i, t in enumerate(t_grid):
            x = (t - c_shift) / c_scale
            if x <= -1.0 or x >= 1.0:
                continue
            acc = mu_arr[0]
            tm1, t0 = 1.0, x
            if K >= 1:
                acc += 2.0 * mu_arr[1] * t0
            for kk in range(2, K + 1):
                t1 = 2.0 * x * t0 - tm1
                acc += 2.0 * mu_arr[kk] * t1
                tm1, t0 = t0, t1
            out[i] = acc / (math.pi * math.sqrt(1.0 - x * x)) / c_scale
        return out

    def eval_density(t_grid):
        return _eval_mu(mu, t_grid)

    def eval_density_trunc(Kp, t_grid):
        """Density from the first Kp+1 moments, Jackson damping recomputed
        for that degree (mirrors the JS evalDensityTrunc)."""
        Kp = max(1, min(K, int(Kp)))
        muT = np.zeros(K + 1)
        q = math.pi / (Kp + 2)
        for kk in range(Kp + 1):
            g = (((Kp + 2 - kk) * math.sin(q) * math.cos(kk * q)
                  + math.cos(q) * math.sin(kk * q)) / ((Kp + 2) * math.sin(q))
                 if jackson else 1.0)
            muT[kk] = g * mu_raw[kk]
        return _eval_mu(muT, t_grid)

    return dict(mu=mu, mu_raw=mu_raw, lmin=lmin, lmax=lmax,
                eval_density=eval_density, eval_density_trunc=eval_density_trunc,
                matvecs=op.count)


# --------------------------- density utilities ---------------------------

def make_grid(lo, hi, npts=400):
    return np.linspace(lo, hi, npts)


def smooth_density(xs, ws, grid, sigma):
    """Gaussian-smoothed sticks; sigma is clamped to the grid step, like the JS."""
    step = grid[1] - grid[0] if len(grid) > 1 else 0.0
    s = max(sigma, step)
    out = np.zeros(len(grid))
    c = 1.0 / (s * math.sqrt(2.0 * math.pi))
    n = len(xs)
    for i, x in enumerate(xs):
        w = ws[i] if ws is not None else 1.0 / n
        d = (grid - x) / s
        mask = np.abs(d) < 8.0
        out[mask] += w * c * np.exp(-0.5 * d[mask] ** 2)
    return out


def slq_density(res, grid, sigma, n_probes=None):
    use = res['per_probe'][:n_probes if n_probes is not None else len(res['per_probe'])]
    out = np.zeros(len(grid))
    if not use:
        return out
    for pr in use:
        out += smooth_density(pr['nodes'], pr['weights'], grid, sigma) / len(use)
    return out


def trapz(y, grid):
    return float(np.trapz(y, grid))


def l1_density_error(a, b, grid):
    return float(np.trapz(np.abs(np.asarray(a) - np.asarray(b)), grid))


def scree_from_density(grid, dens, p, floor=None):
    """Quantile inversion: descending eigenvalue-vs-index curve."""
    n = len(grid)
    d = np.maximum(np.asarray(dens), 0.0)
    above = np.zeros(n)
    for g in range(n - 2, -1, -1):
        above[g] = above[g + 1] + 0.5 * (d[g] + d[g + 1]) * (grid[g + 1] - grid[g])
    total = above[0] if above[0] else 1.0
    out = np.empty(p)
    g = n - 1
    for i in range(1, p + 1):
        target = (i - 0.5) / p * total
        while g > 0 and above[g - 1] < target:
            g -= 1
        if g == 0:
            out[i - 1] = grid[0]
            continue
        a0, a1 = above[g], above[g - 1]
        w = (target - a0) / (a1 - a0) if a1 > a0 else 0.0
        out[i - 1] = grid[g] + w * (grid[g - 1] - grid[g])
    if floor is not None and np.isfinite(floor):
        out = np.maximum(out, floor)
    return out


# --------------------------------- self test ---------------------------------

def _selftest():
    ok_all = True

    def ok(name, cond, detail=''):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(('PASS ' if cond else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))

    # RNG sanity
    r = RNG(1)
    u = [r.uniform() for _ in range(3)]
    ok('RNG deterministic', all(0 <= x < 1 for x in u), ','.join(f'{x:.6f}' for x in u))

    # GOE + eigen + Lanczos k=p reproduces spectrum
    p = 40
    A = build_goe(p, RNG(5), 1.0)
    truth, _ = eigen_sym(A)
    op = MatOp(A)
    lz = lanczos(op, p, seed=11, reorth=True)
    rz = ritz_from_lanczos(lz)
    ok('lanczos k=p == eigh', np.max(np.abs(np.array(rz['values']) - truth)) < 1e-8)
    ok('weights sum to 1', abs(sum(rz['weights']) - 1) < 1e-10)

    # MLP: hvp vs finite differences and vs full Hessian
    mlp = MLP(5, 7, 3, 20, seed=3)
    rng = RNG(9)
    for _ in range(1):
        v = rng.vec(mlp.P, 'gaussian')
    hv = mlp.hvp(v)
    eps = 1e-6
    gp = mlp.grad(mlp.w + eps * v)
    gm = mlp.grad(mlp.w - eps * v)
    fd = (gp - gm) / (2 * eps)
    ok('MLP hvp vs FD', np.linalg.norm(fd - hv) / np.linalg.norm(fd) < 1e-4)
    H = mlp.full_hessian()
    ok('H v == hvp(v)', np.linalg.norm(H @ v - hv) / np.linalg.norm(hv) < 1e-10)

    # block Lanczos full run on clustered matrix
    p2 = 36
    A2 = build_clustered(p2, RNG(7), num_clusters=4, spread=0.0)
    t2, _ = eigen_sym(A2)
    bl = block_lanczos(MatOp(A2), 12, 3, seed=13, reorth=True)
    rz2 = ritz_from_block_lanczos(bl)
    ok('block lanczos full run == eigh', np.max(np.abs(np.array(rz2['values']) - t2)) < 1e-7)

    # SLQ density integrates to 1 and tracks the truth
    p3 = 80
    A3 = build_goe(p3, RNG(8), 1.0)
    t3, _ = eigen_sym(A3)
    res = slq(MatOp(A3), probes=12, k=30, seed=14)
    grid = make_grid(-3, 3, 400)
    est = slq_density(res, grid, 0.1)
    rho = smooth_density(t3, None, grid, 0.1)
    ok('SLQ integrates to 1', abs(trapz(est, grid) - 1) < 0.02, f'{trapz(est, grid):.4f}')
    ok('SLQ L1 small', l1_density_error(est, rho, grid) < 0.15,
       f'{l1_density_error(est, rho, grid):.4f}')

    # KPM
    kp = kpm(MatOp(A3), degree=120, probes=16, seed=15)
    est2 = kp['eval_density'](grid)
    ok('KPM mu0 == 1', abs(kp['mu_raw'][0] - 1) < 1e-12)
    ok('KPM integrates to 1', abs(trapz(est2, grid) - 1) < 0.03, f'{trapz(est2, grid):.4f}')

    # edge + SLQ on a spiked matrix
    A4 = build_spiked(80, RNG(10), num_spikes=4, spike_min=5, spike_max=10)
    t4, _ = eigen_sym(A4)
    ed = edge_plus_slq(MatOp(A4), cycles=2, k_edge=30, m_per_cycle=3,
                       probes=8, k_slq=25, seed=16)
    top = sorted([d['value'] for d in ed['deflated']], reverse=True)[:4]
    ok('edge+SLQ deflates top-4 exactly',
       np.max(np.abs(np.array(top) - t4[-4:][::-1])) < 1e-6)

    print('\nALL PASS' if ok_all else '\nSOME CHECKS FAILED')
    return ok_all


if __name__ == '__main__':
    import sys
    sys.exit(0 if _selftest() else 1)
