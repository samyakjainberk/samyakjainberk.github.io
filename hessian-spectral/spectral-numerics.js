/* spectral-numerics.js
 * Numerics for the spectral-estimation widget accompanying
 * "Peeking at the Hessian" — works in the browser (window.SpecLab)
 * and in Node (module.exports) so the same code that runs in the
 * widget can be tested headlessly.
 *
 * Everything is Float64Array, row-major. Eigenvectors are stored as
 * COLUMNS of the returned `vectors` matrix.
 */
(function (global) {
'use strict';

/* ============================= RNG ============================= */

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class RNG {
  constructor(seed) {
    this.uniform = mulberry32(seed === undefined ? 12345 : seed);
    this._gaussCache = null;
  }
  gauss() {
    if (this._gaussCache !== null) {
      const g = this._gaussCache; this._gaussCache = null; return g;
    }
    let u = 0, v = 0;
    while (u === 0) u = this.uniform();
    v = this.uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    this._gaussCache = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
  rademacher() { return this.uniform() < 0.5 ? -1 : 1; }
  vec(p, dist) {
    const z = new Float64Array(p);
    if (dist === 'rademacher') for (let i = 0; i < p; i++) z[i] = this.rademacher();
    else for (let i = 0; i < p; i++) z[i] = this.gauss();
    return z;
  }
  int(n) { return Math.floor(this.uniform() * n); }
}

/* ======================= basic linear algebra ======================= */

function dot(x, y) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s; }
function norm(x) { return Math.sqrt(dot(x, x)); }
function scaleVec(x, a) { for (let i = 0; i < x.length; i++) x[i] *= a; return x; }
function axpy(y, a, x) { for (let i = 0; i < y.length; i++) y[i] += a * x[i]; return y; }
function copyVec(x) { return Float64Array.from(x); }

function matvec(A, x, p) {
  const y = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let s = 0; const off = i * p;
    for (let j = 0; j < p; j++) s += A[off + j] * x[j];
    y[i] = s;
  }
  return y;
}

function symmetrize(A, p) {
  for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) {
    const m = 0.5 * (A[i * p + j] + A[j * p + i]);
    A[i * p + j] = m; A[j * p + i] = m;
  }
  return A;
}

/* Modified Gram–Schmidt: orthonormalize `vecs` (array of Float64Array)
 * against `against` (array of Float64Array, assumed orthonormal) and
 * against each other. Vectors whose remaining norm is < tol are replaced
 * by fresh random vectors (re-orthogonalized) if rng given, else dropped.
 * Returns { vecs, replaced } — replaced = count of rank-deficient columns. */
function orthonormalize(vecs, against, rng, tol) {
  tol = tol || 1e-10;
  const out = [];
  let replaced = 0;
  const prior = against ? against.slice() : [];
  for (let c = 0; c < vecs.length; c++) {
    let v = copyVec(vecs[c]);
    for (let pass = 0; pass < 2; pass++) {
      for (const u of prior) axpy(v, -dot(u, v), u);
      for (const u of out) axpy(v, -dot(u, v), u);
    }
    let nv = norm(v);
    if (nv < tol) {
      if (!rng) continue;
      replaced++;
      let tries = 0;
      do {
        v = rng.vec(v.length, 'gaussian');
        for (let pass = 0; pass < 2; pass++) {
          for (const u of prior) axpy(v, -dot(u, v), u);
          for (const u of out) axpy(v, -dot(u, v), u);
        }
        nv = norm(v);
      } while (nv < tol && ++tries < 20);
      if (nv < tol) continue;
    }
    scaleVec(v, 1 / nv);
    out.push(v);
  }
  return { vecs: out, replaced };
}

/* ================== symmetric eigensolver (tred2 + tqli) ================== */
/* Householder reduction to tridiagonal form. A (p×p row-major) is
 * OVERWRITTEN with the orthogonal transform Q when wantV. */
function tred2(a, n, d, e, wantV) {
  for (let i = n - 1; i >= 1; i--) {
    const l = i - 1;
    let h = 0, scale = 0;
    if (l > 0) {
      for (let k = 0; k <= l; k++) scale += Math.abs(a[i * n + k]);
      if (scale === 0) e[i] = a[i * n + l];
      else {
        for (let k = 0; k <= l; k++) { a[i * n + k] /= scale; h += a[i * n + k] * a[i * n + k]; }
        let f = a[i * n + l];
        const g0 = (f >= 0 ? -Math.sqrt(h) : Math.sqrt(h));
        e[i] = scale * g0;
        h -= f * g0;
        a[i * n + l] = f - g0;
        f = 0;
        for (let j = 0; j <= l; j++) {
          a[j * n + i] = a[i * n + j] / h;
          let g = 0;
          for (let k = 0; k <= j; k++) g += a[j * n + k] * a[i * n + k];
          for (let k = j + 1; k <= l; k++) g += a[k * n + j] * a[i * n + k];
          e[j] = g / h;
          f += e[j] * a[i * n + j];
        }
        const hh = f / (h + h);
        for (let j = 0; j <= l; j++) {
          f = a[i * n + j];
          const g = e[j] - hh * f;
          e[j] = g;
          for (let k = 0; k <= j; k++) a[j * n + k] -= (f * e[k] + g * a[i * n + k]);
        }
      }
    } else {
      e[i] = a[i * n + l];
    }
    d[i] = h;
  }
  d[0] = 0;
  e[0] = 0;
  for (let i = 0; i < n; i++) {
    const l = i - 1;
    if (wantV) {
      if (d[i] !== 0) {
        for (let j = 0; j <= l; j++) {
          let g = 0;
          for (let k = 0; k <= l; k++) g += a[i * n + k] * a[k * n + j];
          for (let k = 0; k <= l; k++) a[k * n + j] -= g * a[k * n + i];
        }
      }
    }
    d[i] = a[i * n + i];
    if (wantV) {
      a[i * n + i] = 1;
      for (let j = 0; j <= l; j++) { a[j * n + i] = 0; a[i * n + j] = 0; }
    }
  }
}

/* QL with implicit shifts on a tridiagonal (d = diagonal, e = subdiagonal
 * with e[0] unused-in / shifted internally). If z is non-null (n×n row-major,
 * initialized to Q from tred2 or the identity), its COLUMNS accumulate the
 * eigenvectors. d is overwritten with eigenvalues (unsorted). */
function tqli(d, e, n, z) {
  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0;
  for (let l = 0; l < n; l++) {
    let iter = 0, m;
    do {
      for (m = l; m < n - 1; m++) {
        const dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
        if (Math.abs(e[m]) <= Number.EPSILON * dd) break;
      }
      if (m !== l) {
        if (iter++ === 60) throw new Error('tqli: too many iterations');
        let g = (d[l + 1] - d[l]) / (2 * e[l]);
        let r = Math.hypot(g, 1);
        g = d[m] - d[l] + e[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
        let s = 1, c = 1, p2 = 0;
        let underflow = false;
        for (let i = m - 1; i >= l; i--) {
          let f = s * e[i];
          const b = c * e[i];
          r = Math.hypot(f, g);
          e[i + 1] = r;
          if (r === 0) { d[i + 1] -= p2; e[m] = 0; underflow = true; break; }
          s = f / r; c = g / r;
          g = d[i + 1] - p2;
          r = (d[i] - g) * s + 2 * c * b;
          p2 = s * r;
          d[i + 1] = g + p2;
          g = c * r - b;
          if (z) {
            for (let k = 0; k < n; k++) {
              f = z[k * n + i + 1];
              z[k * n + i + 1] = s * z[k * n + i] + c * f;
              z[k * n + i] = c * z[k * n + i] - s * f;
            }
          }
        }
        if (underflow) continue;
        d[l] -= p2; e[l] = g; e[m] = 0;
      }
    } while (m !== l);
  }
}

function sortEigen(d, z, n) {
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => d[a] - d[b]);
  const values = new Float64Array(n);
  let vectors = null;
  for (let i = 0; i < n; i++) values[i] = d[idx[i]];
  if (z) {
    vectors = new Float64Array(n * n);
    for (let j = 0; j < n; j++) {
      const src = idx[j];
      for (let k = 0; k < n; k++) vectors[k * n + j] = z[k * n + src];
    }
  }
  return { values, vectors };
}

/* Full symmetric eigendecomposition. Returns eigenvalues ASCENDING and
 * (optionally) eigenvectors as columns. Does not modify A. */
function eigenSym(A, p, wantVectors) {
  const a = Float64Array.from(A);
  const d = new Float64Array(p), e = new Float64Array(p);
  if (p === 1) return { values: Float64Array.of(A[0]), vectors: wantVectors ? Float64Array.of(1) : null };
  tred2(a, p, d, e, !!wantVectors);
  tqli(d, e, p, wantVectors ? a : null);
  return sortEigen(d, wantVectors ? a : null, p);
}

/* Eigendecomposition of a symmetric tridiagonal (alpha: k diag, beta: k-1 sub).
 * Returns values ascending; vectors (k×k, columns) if wantVectors. */
function eigTridiagonal(alpha, beta, wantVectors) {
  const k = alpha.length;
  const d = Float64Array.from(alpha);
  const e = new Float64Array(k);
  for (let i = 1; i < k; i++) e[i] = beta[i - 1]; // tqli shifts internally
  let z = null;
  if (wantVectors) {
    z = new Float64Array(k * k);
    for (let i = 0; i < k; i++) z[i * k + i] = 1;
  }
  if (k === 1) return { values: Float64Array.of(alpha[0]), vectors: wantVectors ? Float64Array.of(1) : null };
  tqli(d, e, k, z);
  return sortEigen(d, z, k);
}

/* ========================= matrix ensembles ========================= */

function buildGOE(p, rng, scale) {
  scale = scale || 1;
  const A = new Float64Array(p * p);
  const s = scale / Math.sqrt(2 * p);
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) A[i * p + j] = rng.gauss();
  // (M + M^T)/sqrt(2p): semicircle support ~ [-2*scale, 2*scale]
  for (let i = 0; i < p; i++) for (let j = i; j < p; j++) {
    const v = (A[i * p + j] + A[j * p + i]) * s;
    A[i * p + j] = v; A[j * p + i] = v;
  }
  return A;
}

/* symmetric matrix with every element iid Bernoulli(q) in {0,1}:
 * sample the upper triangle (incl. diagonal) and mirror. Top eigenvalue
 * sits near p·q (Perron-type outlier) over a semicircle bulk. */
function buildBernoulli(p, rng, q) {
  q = (q === undefined || !isFinite(q)) ? 0.5 : Math.min(1, Math.max(0, q));
  const A = new Float64Array(p * p);
  for (let i = 0; i < p; i++) for (let j = i; j < p; j++) {
    const v = rng.uniform() < q ? 1 : 0;
    A[i * p + j] = v; A[j * p + i] = v;
  }
  return A;
}

function buildWishart(p, rng, aspect) {
  // A = X X^T / nw, X p×nw. aspect = nw/p (>= 0.1). Marchenko–Pastur.
  const nw = Math.max(2, Math.round(aspect * p));
  const X = new Float64Array(p * nw);
  for (let i = 0; i < X.length; i++) X[i] = rng.gauss();
  const A = new Float64Array(p * p);
  for (let i = 0; i < p; i++) {
    for (let j = i; j < p; j++) {
      let s = 0;
      for (let k = 0; k < nw; k++) s += X[i * nw + k] * X[j * nw + k];
      s /= nw;
      A[i * p + j] = s; A[j * p + i] = s;
    }
  }
  return A;
}

function buildSpiked(p, rng, opts) {
  const numSpikes = Math.max(0, Math.floor(opts.numSpikes === undefined ? 3 : opts.numSpikes));
  const spikeMax = opts.spikeMax === undefined ? 8 : opts.spikeMax;
  const spikeMin = opts.spikeMin === undefined ? 4 : opts.spikeMin;
  const negFrac = opts.negFrac === undefined ? 0 : opts.negFrac;
  const A = buildGOE(p, rng, 1);
  for (let sIdx = 0; sIdx < numSpikes; sIdx++) {
    const u = rng.vec(p, 'gaussian'); scaleVec(u, 1 / norm(u));
    let lam = numSpikes === 1 ? spikeMax
      : spikeMin + (spikeMax - spikeMin) * sIdx / (numSpikes - 1);
    if (rng.uniform() < negFrac) lam = -lam;
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) A[i * p + j] += lam * u[i] * u[j];
  }
  return A;
}

/* Eigenvalues placed in `numClusters` tight clusters (multiplicity!), then
 * conjugated by a random orthogonal matrix. Great for block-Lanczos demos. */
function buildClustered(p, rng, opts) {
  const numClusters = Math.max(1, Math.floor(opts.numClusters || 4));
  const spread = opts.spread === undefined ? 1e-4 : opts.spread;
  const lo = opts.lo === undefined ? -1 : opts.lo;
  const hi = opts.hi === undefined ? 5 : opts.hi;
  const centers = [];
  for (let c = 0; c < numClusters; c++)
    centers.push(numClusters === 1 ? hi : lo + (hi - lo) * c / (numClusters - 1));
  const lam = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    const c = centers[i % numClusters];
    lam[i] = c + spread * rng.gauss();
  }
  return rotateDiag(lam, p, rng);
}

/* log-spaced eigenvalues from `hi` down over `decades` decades, rotated. */
function buildLogSpaced(p, rng, opts) {
  const decades = opts.decades === undefined ? 3 : opts.decades;
  const hi = opts.hi === undefined ? 10 : opts.hi;
  const lam = new Float64Array(p);
  for (let i = 0; i < p; i++) lam[i] = hi * Math.pow(10, -decades * i / (p - 1 || 1));
  return rotateDiag(lam, p, rng);
}

function rotateDiag(lam, p, rng) {
  // random orthogonal Q via MGS on a gaussian matrix, A = Q diag(lam) Q^T
  const cols = [];
  for (let j = 0; j < p; j++) cols.push(rng.vec(p, 'gaussian'));
  const Q = orthonormalize(cols, null, rng).vecs;
  const A = new Float64Array(p * p);
  for (let k = 0; k < p; k++) {
    const q = Q[k], l = lam[k];
    for (let i = 0; i < p; i++) {
      const li = l * q[i];
      for (let j = i; j < p; j++) A[i * p + j] += li * q[j];
    }
  }
  for (let i = 0; i < p; i++) for (let j = 0; j < i; j++) A[i * p + j] = A[j * p + i];
  return A;
}

/* ==================== MLP + exact HVP (Pearlmutter R-op) ==================== */
/* One-hidden-layer MLP, MSE loss L = 1/(2n) Σ ||y - t||².
 * Parameter layout (flattened, length P):
 *   W1 (h×dIn), b1 (h), W2 (dOut×h), b2 (dOut)                              */
class MLP {
  constructor(opts) {
    this.dIn = Math.max(1, Math.floor(opts.dIn)); this.h = Math.max(1, Math.floor(opts.hidden)); this.dOut = Math.max(1, Math.floor(opts.dOut));
    this.n = Math.max(1, Math.floor(opts.n));
    this.act = opts.activation || 'tanh';
    this.P = this.h * this.dIn + this.h + this.dOut * this.h + this.dOut;
    const rng = new RNG(opts.seed === undefined ? 7 : opts.seed);
    this.rng = rng;
    // data
    this.X = new Float64Array(this.n * this.dIn);
    for (let i = 0; i < this.X.length; i++) this.X[i] = rng.gauss();
    this.T = new Float64Array(this.n * this.dOut);
    if ((opts.targetMode || 'teacher') === 'teacher') {
      const tw = this._initW(new RNG((opts.seed === undefined ? 7 : opts.seed) + 1000), 1.0);
      const noise = opts.noise === undefined ? 0.1 : opts.noise;
      for (let s = 0; s < this.n; s++) {
        const y = this._forwardOne(tw, s).y;
        for (let o = 0; o < this.dOut; o++) this.T[s * this.dOut + o] = y[o] + noise * rng.gauss();
      }
    } else {
      for (let i = 0; i < this.T.length; i++) this.T[i] = rng.gauss();
    }
    this.w = this._initW(rng, 1.0);
  }
  _initW(rng, gain) {
    const w = new Float64Array(this.P);
    const s1 = gain / Math.sqrt(this.dIn), s2 = gain / Math.sqrt(this.h);
    let o = 0;
    for (let i = 0; i < this.h * this.dIn; i++) w[o++] = s1 * rng.gauss();
    for (let i = 0; i < this.h; i++) w[o++] = 0;
    for (let i = 0; i < this.dOut * this.h; i++) w[o++] = s2 * rng.gauss();
    for (let i = 0; i < this.dOut; i++) w[o++] = 0;
    return w;
  }
  _unpack(w) {
    const { dIn, h, dOut } = this;
    let o = 0;
    const W1 = w.subarray(o, o + h * dIn); o += h * dIn;
    const b1 = w.subarray(o, o + h); o += h;
    const W2 = w.subarray(o, o + dOut * h); o += dOut * h;
    const b2 = w.subarray(o, o + dOut);
    return { W1, b1, W2, b2 };
  }
  _phi(a) { // activation value + first + second derivative
    if (this.act === 'relu') {
      const v = a > 0 ? a : 0;
      return [v, a > 0 ? 1 : 0, 0];
    }
    const t = Math.tanh(a);
    const d1 = 1 - t * t;
    return [t, d1, -2 * t * d1];
  }
  _forwardOne(w, s) {
    const { dIn, h, dOut } = this;
    const { W1, b1, W2, b2 } = this._unpack(w);
    const x = this.X.subarray(s * dIn, (s + 1) * dIn);
    const a1 = new Float64Array(h), h1 = new Float64Array(h),
          d1 = new Float64Array(h), dd1 = new Float64Array(h);
    for (let i = 0; i < h; i++) {
      let s2 = b1[i];
      for (let j = 0; j < dIn; j++) s2 += W1[i * dIn + j] * x[j];
      a1[i] = s2;
      const ph = this._phi(s2);
      h1[i] = ph[0]; d1[i] = ph[1]; dd1[i] = ph[2];
    }
    const y = new Float64Array(dOut);
    for (let o = 0; o < dOut; o++) {
      let s2 = b2[o];
      for (let i = 0; i < h; i++) s2 += W2[o * h + i] * h1[i];
      y[o] = s2;
    }
    return { x, a1, h1, d1, dd1, y };
  }
  loss(w) {
    w = w || this.w;
    let L = 0;
    for (let s = 0; s < this.n; s++) {
      const { y } = this._forwardOne(w, s);
      for (let o = 0; o < this.dOut; o++) {
        const r = y[o] - this.T[s * this.dOut + o];
        L += 0.5 * r * r;
      }
    }
    return L / this.n;
  }
  grad(w) {
    w = w || this.w;
    const { dIn, h, dOut, n } = this;
    const { W2 } = this._unpack(w);
    const g = new Float64Array(this.P);
    const { W1: gW1, b1: gb1, W2: gW2, b2: gb2 } = this._unpack(g);
    for (let s = 0; s < n; s++) {
      const f = this._forwardOne(w, s);
      const del2 = new Float64Array(dOut);
      for (let o = 0; o < dOut; o++) del2[o] = (f.y[o] - this.T[s * dOut + o]) / n;
      for (let o = 0; o < dOut; o++) {
        gb2[o] += del2[o];
        for (let i = 0; i < h; i++) gW2[o * h + i] += del2[o] * f.h1[i];
      }
      const del1 = new Float64Array(h);
      for (let i = 0; i < h; i++) {
        let s2 = 0;
        for (let o = 0; o < dOut; o++) s2 += W2[o * h + i] * del2[o];
        del1[i] = s2 * f.d1[i];
      }
      for (let i = 0; i < h; i++) {
        gb1[i] += del1[i];
        for (let j = 0; j < dIn; j++) gW1[i * dIn + j] += del1[i] * f.x[j];
      }
    }
    return g;
  }
  /* Exact Hessian-vector product via forward-over-reverse (R-op). */
  hvp(v, w) {
    w = w || this.w;
    const { dIn, h, dOut, n } = this;
    const { W2 } = this._unpack(w);
    const { W1: V1, b1: vb1, W2: V2, b2: vb2 } = this._unpack(v);
    const Hv = new Float64Array(this.P);
    const { W1: HW1, b1: Hb1, W2: HW2, b2: Hb2 } = this._unpack(Hv);
    for (let s = 0; s < n; s++) {
      const f = this._forwardOne(w, s);
      // tangent forward
      const da1 = new Float64Array(h), dh1 = new Float64Array(h);
      for (let i = 0; i < h; i++) {
        let s2 = vb1[i];
        for (let j = 0; j < dIn; j++) s2 += V1[i * dIn + j] * f.x[j];
        da1[i] = s2;
        dh1[i] = f.d1[i] * s2;
      }
      const dy = new Float64Array(dOut);
      for (let o = 0; o < dOut; o++) {
        let s2 = vb2[o];
        for (let i = 0; i < h; i++) s2 += V2[o * h + i] * f.h1[i] + W2[o * h + i] * dh1[i];
        dy[o] = s2;
      }
      // primal backward
      const del2 = new Float64Array(dOut);
      for (let o = 0; o < dOut; o++) del2[o] = (f.y[o] - this.T[s * dOut + o]) / n;
      // tangent backward:  del2dot = dy / n   (MSE: dL/dy = (y - t)/n)
      const del2dot = new Float64Array(dOut);
      for (let o = 0; o < dOut; o++) del2dot[o] = dy[o] / n;
      // output-layer blocks
      for (let o = 0; o < dOut; o++) {
        Hb2[o] += del2dot[o];
        for (let i = 0; i < h; i++)
          HW2[o * h + i] += del2dot[o] * f.h1[i] + del2[o] * dh1[i];
      }
      // hidden-layer tangent backward:
      // del1 = (W2^T del2) ⊙ phi'(a1)
      // del1dot = (V2^T del2 + W2^T del2dot) ⊙ phi'(a1) + (W2^T del2) ⊙ phi''(a1) ⊙ da1
      const del1 = new Float64Array(h), del1dot = new Float64Array(h);
      for (let i = 0; i < h; i++) {
        let w2d = 0, tang = 0;
        for (let o = 0; o < dOut; o++) {
          w2d += W2[o * h + i] * del2[o];
          tang += V2[o * h + i] * del2[o] + W2[o * h + i] * del2dot[o];
        }
        del1[i] = w2d * f.d1[i];
        del1dot[i] = tang * f.d1[i] + w2d * f.dd1[i] * da1[i];
      }
      for (let i = 0; i < h; i++) {
        Hb1[i] += del1dot[i];
        for (let j = 0; j < dIn; j++)
          HW1[i * dIn + j] += del1dot[i] * f.x[j] + del1[i] * 0 /* x has no tangent */;
      }
    }
    return Hv;
  }
  train(steps, lr) {
    const hist = [];
    for (let t = 0; t < steps; t++) {
      const g = this.grad(this.w);
      for (let i = 0; i < this.P; i++) this.w[i] -= lr * g[i];
      if (t % Math.max(1, Math.floor(steps / 50)) === 0 || t === steps - 1)
        hist.push({ step: t, loss: this.loss(this.w) });
    }
    return hist;
  }
  fullHessian() {
    const P = this.P;
    const H = new Float64Array(P * P);
    const e = new Float64Array(P);
    for (let j = 0; j < P; j++) {
      e.fill(0); e[j] = 1;
      const col = this.hvp(e, this.w);
      for (let i = 0; i < P; i++) H[i * P + j] = col[i];
    }
    return symmetrize(H, P);
  }
}

/* ====================== operator abstraction ====================== */
/* op = { p, mv(v) -> Av, count } — count = number of matvecs (HVPs). */

function makeMatOp(A, p) {
  const op = {
    p, count: 0,
    mv(v) { op.count++; return matvec(A, v, p); },
  };
  return op;
}

function makeHvpOp(mlp) {
  const op = {
    p: mlp.P, count: 0,
    mv(v) { op.count++; return mlp.hvp(v); },
  };
  return op;
}

/* Deflated operator: (I - UU^T) A (I - UU^T), with U an array of
 * orthonormal Float64Arrays. Projections are not counted as matvecs. */
function makeDeflatedOp(op, U) {
  function project(v) {
    const w = copyVec(v);
    for (const u of U) axpy(w, -dot(u, w), u);
    return w;
  }
  return {
    p: op.p,
    get count() { return op.count; },
    mv(v) { return project(op.mv(project(v))); },
    project,
  };
}

/* ===================== power iteration + deflation ===================== */
/* Extract numEigs eigenpairs (largest |λ| first). Each new iterate is kept
 * orthogonal to previously found eigenvectors (deflation by projection).
 * shift c: iterate on A + cI to target the top of the shifted spectrum. */
function powerDeflation(op, opts) {
  const p = op.p;
  const numEigs = Math.min(Math.max(1, Math.floor(opts.numEigs || 3)), p);
  const iters = Math.max(1, Math.floor(opts.iters || 100));
  const shift = opts.shift || 0;
  const rng = new RNG(opts.seed === undefined ? 1 : opts.seed);
  const found = [];   // {value, vector, history: [rayleigh per iter]}
  const U = [];
  for (let m = 0; m < numEigs; m++) {
    let x = rng.vec(p, 'gaussian');
    for (const u of U) axpy(x, -dot(u, x), u);
    scaleVec(x, 1 / norm(x));
    const history = [];
    let rq = 0;
    for (let t = 0; t < iters; t++) {
      let y = op.mv(x);
      if (shift !== 0) axpy(y, shift, x);
      for (const u of U) axpy(y, -dot(u, y), u);   // deflate
      // exact signed Rayleigh quotient of the ORIGINAL A at the current
      // unit iterate x, at zero extra matvecs: x^T A x = x^T (A+cI) x − c
      history.push(dot(x, y) - shift);
      const ny = norm(y);
      if (ny < 1e-300) break;                       // hit an exact null space
      scaleVec(y, 1 / ny);
      x = y;
    }
    // one extra matvec for an accurate signed Rayleigh quotient
    const Ax = op.mv(x);
    rq = dot(x, Ax);
    found.push({ value: rq, vector: x, history });
    U.push(x);
  }
  return { eigs: found, matvecs: op.count };
}

/* ========================= Lanczos iteration ========================= */
/* Returns tridiagonal (alpha, beta), basis V (array of k vectors), Ritz
 * values/weights/vectors. reorth: full re-orthogonalization (2× MGS).
 * startVec: optional start (will be normalized; projected against `against`).
 * against: array of orthonormal vectors to keep the whole Krylov space
 * orthogonal to (used by deflation / orthogonalized probes). */
function lanczos(op, opts) {
  const p = op.p;
  const k = Math.min(Math.max(1, Math.floor(opts.k)), p);
  const reorth = opts.reorth !== false;
  const against = opts.against || [];
  const rng = new RNG(opts.seed === undefined ? 2 : opts.seed);
  let v = opts.startVec ? copyVec(opts.startVec) : rng.vec(p, opts.dist || 'gaussian');
  for (const u of against) axpy(v, -dot(u, v), u);
  let nv = norm(v);
  if (nv < 1e-12) return { alpha: [], beta: [], V: [], breakdowns: 0, exhausted: true };
  scaleVec(v, 1 / nv);
  const V = [v];
  const alpha = [], beta = [];
  let breakdowns = 0, exhausted = false, betaFinal = 0;
  let vPrev = null, betaPrev = 0;
  for (let j = 0; j < k; j++) {
    const vj = V[j];
    let w = op.mv(vj);
    const a = dot(vj, w);
    alpha.push(a);
    axpy(w, -a, vj);
    if (vPrev) axpy(w, -betaPrev, vPrev);
    if (reorth) {
      for (let pass = 0; pass < 2; pass++) {
        for (const u of against) axpy(w, -dot(u, w), u);
        for (const u of V) axpy(w, -dot(u, w), u);
      }
    } else {
      for (const u of against) axpy(w, -dot(u, w), u);
    }
    const b = norm(w);
    if (j === k - 1) { betaFinal = b; break; }  // β_k: norm of the final residual, for residual bounds
    if (b < 1e-10 * (Math.abs(a) + 1)) {
      // breakdown: Krylov space is invariant; restart with a fresh
      // random vector orthogonal to everything found so far.
      breakdowns++;
      const res = orthonormalize([rng.vec(p, 'gaussian')], against.concat(V), rng);
      if (res.vecs.length === 0) { exhausted = true; break; }
      V.push(res.vecs[0]);
      beta.push(0);
      vPrev = vj; betaPrev = 0;
    } else {
      scaleVec(w, 1 / b);
      V.push(w);
      beta.push(b);
      vPrev = vj; betaPrev = b;
    }
  }
  return { alpha, beta, V, breakdowns, exhausted, betaFinal };
}

/* Ritz pairs from a Lanczos run. Returns values ASCENDING with
 * weights τ_j = (S_{1j})², residual bounds β_k|S_{kj}| (using the true
 * final residual norm β_k = lz.betaFinal), and (optionally) Ritz vectors
 * y_j = V s_j. In exact arithmetic ||A y_j − θ_j y_j|| = β_k |S_{kj}|. */
function ritzFromLanczos(lz, wantVectors) {
  const k = lz.alpha.length;
  if (k === 0) return { values: [], weights: [], residuals: [], vectors: [] };
  const { values, vectors: S } = eigTridiagonal(lz.alpha, lz.beta, true);
  const weights = new Float64Array(k), residuals = new Float64Array(k);
  const betaLast = lz.betaFinal || 0;
  for (let j = 0; j < k; j++) {
    const s1 = S[0 * k + j];
    weights[j] = s1 * s1;
    residuals[j] = Math.abs(betaLast * S[(k - 1) * k + j]);
  }
  let vectors = null;
  if (wantVectors) {
    vectors = [];
    const p = lz.V[0].length;
    for (let j = 0; j < k; j++) {
      const y = new Float64Array(p);
      for (let i = 0; i < k; i++) axpy(y, S[i * k + j], lz.V[i]);
      vectors.push(y);
    }
  }
  return { values: Array.from(values), weights: Array.from(weights), residuals: Array.from(residuals), vectors };
}

/* ========================= block Lanczos ========================= */
/* k block iterations of width b. Builds the dense (kb×kb) block-tridiagonal
 * projection. reorth: orthogonalize each new block against ALL previous
 * blocks (else only the previous two, per the three-term block recurrence).
 * startBlock: optional array of b vectors (will be orthonormalized).      */
function blockLanczos(op, opts) {
  const p = op.p;
  const b = Math.max(1, Math.floor(opts.b || 2));
  const k = Math.max(1, Math.min(Math.floor(opts.k), Math.floor(p / b)));
  const reorth = opts.reorth !== false;
  const against = opts.against || [];
  const rng = new RNG(opts.seed === undefined ? 3 : opts.seed);
  const dim = k * b;
  const T = new Float64Array(dim * dim);
  // initial block
  let start = opts.startBlock
    ? opts.startBlock.map(copyVec)
    : Array.from({ length: b }, () => rng.vec(p, opts.dist || 'gaussian'));
  let res0 = orthonormalize(start, against, rng);
  let replaced = res0.replaced;
  if (res0.vecs.length < b) return { T: null, exhausted: true };
  const blocks = [res0.vecs];             // each block: array of b vectors
  let Bprev = null;                        // b×b, B_{j-1}
  let exhausted = false;
  for (let j = 0; j < k; j++) {
    const Qj = blocks[j];
    // W = A Qj
    const W = Qj.map(q => op.mv(q));
    // Aj = Qj^T W (symmetrized)
    const Aj = new Float64Array(b * b);
    for (let r = 0; r < b; r++) for (let c = 0; c < b; c++) Aj[r * b + c] = dot(Qj[r], W[c]);
    for (let r = 0; r < b; r++) for (let c = r + 1; c < b; c++) {
      const m = 0.5 * (Aj[r * b + c] + Aj[c * b + r]);
      Aj[r * b + c] = m; Aj[c * b + r] = m;
    }
    // T block (j,j) = Aj
    for (let r = 0; r < b; r++) for (let c = 0; c < b; c++)
      T[(j * b + r) * dim + (j * b + c)] = Aj[r * b + c];
    if (j === k - 1) break;
    // W -= Qj Aj + Q_{j-1} B_{j-1}^T
    for (let c = 0; c < b; c++) {
      for (let r = 0; r < b; r++) axpy(W[c], -Aj[r * b + c], Qj[r]);
      if (Bprev) {
        const Qm = blocks[j - 1];
        // (Q_{j-1} B_{j-1}^T)_col c = Σ_r Q_{j-1}[r] * B_{j-1}[c*b+r]  (B row c? see below)
        for (let r = 0; r < b; r++) axpy(W[c], -Bprev[c * b + r], Qm[r]);
      }
    }
    // full reorthogonalization (or cleanup of the previous two only)
    const priorAll = reorth
      ? against.concat(...blocks)
      : against.concat(blocks[j], j > 0 ? blocks[j - 1] : []);
    for (let pass = 0; pass < 2; pass++)
      for (const w of W) for (const u of priorAll) axpy(w, -dot(u, w), u);
    // QR: W = Q_{j+1} Bj   (Bj upper-triangular, b×b; Bj[c*b+r] = <q_r, w_c> for r<=c)
    const Bj = new Float64Array(b * b);
    const Qn = [];
    let localReplaced = 0;
    for (let c = 0; c < b; c++) {
      let wv = W[c];
      for (let r = 0; r < Qn.length; r++) {
        const proj = dot(Qn[r], wv);
        Bj[c * b + r] = proj;
        axpy(wv, -proj, Qn[r]);
      }
      let nw = norm(wv);
      if (nw < 1e-10) {
        // rank-deficient: replace with random vector orthogonal to everything
        localReplaced++;
        const res = orthonormalize([rng.vec(p, 'gaussian')],
          against.concat(...blocks, Qn), rng);
        if (res.vecs.length === 0) { exhausted = true; break; }
        Qn.push(res.vecs[0]);
        Bj[c * b + c] = 0;
      } else {
        Bj[c * b + c] = nw;
        Qn.push(scaleVec(copyVec(wv), 1 / nw));
      }
    }
    if (exhausted || Qn.length < b) { exhausted = true; break; }
    replaced += localReplaced;
    // T blocks (j+1,j) = Bj^T-ish: A Qj = Q_{j-1} B_{j-1}^T + Qj Aj + Q_{j+1} Bj
    //   => Q_{j+1}^T A Qj = Bj  where Bj[c*b+r] = <q^{new}_r, (A Qj)_col c>
    // So T[(j+1)b+r, jb+c] = Bj[c*b+r], and symmetric transpose above.
    for (let r = 0; r < b; r++) for (let c = 0; c < b; c++) {
      T[((j + 1) * b + r) * dim + (j * b + c)] = Bj[c * b + r];
      T[(j * b + c) * dim + ((j + 1) * b + r)] = Bj[c * b + r];
    }
    blocks.push(Qn);
    Bprev = Bj;
  }
  const usedBlocks = blocks.length;
  const usedDim = usedBlocks * b;
  // extract the used part of T
  let Tu = T;
  if (usedDim < dim) {
    Tu = new Float64Array(usedDim * usedDim);
    for (let r = 0; r < usedDim; r++) for (let c = 0; c < usedDim; c++)
      Tu[r * usedDim + c] = T[r * dim + c];
  }
  return { T: Tu, dim: usedDim, b, blocks, exhausted, replaced };
}

/* Ritz pairs from a block-Lanczos run. weights = ||S_{1:b, j}||² (sums to b). */
function ritzFromBlockLanczos(bl, wantVectors) {
  if (!bl.T) return { values: [], weights: [], vectors: [] };
  const dim = bl.dim, b = bl.b;
  const { values, vectors: S } = eigenSym(bl.T, dim, true);
  const weights = new Float64Array(dim);
  for (let j = 0; j < dim; j++) {
    let s = 0;
    for (let r = 0; r < b; r++) { const v = S[r * dim + j]; s += v * v; }
    weights[j] = s;
  }
  let vectors = null;
  if (wantVectors) {
    vectors = [];
    const p = bl.blocks[0][0].length;
    for (let j = 0; j < dim; j++) {
      const y = new Float64Array(p);
      for (let i = 0; i < dim; i++) {
        const blk = Math.floor(i / b), r = i % b;
        axpy(y, S[i * dim + j], bl.blocks[blk][r]);
      }
      vectors.push(y);
    }
  }
  return { values: Array.from(values), weights: Array.from(weights), vectors };
}

/* ============================ SLQ family ============================ */
/* Standard / orthogonalized-probe / block / block+orth SLQ.
 * Returns per-probe quadrature { nodes, weights } with weights summing to 1
 * per probe (block probes: normalized by b), plus matvec count.
 *
 * opts: probes, k, dist ('rademacher'|'gaussian'), seed, reorth,
 *       orthProbes (bool), block (b>=2 => block SLQ), deflateU (array of
 *       orthonormal vectors: probes and Krylov restricted to complement)  */
function slq(op, opts) {
  const p = op.p;
  const probes = Math.max(1, Math.floor(opts.probes || 8));
  const k = Math.max(1, Math.floor(opts.k || 30));
  const b = Math.max(1, Math.floor(opts.block || 1));
  const dist = opts.dist || 'rademacher';
  const rng = new RNG(opts.seed === undefined ? 4 : opts.seed);
  const deflateU = opts.deflateU || [];
  const baseOp = deflateU.length ? makeDeflatedOp(op, deflateU) : op;
  const perProbe = [];
  const krylovBank = [];                 // accumulated basis vectors (orthProbes)
  let exhausted = false;
  for (let s = 0; s < probes; s++) {
    const against = deflateU.concat(opts.orthProbes ? krylovBank : []);
    if (b === 1) {
      let z = rng.vec(p, dist);
      const res = orthonormalize([z], against, null);
      if (res.vecs.length === 0) { exhausted = true; break; }
      const lz = lanczos(baseOp, {
        k, reorth: opts.reorth !== false, startVec: res.vecs[0],
        against, seed: rng.int(1e9),
      });
      if (lz.alpha.length === 0) { exhausted = true; break; }
      const rz = ritzFromLanczos(lz, false);
      perProbe.push({ nodes: rz.values, weights: rz.weights });
      if (opts.orthProbes) for (const v of lz.V) krylovBank.push(v);
    } else {
      const start = Array.from({ length: b }, () => rng.vec(p, dist));
      const startRes = orthonormalize(start, against, null);
      if (startRes.vecs.length < b) { exhausted = true; break; }
      const bl = blockLanczos(baseOp, {
        k, b, reorth: opts.reorth !== false, startBlock: startRes.vecs,
        against, seed: rng.int(1e9),
      });
      if (!bl.T) { exhausted = true; break; }
      const rz = ritzFromBlockLanczos(bl, false);
      perProbe.push({ nodes: rz.values, weights: rz.weights.map(w => w / b) });
      if (opts.orthProbes) for (const blk of bl.blocks) for (const v of blk) krylovBank.push(v);
    }
  }
  return { perProbe, matvecs: op.count, exhausted, probesRun: perProbe.length };
}

/* ====================== edge (Lanczos) + SLQ ====================== */
/* cycles × [run Lanczos on the deflated operator, harvest the mPerCycle
 * most extreme Ritz pairs], then SLQ on the final deflated operator.
 * edgeWhich: 'top' (largest θ), 'both' (both ends of the spectrum).     */
function edgePlusSlq(op, opts) {
  const p = op.p;
  const cycles = opts.cycles || 2;
  const kEdge = opts.kEdge || 30;
  const mPerCycle = opts.mPerCycle || 3;
  const edgeWhich = opts.edgeWhich || 'both';
  const rng = new RNG(opts.seed === undefined ? 5 : opts.seed);
  const U = [], deflated = [];
  for (let c = 0; c < cycles; c++) {
    const dop = U.length ? makeDeflatedOp(op, U) : op;
    const lz = lanczos(dop, { k: kEdge, reorth: true, against: U, seed: rng.int(1e9) });
    if (lz.alpha.length === 0) break;
    const rz = ritzFromLanczos(lz, true);
    const idx = rz.values.map((v, i) => i);
    let picked;
    if (edgeWhich === 'top') {
      picked = idx.sort((a2, b2) => rz.values[b2] - rz.values[a2]).slice(0, mPerCycle);
    } else {
      const desc = idx.slice().sort((a2, b2) => rz.values[b2] - rz.values[a2]);
      const nTop = Math.ceil(mPerCycle / 2), nBot = Math.floor(mPerCycle / 2);
      picked = desc.slice(0, nTop).concat(desc.slice(desc.length - nBot));
    }
    for (const j of picked) {
      const res = orthonormalize([rz.vectors[j]], U, null);
      if (res.vecs.length === 0) continue;
      U.push(res.vecs[0]);
      deflated.push({ value: rz.values[j], residual: rz.residuals[j] });
      if (U.length >= p - 2) break;
    }
    if (U.length >= p - 2) break;
  }
  const slqRes = slq(op, {
    probes: opts.probes || 8, k: opts.kSlq || 30, dist: opts.dist || 'rademacher',
    seed: rng.int(1e9), deflateU: U, orthProbes: !!opts.orthProbes,
    block: opts.block || 1, reorth: opts.reorth,
  });
  return { deflated, U, slq: slqRes, matvecs: op.count };
}

/* ====================== Kernel Polynomial Method ====================== */
/* Estimate spectral range with a short Lanczos run, rescale to [-1,1],
 * accumulate Chebyshev moments via Hutchinson, apply Jackson damping.
 * Returns { mu (damped), muRaw, lmin, lmax, evalDensity(tGrid) }        */
function kpm(op, opts) {
  const p = op.p;
  const K = Math.max(1, Math.floor(opts.degree || 80));
  const probes = Math.max(1, Math.floor(opts.probes || 8));
  const dist = opts.dist || 'rademacher';
  const jackson = opts.jackson !== false;
  const rng = new RNG(opts.seed === undefined ? 6 : opts.seed);
  // spectral range (a few extra matvecs)
  let lmin = opts.lmin, lmax = opts.lmax;
  if (lmin === undefined || lmax === undefined) {
    const lz = lanczos(op, { k: Math.min(40, p), reorth: true, seed: rng.int(1e9) });
    const rz = ritzFromLanczos(lz, false);
    const vmin = Math.min(...rz.values), vmax = Math.max(...rz.values);
    const pad = 0.05 * (vmax - vmin) + 1e-10;
    if (lmin === undefined) lmin = vmin - pad;
    if (lmax === undefined) lmax = vmax + pad;
  }
  const cScale = (lmax - lmin) / 2, cShift = (lmax + lmin) / 2;
  if (!(cScale > 1e-8 * Math.max(1, Math.abs(cShift))))
    throw new Error('KPM: degenerate spectral range [lmin, lmax]; spectrum is (numerically) a single point');
  // Ã v = (A v - cShift v)/cScale
  const mvT = (v) => {
    const y = op.mv(v);
    for (let i = 0; i < p; i++) y[i] = (y[i] - cShift * v[i]) / cScale;
    return y;
  };
  const muSum = new Float64Array(K + 1);
  for (let s = 0; s < probes; s++) {
    const z = rng.vec(p, dist);
    // normalize per probe by ||z||²: makes μ₀ = 1 exactly for BOTH
    // distributions (for Rademacher ||z||² = p anyway) — otherwise
    // gaussian probes rescale the whole density by O(1/√(sp)) noise
    const nz2 = dot(z, z);
    let w0 = copyVec(z);
    let w1 = mvT(z);
    muSum[0] += dot(z, w0) / nz2;
    if (K >= 1) muSum[1] += dot(z, w1) / nz2;
    for (let kk = 2; kk <= K; kk++) {
      const w2 = mvT(w1);
      for (let i = 0; i < p; i++) w2[i] = 2 * w2[i] - w0[i];
      muSum[kk] += dot(z, w2) / nz2;
      w0 = w1; w1 = w2;
    }
  }
  const muRaw = new Float64Array(K + 1);
  for (let kk = 0; kk <= K; kk++) muRaw[kk] = muSum[kk] / probes;
  // bracketing guard: if [lmin, lmax] misses part of the spectrum,
  // T_k(Ã) grows like cosh(k·acosh|x|) and the moments explode
  let maxAbsMu = 0;
  for (let kk = 0; kk <= K; kk++) maxAbsMu = Math.max(maxAbsMu, Math.abs(muRaw[kk]));
  if (!(maxAbsMu <= 2 * Math.max(1e-12, Math.abs(muRaw[0]))))
    throw new Error('KPM: moments diverged — [lmin, lmax] does not bracket the spectrum; widen the range');
  const mu = new Float64Array(K + 1);
  for (let kk = 0; kk <= K; kk++) {
    let g = 1;
    if (jackson) {
      const q = Math.PI / (K + 2);
      g = ((K + 2 - kk) * Math.sin(q) * Math.cos(kk * q)
         + Math.cos(q) * Math.sin(kk * q)) / ((K + 2) * Math.sin(q));
    }
    mu[kk] = g * muRaw[kk];
  }
  /* density on the ORIGINAL eigenvalue axis t (Jacobian included);
   * evalDensityMu evaluates any moment vector (e.g. muRaw for the
   * undamped estimate — same probes, zero extra matvecs) */
  function evalDensityMu(muArr, tGrid) {
    const out = new Float64Array(tGrid.length);
    for (let i = 0; i < tGrid.length; i++) {
      const x = (tGrid[i] - cShift) / cScale;
      if (x <= -1 || x >= 1) { out[i] = 0; continue; }
      // Chebyshev series via recurrence
      let acc = muArr[0];
      let Tm1 = 1, T0 = x;
      if (K >= 1) acc += 2 * muArr[1] * T0;
      for (let kk = 2; kk <= K; kk++) {
        const T1 = 2 * x * T0 - Tm1;
        acc += 2 * muArr[kk] * T1;
        Tm1 = T0; T0 = T1;
      }
      const val = acc / (Math.PI * Math.sqrt(1 - x * x));
      out[i] = val / cScale;   // Jacobian: dx/dt = 1/cScale
    }
    return out;
  }
  function evalDensity(tGrid) { return evalDensityMu(mu, tGrid); }
  /* density from only the first Kp+1 moments, with Jackson damping recomputed
   * for that degree — lets callers plot degree-convergence at zero extra matvecs */
  function evalDensityTrunc(Kp, tGrid) {
    Kp = Math.max(1, Math.min(K, Math.floor(Kp)));
    const muT = new Float64Array(K + 1);          // zeros beyond Kp
    const q = Math.PI / (Kp + 2);
    for (let kk = 0; kk <= Kp; kk++) {
      const g = jackson
        ? ((Kp + 2 - kk) * Math.sin(q) * Math.cos(kk * q)
           + Math.cos(q) * Math.sin(kk * q)) / ((Kp + 2) * Math.sin(q))
        : 1;
      muT[kk] = g * muRaw[kk];
    }
    return evalDensityMu(muT, tGrid);
  }
  return { mu: Array.from(mu), muRaw: Array.from(muRaw), lmin, lmax,
           evalDensity, evalDensityMu, evalDensityTrunc, matvecs: op.count };
}

/* ========================= density utilities ========================= */

function makeGrid(lo, hi, npts) {
  npts = npts || 400;
  const g = new Float64Array(npts);
  for (let i = 0; i < npts; i++) g[i] = lo + (hi - lo) * i / (npts - 1);
  return g;
}

/* Gaussian-smoothed density from sticks: Σ_i w_i g_σ(t - x_i).
 * If weights sum to 1 the result integrates to 1. σ is clamped to at
 * least the grid spacing — a kernel narrower than a grid cell aliases
 * (a stick's sampled mass then depends on where it falls in the cell). */
function smoothDensity(xs, ws, grid, sigma) {
  const out = new Float64Array(grid.length);
  const step = grid.length > 1 ? grid[1] - grid[0] : 0;
  const s = Math.max(sigma, step);
  const c = 1 / (s * Math.sqrt(2 * Math.PI));
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i], w = ws ? ws[i] : 1 / xs.length;
    for (let g = 0; g < grid.length; g++) {
      const d = (grid[g] - x) / s;
      if (Math.abs(d) < 8) out[g] += w * c * Math.exp(-0.5 * d * d);
    }
  }
  return out;
}

/* Aggregate SLQ per-probe results (first nProbes of them) into a smoothed
 * density on `grid`. Each probe's weights sum to 1; average over probes. */
function slqDensity(slqResult, grid, sigma, nProbes) {
  const use = slqResult.perProbe.slice(0, nProbes === undefined ? slqResult.perProbe.length : nProbes);
  const out = new Float64Array(grid.length);
  if (use.length === 0) return out;
  for (const pr of use) {
    const d = smoothDensity(pr.nodes, pr.weights, grid, sigma);
    for (let g = 0; g < grid.length; g++) out[g] += d[g] / use.length;
  }
  return out;
}

function trapz(y, grid) {
  let s = 0;
  for (let i = 1; i < grid.length; i++) s += 0.5 * (y[i] + y[i - 1]) * (grid[i] - grid[i - 1]);
  return s;
}

function l1DensityError(a, b2, grid) {
  const d = new Float64Array(grid.length);
  for (let i = 0; i < grid.length; i++) d[i] = Math.abs(a[i] - b2[i]);
  return trapz(d, grid);
}

/* Invert a (normalized) density into an eigenvalue-vs-index curve:
 * returns λ̂_i for i = 1..p, DESCENDING — λ̂_i is the point with mass
 * (i − ½)/p above it, i.e. the estimated i-th largest eigenvalue.
 * dens must integrate to ~1 on grid (renormalized internally).
 * Optional floor: clamp the curve from below (e.g. the smallest quadrature
 * node) — smoothing tails otherwise push the last quantiles ~2σ below the
 * true support, which reads as spurious negative eigenvalues. */
function screeFromDensity(grid, dens, p, floor) {
  const n = grid.length;
  // cumulative mass ABOVE grid[g], via trapezoid from the right
  const above = new Float64Array(n);
  for (let g = n - 2; g >= 0; g--)
    above[g] = above[g + 1] + 0.5 * (Math.max(0, dens[g]) + Math.max(0, dens[g + 1])) * (grid[g + 1] - grid[g]);
  const total = above[0] || 1;
  const out = new Float64Array(p);
  let g = n - 1;
  for (let i = 1; i <= p; i++) {
    const target = (i - 0.5) / p * total;
    while (g > 0 && above[g - 1] < target) g--;
    if (g === 0) { out[i - 1] = grid[0]; continue; }
    const a0 = above[g], a1 = above[g - 1];   // a1 >= target > a0 (above increases leftward)
    const w = a1 > a0 ? (target - a0) / (a1 - a0) : 0;
    out[i - 1] = grid[g] + w * (grid[g - 1] - grid[g]);
  }
  if (floor !== undefined && isFinite(floor))
    for (let i = 0; i < p; i++) if (out[i] < floor) out[i] = floor;
  return out;
}

/* Greedy nearest matching of estimated eigenvalues to true ones
 * (each true eigenvalue used at most once). Returns [{est, truth, err}]. */
function matchToTruth(estimates, truth) {
  const used = new Array(truth.length).fill(false);
  return estimates.map(est => {
    let best = -1, bestD = Infinity;
    for (let i = 0; i < truth.length; i++) {
      if (used[i]) continue;
      const d = Math.abs(truth[i] - est);
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best >= 0) used[best] = true;
    return { est, truth: best >= 0 ? truth[best] : NaN, err: bestD };
  });
}

/* ============================ exports ============================ */

const SpecLab = {
  RNG, mulberry32,
  dot, norm, axpy, scaleVec, copyVec, matvec, symmetrize, orthonormalize,
  eigenSym, eigTridiagonal,
  buildGOE, buildBernoulli, buildWishart, buildSpiked, buildClustered, buildLogSpaced, rotateDiag,
  MLP,
  makeMatOp, makeHvpOp, makeDeflatedOp,
  powerDeflation,
  lanczos, ritzFromLanczos,
  blockLanczos, ritzFromBlockLanczos,
  slq, edgePlusSlq, kpm,
  makeGrid, smoothDensity, slqDensity, trapz, l1DensityError, matchToTruth, screeFromDensity,
};

if (typeof module !== 'undefined' && module.exports) module.exports = SpecLab;
global.SpecLab = SpecLab;

})(typeof window !== 'undefined' ? window : globalThis);
