"""
train_capture.py — GPU training-dynamics capture for the training-spectra viewer.

Trains a scalar-output L-layer MLP and, at checkpoints along training, computes
the EXACT eigenspectra of three curvature matrices of the MSE loss
L(w) = 1/(2n) sum_i (f(x_i) - y_i)^2:

  GN  = (1/n) J J^T          (Gauss-Newton; J has per-sample gradients of f)
  F   = (1/n) sum_i r_i H_i  (functional Hessian: per-sample Hessians of f
                              contracted with the residuals r_i = f(x_i)-y_i)
  H   = GN + F               (full Hessian of the loss)

plus per-quantity SLQ estimates (12 probes x k=60) and their L1 error against
the exact smoothed density ("error statistics"), the loss at every step, and
the sharpness lambda_max(H) at every checkpoint. Output is one JSON consumed
by training-spectra.html.

Runs on GPU when available (p = 10k is a few seconds per checkpoint on an
A6000) and falls back to CPU. Example:

  python3 train_capture.py --depth 3 --width 66 --din 16 --n 256 --act tanh \
      --opt gd --lr 0.05 --steps 1500 --ckpts 30 --out training-capture.json
"""
import argparse
import json
import math
import time

import torch
import torch.func as tfn


def build_shapes(din, width, depth, dout=1):
    """depth = number of hidden layers; dout outputs (1 except CIFAR-10's 10)."""
    dims = [din] + [width] * depth + [dout]
    shapes = []
    for i in range(len(dims) - 1):
        shapes.append((dims[i + 1], dims[i]))   # W
        shapes.append((dims[i + 1],))           # b
    return shapes


def unflatten(flat, shapes):
    out, o = [], 0
    for sh in shapes:
        numel = 1
        for s in sh:
            numel *= s
        out.append(flat[o:o + numel].view(sh))
        o += numel
    return out


ACTS = {
    'tanh': torch.tanh,
    'relu': torch.relu,
    'gelu': torch.nn.functional.gelu,
    'elu': torch.nn.functional.elu,
    'linear': lambda x: x,
}


def make_forward(shapes, act):
    phi = ACTS[act]
    n_layers = len(shapes) // 2

    def forward(flat, X):
        ps = unflatten(flat, shapes)
        h = X
        for i in range(n_layers):
            W, b = ps[2 * i], ps[2 * i + 1]
            h = h @ W.T + b
            if i < n_layers - 1:
                h = phi(h)
        return h                     # (batch, dout); dout = 1 except CIFAR-10
    return forward


def init_params(shapes, gen, device):
    parts = []
    for sh in shapes:
        if len(sh) == 2:
            fan_in = sh[1]
            parts.append(torch.randn(sh, generator=gen, device=device).reshape(-1) / math.sqrt(fan_in))
        else:
            parts.append(torch.zeros(sh[0], device=device))
    return torch.cat(parts)


def smooth_density(vals, grid, sigma):
    """Gaussian-smoothed spectral density (weights 1/len(vals) each)."""
    d = (grid[None, :] - vals[:, None]) / sigma
    return torch.exp(-0.5 * d * d).sum(0) / (len(vals) * sigma * math.sqrt(2 * math.pi))


def l1_err(a, b, grid):
    return torch.trapz((a - b).abs(), grid).item()


def slq_spectrum(matvec, p, probes, k, gen, device):
    """Standard SLQ with full reorthogonalization; returns (nodes, weights) lists."""
    nodes, weights = [], []
    for _ in range(probes):
        z = torch.randint(0, 2, (p,), generator=gen, device=device, dtype=torch.float32) * 2 - 1
        v = z / z.norm()
        V = [v]
        alpha, beta = [], []
        b_prev, v_prev = 0.0, None
        for j in range(k):
            w = matvec(V[j])
            a = torch.dot(V[j], w)
            alpha.append(a.item())
            w = w - a * V[j]
            if v_prev is not None:
                w = w - b_prev * v_prev
            for u in V:                       # full reorthogonalization
                w = w - torch.dot(u, w) * u
            b = w.norm()
            if j == k - 1:
                break
            if b < 1e-8:
                break
            w = w / b
            beta.append(b.item())
            v_prev, b_prev = V[j], b
            V.append(w)
        m = len(alpha)
        T = torch.zeros(m, m, dtype=torch.float64)
        for i2 in range(m):
            T[i2, i2] = alpha[i2]
        for i2, bb in enumerate(beta[:m - 1]):
            T[i2, i2 + 1] = bb
            T[i2 + 1, i2] = bb
        evals, S = torch.linalg.eigh(T)
        nodes.append(evals.tolist())
        weights.append((S[0, :] ** 2).tolist())
    return nodes, weights


def block_slq(M, p, b, s, k, gen, device):
    """Block SLQ: s random p x b Rademacher blocks, k block-Lanczos steps with
    full reorthogonalization; returns (nodes, weights) lists, one entry per block,
    weights tau_j = ||S_{1:b,j}||^2 / b (each block's weights sum to one)."""
    out_nodes, out_wts = [], []
    b = max(1, min(b, p))
    k_eff = max(1, min(k, p // b))
    for _ in range(s):
        Z = torch.randint(0, 2, (p, b), generator=gen, device=device, dtype=torch.float32) * 2 - 1
        Q, _ = torch.linalg.qr(Z)
        Qs = [Q]
        A_blocks, B_blocks = [], []
        Qprev, Bprev = None, None
        for j in range(k_eff):
            W = M @ Qs[j]
            A = Qs[j].T @ W
            A = 0.5 * (A + A.T)
            A_blocks.append(A)
            W = W - Qs[j] @ A
            if Qprev is not None:
                W = W - Qprev @ Bprev.T
            for U in Qs:                      # full reorthogonalization
                W = W - U @ (U.T @ W)
            if j == k_eff - 1:
                break
            Qn, Bn = torch.linalg.qr(W)
            dmax = torch.max(torch.abs(torch.diagonal(Bn))).item()
            if torch.min(torch.abs(torch.diagonal(Bn))).item() < 1e-7 * max(1.0, dmax):
                break                          # block Krylov breakdown
            B_blocks.append(Bn)
            Qprev, Bprev = Qs[j], Bn
            Qs.append(Qn)
        m = len(A_blocks)
        T = torch.zeros(m * b, m * b, device=device)
        for j, Ab in enumerate(A_blocks):
            T[j * b:(j + 1) * b, j * b:(j + 1) * b] = Ab
        for j, Bb in enumerate(B_blocks[:m - 1]):
            T[j * b:(j + 1) * b, (j + 1) * b:(j + 2) * b] = Bb.T
            T[(j + 1) * b:(j + 2) * b, j * b:(j + 1) * b] = Bb
        evals, S = torch.linalg.eigh(T.double())
        tau = (S[:b, :] ** 2).sum(0) / b
        out_nodes.append(evals.tolist())
        out_wts.append(tau.tolist())
    return out_nodes, out_wts


def slq_density(nodes, weights, grid, sigma):
    dens = torch.zeros_like(grid)
    for nd, wt in zip(nodes, weights):
        v = torch.tensor(nd, dtype=grid.dtype, device=grid.device)
        w = torch.tensor(wt, dtype=grid.dtype, device=grid.device)
        d = (grid[None, :] - v[:, None]) / sigma
        dens += (w[:, None] * torch.exp(-0.5 * d * d)).sum(0) / (sigma * math.sqrt(2 * math.pi))
    return dens / len(nodes)


def kpm_moments(M, p, probes, deg, a, b, gen, device):
    """Chebyshev moments mu_k = (1/p) Tr T_k(A~) via Hutchinson; A~ maps [a,b] -> [-1,1]."""
    mus = torch.zeros(deg + 1, dtype=torch.float64)
    scale = 2.0 / (b - a)
    shift = (a + b) / (b - a)

    def amap(v):
        return scale * (M @ v) - shift * v

    for _ in range(probes):
        z = torch.randint(0, 2, (p,), generator=gen, device=device, dtype=torch.float32) * 2 - 1
        w0 = z
        w1 = amap(z)
        mus[0] += p                       # z^T z = p exactly (Rademacher)
        mus[1] += torch.dot(z, w1).item()
        for k in range(2, deg + 1):
            wn = 2 * amap(w1) - w0
            mus[k] += torch.dot(z, wn).item()
            w0, w1 = w1, wn
    return (mus / (probes * p)).tolist()


def kpm_density(mu, a, b, grid):
    """Jackson-damped KPM density on the original axis, clipped at zero."""
    K = len(mu) - 1
    kk = torch.arange(K + 1, dtype=torch.float64)
    N = K + 1
    jack = ((N - kk + 1) * torch.cos(math.pi * kk / (N + 1))
            + torch.sin(math.pi * kk / (N + 1)) / math.tan(math.pi / (N + 1))) / (N + 1)
    x = (2 * grid.double() - (a + b)) / (b - a)
    inside = (x.abs() < 1)
    xs = x.clamp(-0.999999, 0.999999)
    theta = torch.arccos(xs)
    dens = torch.zeros_like(x)
    for k in range(K + 1):
        term = (jack[k] * mu[k]) * torch.cos(k * theta)
        dens += term if k == 0 else 2 * term
    dens = dens / (math.pi * torch.sqrt(1 - xs * xs)) * (2.0 / (b - a))
    dens = torch.where(inside, dens.clamp(min=0.0), torch.zeros_like(dens))
    return dens.to(grid.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=3, help='number of hidden layers')
    ap.add_argument('--width', type=int, default=66)
    ap.add_argument('--din', type=int, default=16)
    ap.add_argument('--n', type=int, default=256, help='dataset size')
    ap.add_argument('--batch', type=int, default=0, help='batch size; 0 = full batch')
    ap.add_argument('--act', choices=list(ACTS), default='tanh')
    ap.add_argument('--opt', choices=['gd', 'signgd', 'gn', 'spectral'], default='gd')
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--gn-damping', type=float, default=1e-3)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--ckpt-every', type=int, default=1,
                    help='compute spectra every N steps (default 1 = every iteration)')
    ap.add_argument('--ckpts', type=int, default=0,
                    help='legacy: exact number of evenly spaced checkpoints; used only if --ckpt-every 0')
    ap.add_argument('--dataset', choices=['teacher', 'parity', 'chebyshev', 'cifar10'],
                    default='teacher')
    ap.add_argument('--parity-k', type=int, default=3, help='sparse parity: number of relevant bits')
    ap.add_argument('--cheby-deg', type=int, default=4, help='chebyshev: polynomial degree')
    ap.add_argument('--cifar-size', type=int, default=8, help='CIFAR-10: images average-pooled to size x size')
    ap.add_argument('--noise', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--slq-probes', type=int, default=12)
    ap.add_argument('--slq-k', type=int, default=60)
    ap.add_argument('--bslq-b', type=int, default=4, help='block SLQ block size')
    ap.add_argument('--bslq-s', type=int, default=3, help='block SLQ number of blocks')
    ap.add_argument('--bslq-k', type=int, default=30, help='block SLQ Lanczos steps')
    ap.add_argument('--kpm', type=int, default=0, help='1 = also run KPM per checkpoint')
    ap.add_argument('--kpm-probes', type=int, default=8)
    ap.add_argument('--kpm-deg', type=int, default=80)
    ap.add_argument('--hvp-chunk', type=int, default=256)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='training-capture.json')
    ap.add_argument('--stream', default='', help='optional JSONL path; training progress is appended live')
    ap.add_argument('--init-from', default='',
                    help='path to a previous capture JSON; start from its final_params')
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    cgen = torch.Generator().manual_seed(args.seed + 1)

    n = args.n
    # ── dataset: X (n, din_eff), Y (n, C) ──
    if args.dataset == 'cifar10':
        din_eff, C = 3 * args.cifar_size ** 2, 10
    else:
        din_eff, C = args.din, 1
    shapes = build_shapes(din_eff, args.width, args.depth, C)
    forward = make_forward(shapes, args.act)
    flat = init_params(shapes, gen, dev)
    p = flat.numel()
    if args.init_from:
        prev = json.load(open(args.init_from))
        fp = prev.get('final_params')
        if fp is None or len(fp) != p:
            raise SystemExit(f'--init-from: incompatible capture '
                             f'(final_params len {len(fp) if fp else None} vs p={p})')
        flat = torch.tensor(fp, dtype=torch.float32, device=dev)
        print(f'continuing from {args.init_from}', flush=True)
    B = args.batch if 0 < args.batch <= n else n
    print(f'p = {p} parameters | dataset = {args.dataset} (d_in = {din_eff}, classes = {C}) '
          f'| device = {dev} | act = {args.act} | opt = {args.opt} '
          f'| lr = {args.lr} | batch = {B}/{n}', flush=True)

    stream = open(args.stream, 'w', buffering=1) if args.stream else None
    t_wall = time.time()

    def emit(obj):
        if stream:
            stream.write(json.dumps(obj) + '\n')

    emit(dict(t='config', config=dict(depth=args.depth, width=args.width, din=din_eff, n=n,
                                      dataset=args.dataset, dout=C, parity_k=args.parity_k,
                                      cheby_deg=args.cheby_deg, cifar_size=args.cifar_size,
                                      ckpt_every=args.ckpt_every, kpm=args.kpm, noise=args.noise,
                                      kpm_probes=args.kpm_probes, kpm_deg=args.kpm_deg,
                                      batch=B, act=args.act, opt=args.opt, lr=args.lr,
                                      steps=args.steps, gn_damping=args.gn_damping,
                                      seed=args.seed, p=p, slq_probes=args.slq_probes,
                                      slq_k=args.slq_k, bslq_b=args.bslq_b, bslq_s=args.bslq_s,
                                      bslq_k=args.bslq_k, device=str(dev))))

    if args.dataset == 'teacher':
        X = torch.randn(n, din_eff, generator=gen, device=dev)
        teacher = init_params(shapes, torch.Generator(device=dev).manual_seed(args.seed + 777), dev)
        with torch.no_grad():
            y = forward(teacher, X) + args.noise * torch.randn(n, 1, generator=gen, device=dev)
    elif args.dataset == 'parity':
        kbits = max(1, min(args.parity_k, din_eff))
        X = (torch.randint(0, 2, (n, din_eff), generator=gen, device=dev,
                           dtype=torch.float32) * 2 - 1)
        y = X[:, :kbits].prod(dim=1, keepdim=True) \
            + args.noise * torch.randn(n, 1, generator=gen, device=dev)
    elif args.dataset == 'chebyshev':
        X = torch.rand(n, din_eff, generator=gen, device=dev) * 2 - 1
        y = torch.cos(args.cheby_deg * torch.arccos(X[:, :1].clamp(-1, 1))) \
            + args.noise * torch.randn(n, 1, generator=gen, device=dev)
    else:                                        # cifar10
        import os
        import pickle
        cdir = next((c for c in ('/nas/ucb/samsj/cifar-10-batches-py',
                                 '/nas/ucb/samsj/data/cifar-10-batches-py',
                                 os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'cifar-10-batches-py'))
                     if os.path.isdir(c)), None)
        if cdir is None:
            raise FileNotFoundError('cifar-10-batches-py not found on this machine')
        xs, labels = [], []
        for bi in range(1, 6):
            with open(os.path.join(cdir, f'data_batch_{bi}'), 'rb') as fh:
                batch = pickle.load(fh, encoding='bytes')
            xs.append(torch.tensor(batch[b'data'], dtype=torch.float32))
            labels += list(batch[b'labels'])
        Xall = torch.cat(xs).view(-1, 3, 32, 32) / 255.0
        yall = torch.tensor(labels, dtype=torch.long)
        idx = torch.randperm(Xall.shape[0], generator=cgen)[:n]
        Xi, yi = Xall[idx], yall[idx]
        if args.cifar_size != 32:
            Xi = torch.nn.functional.adaptive_avg_pool2d(Xi, (args.cifar_size, args.cifar_size))
        Xi = Xi.reshape(n, -1)
        Xi = (Xi - Xi.mean()) / (Xi.std() + 1e-8)
        X = Xi.to(dev)
        y = torch.zeros(n, 10, device=dev)
        y[torch.arange(n), yi.to(dev)] = 1.0     # one-hot targets, MSE (noise flag ignored)

    def loss_fn(fl, Xb, yb):
        r = forward(fl, Xb) - yb                 # (batch, C)
        return 0.5 * (r * r).sum(-1).mean()

    grad_full = tfn.grad(lambda fl: loss_fn(fl, X, y))

    def per_sample_grads(fl):
        # rows of J are per-(sample, class) gradients of f: (n, C, p) -> (n*C, p)
        f_single = lambda f2, x: forward(f2, x[None, :])[0]              # (C,)
        Jnc = tfn.vmap(tfn.jacrev(f_single), in_dims=(None, 0))(fl, X)   # (n, C, p)
        return Jnc.reshape(-1, p)

    def hvp_full(fl, v):
        return tfn.jvp(grad_full, (fl,), (v,))[1]

    # weight-matrix slices for the spectral optimizer
    mat_slices, o = [], 0
    for sh in shapes:
        numel = int(torch.tensor(sh).prod())
        if len(sh) == 2:
            mat_slices.append((o, o + numel, sh))
        o += numel

    if args.ckpt_every >= 1:
        ck_steps = sorted(set(list(range(0, args.steps + 1, args.ckpt_every)) + [args.steps]))
    else:
        ck_steps = sorted(set(int(round(s)) for s in
                              torch.linspace(0, args.steps, max(2, args.ckpts)).tolist()))
    losses, checkpoints = [], []
    t_start = time.time()

    def checkpoint(step):
        t0 = time.time()
        J = per_sample_grads(flat).detach()                    # (n*C, p)
        G = (J.T @ J) / n                                      # (p, p); 1/n even for C > 1
        # full Hessian by chunked, vmapped HVPs over the identity
        Hcols = []
        eye = torch.eye(p, device=dev)
        for c0 in range(0, p, args.hvp_chunk):
            Vc = eye[c0:c0 + args.hvp_chunk]
            Hcols.append(tfn.vmap(lambda v: hvp_full(flat, v))(Vc).detach())
        H = torch.cat(Hcols, 0)
        H = 0.5 * (H + H.T)
        F = H - G
        ev = {}
        for name, M in (('H', H), ('GN', G), ('F', F)):
            ev[name] = torch.linalg.eigvalsh(M.float()).cpu()
        # SLQ and block-SLQ error statistics per quantity
        slq_l1, bslq_l1, bslq, kpm_l1, kpm = {}, {}, {}, {}, {}
        for name, M in (('H', H), ('GN', G), ('F', F)):
            evs = ev[name].to(dev)
            lo, hi = evs.min().item(), evs.max().item()
            span = max(hi - lo, 1e-8 * max(1.0, abs(hi), abs(lo)))
            sigma = span / 60
            grid = torch.linspace(lo - 0.06 * span - 4 * sigma, hi + 0.06 * span + 4 * sigma,
                                  600, device=dev)
            rho = smooth_density(evs, grid, sigma)
            nodes, wts = slq_spectrum(lambda v: M @ v, p, args.slq_probes, args.slq_k, gen, dev)
            est = slq_density(nodes, wts, grid, sigma)
            slq_l1[name] = l1_err(est, rho, grid)
            bnodes, bwts = block_slq(M, p, args.bslq_b, args.bslq_s, args.bslq_k, gen, dev)
            best = slq_density(bnodes, bwts, grid, sigma)
            bslq_l1[name] = l1_err(best, rho, grid)
            bslq[name] = dict(nodes=[[float(f'{v:.6g}') for v in blk] for blk in bnodes],
                              weights=[[float(f'{v:.6g}') for v in blk] for blk in bwts])
            if args.kpm:
                pad = 0.05 * span
                ka, kb = lo - pad, hi + pad
                mu = kpm_moments(M, p, args.kpm_probes, args.kpm_deg, ka, kb, gen, dev)
                kest = kpm_density(mu, ka, kb, grid)
                kpm_l1[name] = l1_err(kest, rho, grid)
                kpm[name] = dict(mu=[float(f'{v:.6g}') for v in mu], a=float(f'{ka:.6g}'), b=float(f'{kb:.6g}'))
        checkpoints.append(dict(
            step=step,
            loss=loss_fn(flat, X, y).item(),
            sharpness=ev['H'][-1].item(),
            evals={k2: [float(f'{v:.6g}') for v in ev[k2].tolist()] for k2 in ev},
            slq_l1=slq_l1,
            bslq_l1=bslq_l1,
            bslq=bslq,
            **(dict(kpm_l1=kpm_l1, kpm=kpm) if args.kpm else {}),
        ))
        ck = checkpoints[-1]
        emit(dict(t='ckpt', wall=round(time.time() - t_wall, 2), **ck))
        del H, G, F, J, Hcols
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'  ckpt @ step {step}: loss {checkpoints[-1]["loss"]:.5f}  '
              f'sharpness {checkpoints[-1]["sharpness"]:.4f}  '
              f'slq L1 H/GN/F = {slq_l1["H"]:.3f}/{slq_l1["GN"]:.3f}/{slq_l1["F"]:.3f}  '
              f'block L1 = {bslq_l1["H"]:.3f}/{bslq_l1["GN"]:.3f}/{bslq_l1["F"]:.3f}  '
              f'({time.time() - t0:.1f}s)', flush=True)

    for step in range(args.steps + 1):
        cur_loss = loss_fn(flat, X, y).item()
        if not math.isfinite(cur_loss):
            print(f'stopping early: non-finite loss at step {step} (diverged)', flush=True)
            break
        losses.append(cur_loss)
        emit(dict(t='loss', s=step, l=float(f'{losses[-1]:.6g}'), wall=round(time.time() - t_wall, 2)))
        if step in ck_steps:
            checkpoint(step)
        if step == args.steps:
            break
        # one optimizer step on a batch
        if B < n:
            idx = torch.randperm(n, generator=cgen)[:B].to(dev)
            Xb, yb = X[idx], y[idx]
        else:
            Xb, yb = X, y
        g = tfn.grad(lambda fl: loss_fn(fl, Xb, yb))(flat)
        with torch.no_grad():
            if args.opt == 'gd':
                flat -= args.lr * g
            elif args.opt == 'signgd':
                flat -= args.lr * torch.sign(g)
            elif args.opt == 'gn':
                f_single = lambda f2, x: forward(f2, x[None, :])[0]
                Jb = tfn.vmap(tfn.jacrev(f_single), in_dims=(None, 0))(flat, Xb)
                Jb = Jb.reshape(-1, p)                          # (B*C, p)
                rb = (forward(flat, Xb) - yb).reshape(-1)       # (B*C,)
                Bn = Xb.shape[0]
                A = Jb @ Jb.T / Bn + args.gn_damping * torch.eye(len(rb), device=dev)
                u = torch.linalg.solve(A, rb)
                flat -= args.lr * (Jb.T @ u) / Bn
            elif args.opt == 'spectral':
                step_v = torch.zeros_like(g)
                for (a0, a1, sh) in mat_slices:
                    Gm = g[a0:a1].view(sh)
                    U, S, Vh = torch.linalg.svd(Gm, full_matrices=False)
                    step_v[a0:a1] = (U @ Vh).reshape(-1)
                mask = torch.ones_like(g, dtype=torch.bool)
                for (a0, a1, _) in mat_slices:
                    mask[a0:a1] = False
                step_v[mask] = torch.sign(g[mask])
                flat -= args.lr * step_v

    out = dict(
        config=dict(depth=args.depth, width=args.width, din=din_eff, n=n, batch=B,
                    dataset=args.dataset, dout=C, parity_k=args.parity_k,
                    cheby_deg=args.cheby_deg, cifar_size=args.cifar_size,
                    ckpt_every=args.ckpt_every, kpm=args.kpm, noise=args.noise,
                    kpm_probes=args.kpm_probes, kpm_deg=args.kpm_deg,
                    act=args.act, opt=args.opt, lr=args.lr, steps=args.steps,
                    gn_damping=args.gn_damping, seed=args.seed, p=p,
                    slq_probes=args.slq_probes, slq_k=args.slq_k,
                    bslq_b=args.bslq_b, bslq_s=args.bslq_s, bslq_k=args.bslq_k,
                    device=str(dev)),
        losses=[float(f'{v:.6g}') for v in losses],
        final_params=[float(f'{v:.8g}') for v in flat.tolist()],
        checkpoints=checkpoints,
    )
    with open(args.out, 'w') as fh:
        json.dump(out, fh)
    emit(dict(t='done', wall=round(time.time() - t_wall, 2), out=args.out))
    if stream:
        stream.close()
    print(f'wrote {args.out} ({len(checkpoints)} checkpoints, p={p}) '
          f'in {time.time() - t_start:.0f}s total', flush=True)


if __name__ == '__main__':
    main()
