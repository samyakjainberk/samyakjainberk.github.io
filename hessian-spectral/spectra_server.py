"""
spectra_server.py — serves the blog directory AND launches live GPU training
captures for the "Training-time spectra" card embedded in the blogpost.

Endpoints (everything else is static file serving of this directory):
  GET  /api/ping           -> {ok, gpu, running}
  POST /api/run            -> body: JSON hyperparameters; returns {id}
  GET  /api/tail?id=&off=  -> {off, done, exit, data} new bytes of the run's
                              JSONL stream starting at byte offset `off`
  GET  /api/stop?id=       -> terminates the run

Runs are subprocesses of train_capture.py with --stream captures/<id>.jsonl.
All hyperparameters are validated and clamped server-side. At most
MAX_RUNNING captures run concurrently.

Usage:  python3 spectra_server.py [--port 8974] [--bind 0.0.0.0]
On SLURM, submit spectra_server.sbatch and follow the tunnel instructions
it prints into its log.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

BLOG_DIR = os.path.dirname(os.path.abspath(__file__))
CAP_DIR = os.path.join(BLOG_DIR, 'captures')
MAX_RUNNING = 3

jobs = {}          # id -> {proc, jsonl, log}
jobs_lock = threading.Lock()
job_counter = [0]

INT_PARAMS = {   # name -> (min, max, default)
    'depth': (1, 8, 3), 'width': (1, 256, 66), 'din': (1, 512, 16),
    'n': (2, 4096, 256), 'batch': (0, 4096, 0), 'steps': (1, 20000, 1500),
    'ckpt_every': (0, 20000, 1), 'ckpts': (0, 120, 0), 'seed': (0, 10 ** 9, 0),
    'slq_probes': (1, 64, 12), 'slq_k': (4, 200, 60),
    'bslq_b': (1, 16, 4), 'bslq_s': (1, 16, 3), 'bslq_k': (2, 100, 30),
    'kpm': (0, 1, 0), 'kpm_probes': (1, 64, 8), 'kpm_deg': (4, 400, 80),
    'parity_k': (1, 64, 3), 'cheby_deg': (1, 32, 4), 'cifar_size': (4, 32, 8),
}
FLOAT_PARAMS = {'lr': (1e-8, 100.0, 0.05), 'gn_damping': (1e-10, 10.0, 1e-3),
                'noise': (0.0, 10.0, 0.1)}
CHOICE_PARAMS = {'act': ({'tanh', 'relu', 'gelu', 'elu', 'linear'}, 'tanh'),
                 'opt': ({'gd', 'signgd', 'gn', 'spectral'}, 'gd'),
                 'dataset': ({'teacher', 'parity', 'chebyshev', 'cifar10'}, 'teacher')}


def clamp_params(body):
    args = []
    vals = {}
    for name, (lo, hi, d) in INT_PARAMS.items():
        v = body.get(name, d)
        try:
            v = int(v)
        except (TypeError, ValueError):
            v = d
        v = max(lo, min(hi, v))
        vals[name] = v
        args += ['--' + name.replace('_', '-'), str(v)]
    for name, (lo, hi, d) in FLOAT_PARAMS.items():
        v = body.get(name, d)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = d
        if not (v == v) or v in (float('inf'), float('-inf')):
            v = d
        v = max(lo, min(hi, v))
        args += ['--' + name.replace('_', '-'), repr(v)]
    for name, (choices, d) in CHOICE_PARAMS.items():
        v = body.get(name, d)
        if v not in choices:
            v = d
        vals[name] = v
        args += ['--' + name, v]
    return args, vals


def gpu_name():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out.splitlines()[0] if out else 'none'
    except Exception:
        return 'none'


def running_count():
    with jobs_lock:
        return sum(1 for j in jobs.values() if j['proc'].poll() is None)


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/api/ping':
            return self.send_json({'ok': True, 'gpu': gpu_name(), 'running': running_count()})
        if u.path == '/api/tail':
            q = parse_qs(u.query)
            jid = (q.get('id') or [''])[0]
            off = int((q.get('off') or ['0'])[0])
            with jobs_lock:
                job = jobs.get(jid)
            if not job or not re.fullmatch(r'run\d+_\d+', jid):
                return self.send_json({'error': 'unknown id'}, 404)
            exit_code = job['proc'].poll()
            data = ''
            try:
                with open(job['jsonl'], 'r') as fh:
                    fh.seek(max(0, off))
                    data = fh.read()
            except FileNotFoundError:
                pass
            resp = {'off': max(0, off) + len(data.encode()), 'data': data,
                    'done': exit_code is not None, 'exit': exit_code}
            if exit_code not in (None, 0):
                try:
                    with open(job['log']) as fh:
                        resp['log'] = fh.read()[-2000:]
                except FileNotFoundError:
                    pass
            return self.send_json(resp)
        if u.path == '/api/stop':
            q = parse_qs(u.query)
            jid = (q.get('id') or [''])[0]
            with jobs_lock:
                job = jobs.get(jid)
            if not job:
                return self.send_json({'error': 'unknown id'}, 404)
            if job['proc'].poll() is None:
                job['proc'].terminate()
            return self.send_json({'ok': True})
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != '/api/run':
            return self.send_json({'error': 'not found'}, 404)
        if running_count() >= MAX_RUNNING:
            return self.send_json({'error': f'{MAX_RUNNING} captures already running; '
                                            'stop one or wait'}, 429)
        try:
            n = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(n) or b'{}')
            assert isinstance(body, dict)
        except Exception:
            return self.send_json({'error': 'bad JSON body'}, 400)
        os.makedirs(CAP_DIR, exist_ok=True)
        with jobs_lock:
            job_counter[0] += 1
            jid = f'run{job_counter[0]}_{int(time.time())}'
        jsonl = os.path.join(CAP_DIR, jid + '.jsonl')
        out = os.path.join(CAP_DIR, jid + '.json')
        log = os.path.join(CAP_DIR, jid + '.log')
        cargs, vals = clamp_params(body)
        din_eff = 3 * vals['cifar_size'] ** 2 if vals['dataset'] == 'cifar10' else vals['din']
        dout = 10 if vals['dataset'] == 'cifar10' else 1
        dims = [din_eff] + [vals['width']] * vals['depth'] + [dout]
        p_est = sum(dims[i + 1] * dims[i] + dims[i + 1] for i in range(len(dims) - 1))
        if p_est > 25000:
            return self.send_json({'error': f'p = {p_est} parameters is too large for dense '
                                            f'p×p spectra (limit 25000); shrink width/depth/d_in'}, 400)
        cmd = [sys.executable, os.path.join(BLOG_DIR, 'train_capture.py'),
               '--stream', jsonl, '--out', out] + cargs
        with open(log, 'w') as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=BLOG_DIR)
        with jobs_lock:
            jobs[jid] = {'proc': proc, 'jsonl': jsonl, 'log': log}
        return self.send_json({'id': jid})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8974)
    ap.add_argument('--bind', default='0.0.0.0')
    args = ap.parse_args()
    os.makedirs(CAP_DIR, exist_ok=True)
    srv = ThreadingHTTPServer((args.bind, args.port),
                              partial(Handler, directory=BLOG_DIR))
    host = subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip()
    print(f'spectra server on http://{host}:{args.port}/hessian-spectral-estimation.html '
          f'(gpu: {gpu_name()})', flush=True)
    print(f'tunnel from your laptop:  ssh -L {args.port}:{host}:{args.port} <cluster-login> '
          f' then open http://127.0.0.1:{args.port}/hessian-spectral-estimation.html', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
