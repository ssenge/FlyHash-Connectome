"""Separate FlyHash++ study: python -m flypath.plusplus --help.

Algorithms/provenance and limitations: experiments/FLYHASHPP.md.
No paper results are overwritten. All fitted quantities use training rows only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t as student_t

from . import flyhash as fh
from . import replication as rp
from .config import load


@dataclass(frozen=True)
class Spec:
    method: str
    m: int
    s: int
    k: int

    @property
    def key(self):
        return f'{self.method}:m{self.m}:s{self.s}:k{self.k}'


def fixed_top(y, k, priority):
    """Exact lexicographic ties; a fixed cell priority shared across inputs."""
    if not 0 < k <= y.shape[1]:
        raise ValueError('k must be in 1..m')
    return np.lexsort((np.broadcast_to(priority, y.shape), -y), axis=1)[:, :k]


def overlap_energy(w):
    """Number of 4-cycles; penalizes repeated pairs of sampled input features."""
    c = w.astype(np.int64) @ w.astype(np.int64).T
    v = c[np.triu_indices(len(c), 1)]
    return int(np.sum(v * (v - 1) // 2))


def diversify(w, steps, seed):
    """Greedy degree-preserving 2x2 switches that strictly reduce 4-cycles."""
    rng = np.random.default_rng(seed)
    w = w.copy().astype(np.int64)
    c = w @ w.T
    accepted = 0
    for _ in range(steps):
        i, j = rng.choice(w.shape[1], 2, replace=False)
        a0 = np.flatnonzero((w[:, i] == 1) & (w[:, j] == 0))
        b0 = np.flatnonzero((w[:, j] == 1) & (w[:, i] == 0))
        if not len(a0) or not len(b0):
            continue
        a, b = rng.choice(a0), rng.choice(b0)
        # Only overlaps of feature a/b with other features change.
        delta = w[:, j] - w[:, i]
        delta[[a, b]] = 0
        ca, cb = c[a] + delta, c[b] - delta
        choose = lambda v: v * (v - 1) // 2
        ids = np.arange(len(c)); ids = ids[(ids != a) & (ids != b)]
        change = (choose(ca[ids]) + choose(cb[ids]) - choose(c[a, ids]) - choose(c[b, ids])).sum()
        if change < 0:
            w[a, i], w[b, i], w[a, j], w[b, j] = 0, 1, 1, 0
            c[a, ids] = ca[ids]; c[ids, a] = ca[ids]
            c[b, ids] = cb[ids]; c[ids, b] = cb[ids]
            accepted += 1
    return w.astype(np.float32), accepted


def bio_update(w, batch, p=2., delta=.4, rank=2, mask=None):
    """BioHash Eq. 1/2 minibatch direction, weights stored as d x m.

    Effective current = x @ sign(W)|W|^(p-1). Winner learns, rank-r
    neuron anti-learns. Fixed-support masking is our ablation, not BioHash.
    """
    current = batch @ (np.sign(w) * np.abs(w) ** (p - 1))
    order = np.argsort(-current, axis=1, kind='stable')
    g = np.zeros_like(current)
    g[np.arange(len(batch)), order[:, 0]] = 1
    g[np.arange(len(batch)), order[:, rank - 1]] = -delta
    dw = batch.T @ g - w * np.sum(g * current, axis=0)
    if mask is not None:
        dw *= mask
    # Explicit Euler direction with global max-step normalization.
    return dw / max(float(np.abs(dw).max()), 1e-12)


def train_bio(train, w, cfg, seed, mask=None):
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(cfg['epochs']):
        lr = cfg['learning_rate'] * (1 - epoch / cfg['epochs'])
        order = rng.permutation(len(train))
        for start in range(0, len(train), cfg['batch_size']):
            batch = train[order[start:start + cfg['batch_size']]]
            w += lr * bio_update(w, batch, cfg['bio_p'], cfg['bio_delta'], cfg['bio_rank'], mask)
        losses.append(float(np.mean(np.linalg.norm(w, ord=cfg['bio_p'], axis=0))))
    if not np.isfinite(w).all():
        raise ValueError('non-finite BioHash weights')
    return w, losses


METHODS = ('fly', 'balanced', 'diverse', 'calibrated', 'zscore',
           'balanced_calibrated', 'diverse_calibrated', 'rank',
           'adaptive_relative', 'adaptive_external', 'scale_tag',
           'two_global', 'two_concat', 'two_or', 'densefly', 'densefly_probe',
           'biohash', 'biohash_untrained', 'sparse_bio', 'sparse_untrained',
           'gaussian_real', 'gaussian_sign', 'gaussian_ops', 'gaussian_storage')


class Model:
    def __init__(self, spec, cfg, seed):
        self.spec, self.cfg, self.seed = spec, cfg, seed
        self.extra = {}

    def fit(self, train, raw_train, base=None):
        spec, cfg = self.spec, self.cfg
        d, m, s, k = train.shape[1], spec.m, spec.s, spec.k
        rng = np.random.default_rng(self.seed + 111)
        self.kind = spec.method
        self.table_split = None
        self.weighted = False
        if base is not None:
            self.w = base.copy().astype(np.float32)
            m = self.w.shape[1]
        elif self.kind.startswith('gaussian'):
            bits = k
            if self.kind == 'gaussian_ops': bits = max(1, m * s // (2 * d))
            if self.kind == 'gaussian_storage': bits = k * math.ceil(math.log2(m))
            self.w = rng.normal(size=(d, bits)).astype(np.float32)
            self.weighted = True
        else:
            self.w = rp.fly_matrix(d, m, rng, sampled=s)
            if self.kind in ('balanced', 'diverse', 'balanced_calibrated', 'diverse_calibrated',
                             'rank', 'adaptive_relative', 'adaptive_external', 'scale_tag',
                             'two_concat', 'two_or', 'sparse_bio', 'sparse_untrained'):
                p = fh.Projection('base', self.w, [str(i) for i in range(d)])
                self.w = fh.margin_control(p, seed=self.seed + 212, equal_in=True, equal_out=True,
                                           sweeps=10).matrix.astype(np.float32)
            if self.kind in ('two_global', 'two_concat', 'two_or'):
                # Truly independent tables, identical for these three ablations.
                self.w = np.concatenate([
                    rp.fly_matrix(d, m // 2, np.random.default_rng(self.seed + 901), sampled=s),
                    rp.fly_matrix(d, m - m // 2, np.random.default_rng(self.seed + 902), sampled=s)
                ], axis=1)
            if self.kind in ('diverse', 'diverse_calibrated'):
                before = overlap_energy(self.w)
                self.w, accepted = diversify(self.w, cfg['diversity_steps'], self.seed + 313)
                self.extra.update(overlap_before=before, overlap_after=overlap_energy(self.w),
                                  accepted_swaps=accepted)
        if self.kind.startswith('measured_'):
            self.kind = self.kind.removeprefix('measured_')
        if self.kind.startswith('null_'):
            self.kind = self.kind.removeprefix('null_')
        if self.kind in ('biohash', 'biohash_untrained', 'sparse_bio', 'sparse_untrained'):
            mask = (self.w != 0).astype(np.float32) if self.kind.startswith('sparse') else None
            self.w = rng.normal(size=self.w.shape).astype(np.float32)
            if mask is not None: self.w *= mask
            if self.kind in ('biohash', 'sparse_bio'):
                self.w, trace = train_bio(train, self.w, cfg, self.seed + 414, mask)
                self.extra['epoch_mean_weight_p_norm'] = trace
            self.w = np.sign(self.w) * np.abs(self.w) ** (cfg['bio_p'] - 1)
            self.weighted = True
        self.priority = rng.permutation(self.w.shape[1])
        self.sorted_train = None
        if ('calibrated' in self.kind or self.kind == 'zscore' or self.kind.startswith('adaptive')):
            y = train @ self.w
        if self.kind == 'zscore':
            self.mu, self.sd = y.mean(0), np.maximum(y.std(0), 1e-6)
        if 'calibrated' in self.kind:
            self.sorted_train = np.sort(y, axis=0)
        if self.kind in ('two_concat', 'two_or'):
            self.table_split = self.extra.get('table_split', self.w.shape[1] // 2)
        if self.kind == 'adaptive_relative':
            z = (y - y.mean(1, keepdims=True)) / np.maximum(y.std(1, keepdims=True), 1e-6)
            self.threshold = float(np.quantile(z, 1 - k / y.shape[1]))
        if self.kind == 'adaptive_external':
            self.threshold = float(np.quantile(y, 1 - k / y.shape[1]))
        if self.kind == 'scale_tag':
            scale = np.log1p(np.linalg.norm(raw_train, axis=1))
            self.scale_lo, self.scale_hi = np.quantile(scale, [.01, .99])
        return self

    def transform(self, x, raw):
        y = x @ self.w
        kind = self.kind
        if kind == 'gaussian_real' or kind == 'gaussian_ops':
            return y, None
        if kind.startswith('gaussian') or kind.startswith('densefly'):
            z = (y >= 0).astype(np.float32)
            pseudo = None
            if kind == 'densefly_probe':
                # Contiguous activation blocks: Algorithm 1 pseudo-hash.
                pseudo = np.column_stack([y[:, ids].sum(1) > 0
                                          for ids in np.array_split(np.arange(y.shape[1]), min(16, y.shape[1]))])
            return z, pseudo
        if self.sorted_train is not None:
            y = np.column_stack([np.searchsorted(self.sorted_train[:, i], y[:, i], side='right')
                                 / len(self.sorted_train) for i in range(y.shape[1])])
        elif kind == 'zscore':
            y = (y - self.mu) / self.sd
        if kind == 'adaptive_relative':
            y = (y - y.mean(1, keepdims=True)) / np.maximum(y.std(1, keepdims=True), 1e-6)
        if kind.startswith('adaptive'):
            return (y > self.threshold).astype(np.float32), None
        z = np.zeros_like(y, dtype=np.float32)
        if self.table_split is not None:
            b = self.table_split
            for lo, hi, k in ((0, b, self.spec.k // 2), (b, y.shape[1], self.spec.k - self.spec.k // 2)):
                ix = fixed_top(y[:, lo:hi], k, self.priority[lo:hi]) + lo
                z[np.arange(len(z))[:, None], ix] = 1
        else:
            ix = fixed_top(y, self.spec.k, self.priority)
            z[np.arange(len(z))[:, None], ix] = (
                np.arange(self.spec.k, 0, -1) if kind == 'rank' else 1)
        if kind == 'scale_tag':
            scale = np.log1p(np.linalg.norm(raw, axis=1))
            q = np.rint(255 * np.clip((scale - self.scale_lo) / max(self.scale_hi - self.scale_lo, 1e-6), 0, 1))
            z = np.column_stack([z, q / 255])
        return z, None

    def resources(self, gallery_code):
        d, m = self.w.shape
        k = self.spec.k
        nnz = int(np.count_nonzero(self.w))
        binary = not self.weighted
        if self.kind in ('gaussian_real', 'gaussian_ops'):
            bits = 32 * m
        elif self.kind.startswith(('gaussian', 'densefly', 'adaptive')):
            bits = m
        elif self.kind == 'rank':
            bits = k * math.ceil(math.log2(m)) # ordered list of indices
        else:
            bits = k * math.ceil(math.log2(m)) # explicit fixed-width active indices
            if self.kind == 'scale_tag': bits += 8
        info = dict(projection_additions=nnz, projection_multiplications=0 if binary else nnz,
                    projection_operations=nnz if binary else 2 * nnz,
                    code_bits_fixed_width=bits, code_working_bytes_per_item=gallery_code.nbytes / len(gallery_code),
                    weight_working_bytes=self.w.nbytes,
                    fitted_state_bytes=sum(v.nbytes for v in self.__dict__.values() if isinstance(v, np.ndarray)),
                    mean_nonzero_code_entries=float(np.count_nonzero(gallery_code, axis=1).mean()),
                    variable_tag_length=self.kind.startswith('adaptive'),
                    storage_accounting='fixed-width representation estimate; execution uses float32 arrays',
                    extra_encoding_cost=('empirical CDF search per cell' if self.sorted_train is not None else
                                         'log-norm and 8-bit scale' if self.kind == 'scale_tag' else
                                         'threshold normalization' if self.kind.startswith('adaptive') else
                                         'rank ordering' if self.kind == 'rank' else 'threshold or top-k'))
        if self.kind == 'densefly_probe': info['code_bits_fixed_width'] += min(16, m)
        return info


def logchoose(m, k):
    return (math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)) / math.log(2)


def rank_codes(q, g, model, top, qp=None, gp=None):
    """All methods scan codes; probe scans occupied pseudo-bins, not points.

    OR is candidate-union of two full code scans, not an optimized hash index.
    No raw-feature reranking is used. Missing candidates remain -1 (misses).
    """
    rng = np.random.default_rng(model.seed + 515)
    priority = rng.permutation(len(g))
    metric = 'cityblock' if model.kind == 'rank' else 'sqeuclidean'
    order = lambda row, ids: ids[np.lexsort((priority[ids], row[ids]))]
    if model.kind not in ('two_or', 'densefly_probe'):
        dist = cdist(q, g, metric=metric)
        ids = np.arange(len(g))
        return np.array([order(row, ids)[:top] for row in dist]), np.full(len(q), len(g))
    candidates = []
    cap = min(model.cfg['candidate_budget'], len(g))
    if model.kind == 'two_or':
        b = model.table_split
        dl = cdist(q[:, :b], g[:, :b], 'sqeuclidean')
        dr = cdist(q[:, b:], g[:, b:], 'sqeuclidean')
        ids = np.arange(len(g))
        for left, right in zip(dl, dr):
            candidates.append(np.union1d(order(left, ids)[:cap // 2], order(right, ids)[:cap - cap // 2]))
    else:
        bins, assignment = np.unique(gp, axis=0, return_inverse=True)
        members = [np.flatnonzero(assignment == i) for i in range(len(bins))]
        for query in qp:
            bd = np.count_nonzero(bins != query, axis=1)
            selected = []
            for bi in np.argsort(bd, kind='stable'):
                selected.extend(members[bi].tolist())
                if len(selected) >= cap: break
            # Deterministic within-bin truncation; cap is an explicit adaptation.
            candidates.append(np.array(selected[:cap], dtype=int))
    pred = np.full((len(q), top), -1, int)
    for i, ids in enumerate(candidates):
        distances = cdist(q[i:i + 1], g[ids], metric=metric)[0]
        take = ids[np.lexsort((priority[ids], distances))[:top]]
        pred[i, :len(take)] = take
    return pred, np.array([len(x) for x in candidates])


def scores(pred, truth):
    hit = (pred[:, :, None] == truth[:, None, :]).any(2) & (pred >= 0)
    precision = hit.cumsum(1) / np.arange(1, pred.shape[1] + 1)
    return ((precision * hit).sum(1) / truth.shape[1], hit.sum(1) / truth.shape[1])


def split_data(x, cfg, seed):
    counts = [cfg[k] for k in ('train', 'validation', 'queries', 'gallery')]
    if sum(counts) > len(x): raise ValueError('requested split exceeds available data')
    ids = np.random.default_rng(seed + 616).permutation(len(x))[:sum(counts)]
    blocks = np.split(ids, np.cumsum(counts)[:-1])
    return [x[i].astype(np.float32) for i in blocks], [i.tolist() for i in blocks]


def preprocess(blocks, dims):
    start = time.perf_counter()
    train = blocks[0]
    if dims and dims < train.shape[1]:
        mu = train.mean(0)
        _, _, vt = np.linalg.svd(train - mu, full_matrices=False)
        if dims > len(vt): raise ValueError('PCA dimension exceeds training rank bound')
        blocks = [(x - mu) @ vt[:dims].T for x in blocks]
    return [rp.centre(x).astype(np.float32) for x in blocks], time.perf_counter() - start


def interval(values):
    v = np.asarray(values, float)
    mean = float(v.mean())
    if len(v) < 2: return {'mean': mean, 'ci95': None, 'n': len(v)}
    half = float(student_t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return {'mean': mean, 'ci95': [mean - half, mean + half], 'n': len(v)}


def frontier(rows, quality='validation_ap'):
    """Validation frontier, jointly constrained by arithmetic and code bits."""
    return [r['key'] for r in rows if not any(
        s['projection_operations'] <= r['projection_operations'] and
        s['code_bits_fixed_width'] <= r['code_bits_fixed_width'] and
        s[quality] >= r[quality] and
        (s['projection_operations'] < r['projection_operations'] or
         s['code_bits_fixed_width'] < r['code_bits_fixed_width'] or s[quality] > r[quality])
        for s in rows if s is not r)]


def evaluate(model, blocks, raw, truth, cfg):
    start = time.perf_counter()
    gallery, gp = model.transform(blocks[3], raw[3])
    gallery_seconds = time.perf_counter() - start
    result = dict(model.resources(gallery), gallery_encode_seconds=gallery_seconds)
    result['gallery_code_working_bytes'] = gallery.nbytes
    result['reranker_gallery_feature_bytes'] = raw[3].nbytes
    for which, idx in (('validation', 1), ('test', 2)):
        times, encode = [], []
        for _ in range(cfg['timing_repeats']):
            start = time.perf_counter()
            q, qp = model.transform(blocks[idx], raw[idx])
            encode.append(time.perf_counter() - start)
            start = time.perf_counter()
            pred, counts = rank_codes(q, gallery, model, cfg['top'], qp, gp)
            times.append(time.perf_counter() - start)
        ap, rec = scores(pred, truth[which])
        # Equal maximum candidate budget for all models, followed by the same
        # ORIGINAL-feature reranker. Report separately from hash-only AP.
        start = time.perf_counter()
        candidates, _ = rank_codes(q, gallery, model, min(cfg['candidate_budget'], len(gallery)), qp, gp)
        reranked = np.full((len(q), cfg['top']), -1, int)
        candidate_recall = []
        for qi, row in enumerate(candidates):
            ids = row[row >= 0]
            candidate_recall.append(np.isin(truth[which][qi], ids).mean())
            distances = cdist(raw[idx][qi:qi + 1], raw[3][ids], 'sqeuclidean')[0]
            take = ids[np.lexsort((ids, distances))[:cfg['top']]]
            reranked[qi, :len(take)] = take
        rerank_seconds = time.perf_counter() - start
        rap, rrec = scores(reranked, truth[which])
        result.update({f'{which}_ap': float(ap.mean()), f'{which}_recall': float(rec.mean()),
                       f'{which}_query_ap': ap.tolist(), f'{which}_query_recall': rec.tolist(),
                       f'{which}_search_seconds': float(np.median(times)),
                       f'{which}_query_encode_seconds': float(np.median(encode)),
                       f'{which}_candidate_count': float(counts.mean()),
                       f'{which}_candidate_recall': float(np.mean(candidate_recall)),
                       f'{which}_reranked_ap': float(rap.mean()),
                       f'{which}_reranked_recall': float(rrec.mean()),
                       f'{which}_candidate_and_rerank_seconds': rerank_seconds})
    return result


def validate(cfg):
    for k in ('train', 'validation', 'queries', 'gallery', 'top', 'batch_size', 'timing_repeats'):
        if cfg[k] <= 0: raise ValueError(f'{k} must be positive')
    if cfg['top'] > cfg['gallery']: raise ValueError('top exceeds gallery size')
    if cfg['bio_p'] < 2 or cfg['bio_rank'] < 2: raise ValueError('BioHash requires p>=2, rank>=2')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']): raise ValueError('unique seeds required')
    if min(cfg['winners']) < 2: raise ValueError('two-table study requires k>=2')


def summarize(out):
    groups = {}
    for r in out['rows']: groups.setdefault((r['dataset'], r['key']), []).append(r)
    out['summary'] = [{ 'dataset': d, 'key': k,
                       **{metric: interval([r[metric] for r in rows]) for metric in
                          ('test_ap', 'test_recall', 'validation_ap', 'test_reranked_ap',
                           'test_candidate_recall', 'fit_seconds', 'test_search_seconds',
                           'test_query_encode_seconds')},
                       'projection_operations': rows[0]['projection_operations'],
                       'code_bits_fixed_width': rows[0]['code_bits_fixed_width']}
                      for (d, k), rows in groups.items()]
    pairs = [('balanced', 'fly'), ('diverse', 'balanced'), ('calibrated', 'fly'),
             ('zscore', 'fly'), ('balanced_calibrated', 'balanced'),
             ('diverse_calibrated', 'diverse'), ('rank', 'balanced'),
             ('adaptive_relative', 'balanced'), ('adaptive_external', 'balanced'),
             ('scale_tag', 'balanced'), ('two_concat', 'two_global'), ('two_or', 'two_concat'),
             ('densefly', 'fly'), ('densefly_probe', 'densefly'),
             ('biohash', 'biohash_untrained'), ('sparse_bio', 'sparse_untrained'),
             ('measured_two_concat', 'measured_fly'), ('measured_two_or', 'measured_two_concat'),
             ('measured_two_concat', 'null_two_concat'), ('measured_two_or', 'null_two_or'),
             ('measured_sparse_bio', 'measured_sparse_untrained'),
             ('null_sparse_bio', 'null_sparse_untrained')]
    out['paired_ablations'] = []
    lookup = {(r['dataset'], r['seed'], r['method'], r['m'], r['s'], r['k']): r for r in out['rows']}
    for treatment, control in pairs:
        contrasts = {}
        for r in out['rows']:
            if r['method'] != treatment: continue
            base = lookup.get((r['dataset'], r['seed'], control, r['m'], r['s'], r['k']))
            if base:
                contrasts.setdefault((r['dataset'], r['m'], r['s'], r['k']), []).append(
                    (r['test_ap'] - base['test_ap'], r['test_reranked_ap'] - base['test_reranked_ap']))
        for (d, m, s, k), vals in contrasts.items():
            out['paired_ablations'].append(dict(dataset=d, treatment=treatment, control=control,
                                                m=m, s=s, k=k, ap_difference=interval([v[0] for v in vals]),
                                                reranked_ap_difference=interval([v[1] for v in vals])))
    out['validation_frontiers'] = {}
    for dataset in out['config']['datasets']:
        rows = [dict(r, validation_ap=r['validation_ap']['mean']) for r in out['summary'] if r['dataset'] == dataset]
        out['validation_frontiers'][dataset] = frontier(rows) if rows else []


def write(out, path):
    summarize(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    temp.replace(path)
    lines = ['# FlyHash++ exploratory study', '',
             'AP cutoff, splits and all hyperparameters are recorded in the JSON. Intervals are paired-seed t intervals, exploratory and unadjusted.',
             'Timing uses full code scans (including OR); this is not an optimized ANN throughput benchmark.',
             'Frontiers are selected on validation quality and use projection arithmetic plus fixed-width code storage.', '',
             '| Dataset | Method/config | AP (mean) | Recall | Projection ops | Code bits |',
             '|---|---|---:|---:|---:|---:|']
    for r in out['summary']:
        lines.append(f"| {r['dataset']} | {r['key']} | {r['test_ap']['mean']:.4f} | {r['test_recall']['mean']:.4f} | {r['projection_operations']} | {r['code_bits_fixed_width']} |")
    lines += ['', '## Paired ablations', '', '| Dataset/config | Treatment − control | Δ AP | 95% interval |', '|---|---|---:|---|']
    for r in out['paired_ablations']:
        v = r['ap_difference']
        lines.append(f"| {r['dataset']}/m{r['m']}/s{r['s']}/k{r['k']} | {r['treatment']} − {r['control']} | {v['mean']:+.4f} | {v['ci95']} |")
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n')


def run(cfg, path, config_path=None):
    validate(cfg)
    env = dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
               processor=platform.processor(), threads={k: os.environ.get(k) for k in
                 ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = dict(schema=1, status='running', config=cfg, environment=env, git_commit=git,
               implementation_sha256=source_hash, config_path=str(config_path), rows=[], splits=[],
               implementation_sources_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                              for p in ('flypath/plusplus.py', 'flypath/replication.py',
                                                        'flypath/flyhash.py', 'flypath/connectomes.py')},
               provenance={'biohash': 'https://proceedings.mlr.press/v119/ryali20a/ryali20a.pdf',
                           'densefly': 'https://arxiv.org/html/1812.01844'})
    project = load()
    methods = cfg.get('methods', METHODS)
    unknown = set(methods) - set(METHODS)
    if unknown: raise ValueError(f'unknown methods: {unknown}')
    for name in cfg['datasets']:
        if name == 'synthetic':
            rng = np.random.default_rng(999)
            x = (rng.normal(size=(sum(cfg[k] for k in ('train', 'validation', 'queries', 'gallery')), 8))
                 @ rng.normal(size=(8, 32)) + rng.normal(size=(sum(cfg[k] for k in ('train', 'validation', 'queries', 'gallery')), 32)) * .1)
        else:
            x = rp.load_benchmark(project, name)
        for seed in cfg['seeds']:
            raw, ids = split_data(x, cfg, seed)
            blocks, prep_seconds = preprocess(raw, cfg['dimensions'])
            # All methods share truth on ORIGINAL features, including with PCA.
            truth = {key: np.argsort(cdist(raw[i], raw[3], 'sqeuclidean'), axis=1, kind='stable')[:, :cfg['top']]
                     for key, i in (('validation', 1), ('test', 2))}
            out['splits'].append(dict(dataset=name, seed=seed, ids=ids, preprocessing_seconds=prep_seconds))
            d = blocks[0].shape[1]
            specs = [Spec(method, d * expansion, s, k) for expansion in cfg['expansions']
                     for s in cfg['fanins'] for k in cfg['winners'] for method in methods]
            for spec in specs:
                if not 0 < spec.s <= d or spec.k > spec.m or cfg['bio_rank'] > spec.m:
                    raise ValueError(f'invalid configuration {spec}')
                start = time.perf_counter()
                model = Model(spec, cfg, seed).fit(blocks[0], raw[0])
                fit_seconds = time.perf_counter() - start
                metrics = evaluate(model, blocks, raw, truth, cfg)
                out['rows'].append(dict(dataset=name, seed=seed, key=spec.key, **asdict(spec),
                                       fit_seconds=fit_seconds, **metrics, diagnostics=model.extra))
            if cfg.get('bilateral'):
                bilateral(out, name, seed, raw, truth, cfg, project)
            write(out, path)
            print(f'{name} seed {seed}: {len(out["rows"])} rows saved to {path}', flush=True)
    out['status'] = 'complete'
    write(out, path)
    return out


def bilateral(out, name, seed, raw, truth, cfg, project):
    from .connectomes import projection
    a, b = projection(project, 'malecns', 'R'), projection(project, 'malecns', 'L')
    common = sorted(set(a.glomeruli) & set(b.glomeruli))
    if name == 'odours':
        from .experiments import context
        common = context(project).odours.glomeruli
    a, b = fh.align(a, common), fh.align(b, common)
    d = len(common)
    if name == 'odours':
        blocks, prep = preprocess(raw, 0)
    else:
        if raw[0].shape[1] < d: raise ValueError('bilateral PCA needs at least as many features as glomeruli')
        blocks, prep = preprocess(raw, d)
    real = np.concatenate([a.binary, b.binary], axis=1).astype(np.float32)
    null = np.concatenate([fh.curveball(a, seed=seed + 717).binary,
                           fh.curveball(b, seed=seed + 818).binary], axis=1).astype(np.float32)
    for k in cfg['winners']:
        for family, w in (('measured', real), ('null', null)):
            for method in ('fly', 'two_concat', 'two_or', 'sparse_bio', 'sparse_untrained'):
                spec = Spec(f'{family}_{method}', w.shape[1], 0, k)
                model = Model(spec, cfg, seed)
                model.extra['table_split'] = a.matrix.shape[1]
                start = time.perf_counter()
                model.fit(blocks[0], raw[0], base=w)
                elapsed = time.perf_counter() - start
                out['rows'].append(dict(dataset=name, seed=seed, key=spec.key, **asdict(spec),
                    fit_seconds=elapsed, **evaluate(model, blocks, raw, truth, cfg),
                    diagnostics=dict(model.extra, common_glomeruli=common, preprocessing_seconds=prep)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/flyhashpp_pilot.json')
    parser.add_argument('--output', default='results/flyhashpp/pilot.json')
    parser.add_argument('--overwrite', action='store_true', help='replace this study output explicitly')
    args = parser.parse_args(argv)
    path = Path(args.output)
    if path.exists() and not args.overwrite: parser.error(f'{path} exists; choose another output or --overwrite')
    cfg = json.loads(Path(args.config).read_text())
    run(cfg, path, args.config)


if __name__ == '__main__':
    main()
