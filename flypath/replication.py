"""The 2017 benchmark protocol, first with random matrices, then on the connectome.

Dasgupta, Stevens and Navlakha (Science 358:793, 2017), main text:

  data      10,000 vectors each from SIFT (d=128), GloVe (d=300), MNIST (d=784)
  queries   1,000 random inputs; for each, the true top 2% (200) neighbours by
            Euclidean distance between feature vectors
  predicted top 2% by Euclidean distance between hashes
  score     mean average precision, averaged over 50 trials in which the
            projection matrices and the queries change
  input     mean-centred ("centers the mean"; step 1 of the circuit)
  fly       sparse binary projection to m Kenyon cells, winner-take-all top k;
            m = 20k equates the operation count with LSH (their Fig. 2B),
            m = 10d gives the headline comparison (their Fig. 3, e.g. MNIST
            k=4: LSH 16.0%, fly 44.8%)
  LSH       k Gaussian random projections

Not stated in the main text, and taken from the FlyLSH reference code
(Sharma and Navlakha): each Kenyon cell samples 10% of the input dimensions,
and the fly tag is the binary top-k pattern. The LSH hash is the vector of k
projection values; the binary (sign) variant, their Fig. S3, is reported too.
The paper's "mean average precision" is undefined. We report AP@200 (all true
neighbours in the denominator) and recall@200, the list-overlap score of the
FlyLSH code, and precision averaged over retrieved neighbours only; the last
reproduces the published LSH baseline but not the fly scores, and none of them
reproduces the published absolute values.

`replicate` runs that protocol, on their three datasets and on a fourth the
2017 paper did not use: mixtures of DoOR odorant profiles, the input the
circuit receives. `connectome` keeps the protocol and the data but replaces
the random fly matrix with the measured one, compared with its curveball
nulls, with the 2017 random construction at the same size, and with LSH.
"""

from __future__ import annotations

import time
import zipfile

import numpy as np
import scipy.sparse as sp

from . import flyhash as fh
from . import stats
from .config import Config

N_DATA = 10_000
N_QUERIES = 1_000
TOP = 0.02
TRIALS = 50
HASH_LENGTHS = (2, 4, 8, 16, 32)
SAMPLING = 0.10
DATASETS = ("sift", "glove", "mnist", "odours")
SOURCES = {
    "mnist.npz": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
    "siftsmall.tar.gz": "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
    "glove.6B.zip": "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
}


# ---------------------------------------------------------------- data

def load_benchmark(cfg: Config, name: str, n: int = N_DATA, seed: int = 0,
                   odours: np.ndarray | None = None) -> np.ndarray:
    """A fixed subset of `n` vectors (float64, raw feature values),
    downloaded on first use."""
    d = cfg.raw_dir / "bench"
    for fname, url in SOURCES.items():
        fh._fetch(url, d / fname)
    if not (d / "siftsmall").exists():
        import tarfile
        with tarfile.open(d / "siftsmall.tar.gz") as t:
            t.extractall(d, filter="data")
    rng = np.random.default_rng(seed)
    if name == "mnist":
        x = np.load(d / "mnist.npz")["x_train"].reshape(-1, 784).astype(float)
        return x[rng.choice(len(x), n, replace=False)]
    if name == "sift":
        # siftsmall_base is exactly 10,000 SIFT descriptors
        raw = np.fromfile(d / "siftsmall" / "siftsmall_base.fvecs", dtype=np.float32)
        x = raw.reshape(-1, 129)[:, 1:].astype(float)
        return x[rng.choice(len(x), n, replace=False)] if len(x) > n else x
    if name == "glove":
        with zipfile.ZipFile(d / "glove.6B.zip") as z:
            lines = z.read("glove.6B.300d.txt").decode("utf8").splitlines()
        pick = np.sort(rng.choice(len(lines), n, replace=False))
        return np.array([[float(v) for v in lines[i].split(" ")[1:]] for i in pick])
    if name == "odours":
        # the fly's own input: synthetic mixtures of DoOR profiles over the
        # 35 well-measured glomeruli, as in the odour benchmark
        from .experiments import context
        return fh.mixtures(context(cfg).odours.x if odours is None else odours, n, seed=seed)
    raise ValueError(f"unknown benchmark {name!r}")


def centre(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=1, keepdims=True)


def pca(x: np.ndarray, dims: int) -> np.ndarray:
    z = x - x.mean(0)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    return u[:, :dims] * s[:dims]


# ---------------------------------------------------------------- hashes

def fly_matrix(d: int, m: int, rng, sampled: int | None = None) -> np.ndarray:
    """Binary d x m; every Kenyon cell samples `sampled` distinct inputs."""
    s = sampled or max(1, int(round(SAMPLING * d)))
    idx = np.argsort(rng.random((d, m)), axis=0)[:s]
    w = np.zeros((d, m), np.float32)
    w[idx, np.arange(m)] = 1.0
    return w


def winners(y: np.ndarray, k: int, prio: np.ndarray | None = None) -> sp.csr_matrix:
    """Binary top-k tag per row, as a sparse matrix.

    Exact ties in drive (cells with identical inputs) are resolved by
    `argpartition` order unless `prio`, a fixed random priority over cells,
    is given; then a tied cell with higher priority wins. The jitter is far
    below any non-tied gap in drive (checked).
    """
    if prio is not None:
        gaps = np.diff(np.unique(y[: min(len(y), 200)]))
        eps = 0.25 * gaps[gaps > 0].min() if (gaps > 0).any() else 1e-6
        y = y + eps * prio[None, :]
    top = np.argpartition(-y, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(len(y)), k)
    return sp.csr_matrix((np.ones(rows.size, np.float32), (rows, top.ravel())), shape=y.shape)


# ---------------------------------------------------------------- scoring

def truth(x: np.ndarray, queries: np.ndarray, top: int) -> np.ndarray:
    return _nearest(_sqdist(x[queries], x), queries, top, None)


def _sqdist(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (q * q).sum(1)[:, None] + (x * x).sum(1)[None, :] - 2 * q @ x.T


def _nearest(dist: np.ndarray, queries: np.ndarray, top: int, rng) -> np.ndarray:
    """Indices of the `top` smallest distances per query, the query excluded.

    With `rng`, distances are integers (Hamming) and exact ties are broken at
    random by adding a uniform jitter below the smallest gap of 1.
    """
    dist = np.array(dist, dtype=np.float64)
    if rng is not None:
        dist += 0.5 * rng.random(dist.shape)
    dist[np.arange(len(queries)), queries] = np.inf
    part = np.argpartition(dist, top - 1, axis=1)[:, :top]
    return np.take_along_axis(part, np.argsort(np.take_along_axis(dist, part, 1), axis=1), 1)


def average_precision(pred: np.ndarray, true: np.ndarray) -> float:
    """AP@n, n = |true| (200 here): for each query, the sum of precision@i over
    the ranks i of the predicted list at which a true neighbour appears,
    divided by n. True neighbours that are never retrieved count as misses, so
    one correct item at rank 1 out of 200 scores 1/200, not 1."""
    hit = _hits(pred, true)
    prec = np.cumsum(hit, axis=1) / np.arange(1, hit.shape[1] + 1)
    return float(((prec * hit).sum(1) / true.shape[1]).mean())


def ap_retrieved(pred: np.ndarray, true: np.ndarray) -> float:
    """The same sum divided by the number of true neighbours actually
    retrieved, a convention found in parts of the hashing literature. It
    ignores misses (one hit at rank 1 scores 1.0) and is reported only as the
    quantity revision 3 used by mistake as AP; it is not average precision."""
    hit = _hits(pred, true)
    prec = np.cumsum(hit, axis=1) / np.arange(1, hit.shape[1] + 1)
    return float(((prec * hit).sum(1) / np.maximum(hit.sum(1), 1)).mean())


def recall(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of the true neighbours among the predicted list (recall@n)."""
    return float(_hits(pred, true).mean())


def list_overlap(pred: np.ndarray, true: np.ndarray) -> float:
    """The variant in the FlyLSH reference code: mean over depths i of
    |pred[:i] & true[:i]| / i. An item is in both top-i lists iff the larger of
    its two ranks is < i, so the overlaps are a cumulative count of that."""
    q, n = true.shape
    rank_true = np.full((q, int(max(pred.max(), true.max())) + 1), n, np.int64)
    np.put_along_axis(rank_true, true, np.arange(n)[None, :].repeat(q, 0), axis=1)
    both = np.maximum(np.arange(n)[None, :], np.take_along_axis(rank_true, pred, 1))
    counts = np.zeros((q, n + 1), np.int64)
    np.add.at(counts, (np.arange(q)[:, None].repeat(n, 1), both), 1)
    return float((np.cumsum(counts[:, :n], axis=1) / np.arange(1, n + 1)).mean())


def _ap_reference(pred, true) -> float:
    """Direct transcription of the FlyLSH list-overlap score, for the tests."""
    return float(np.mean([np.mean([len(set(p[:i]) & set(t[:i])) / i
                                   for i in range(1, len(t) + 1)])
                          for p, t in zip(pred.tolist(), true.tolist())]))


def _hits(pred, true):
    want = np.zeros((len(true), int(max(pred.max(), true.max())) + 1), bool)
    np.put_along_axis(want, true, True, axis=1)
    return np.take_along_axis(want, pred, axis=1)


METRICS = ("ap", "recall", "overlap", "ap_retrieved")


def _score(pred, true) -> np.ndarray:
    return np.array([average_precision(pred, true), recall(pred, true),
                     list_overlap(pred, true), ap_retrieved(pred, true)])


def rank_binary(tags: sp.csr_matrix, queries, top: int, rng) -> np.ndarray:
    """Predicted neighbours by Euclidean distance between binary tags (=
    Hamming); exact ties broken at random."""
    ones = np.asarray(tags.sum(1)).ravel()
    overlap = (tags[queries] @ tags.T).toarray()
    dist = ones[queries][:, None] + ones[None, :] - 2 * overlap
    return _nearest(dist, queries, top, rng)


def rank_dense(h: np.ndarray, queries, top: int) -> np.ndarray:
    return _nearest(_sqdist(h[queries], h), queries, top, None)


def score_binary(tags: sp.csr_matrix, queries, true, rng) -> np.ndarray:
    return _score(rank_binary(tags, queries, true.shape[1], rng), true)


def score_dense(h: np.ndarray, queries, true) -> np.ndarray:
    return _score(rank_dense(h, queries, true.shape[1]), true)


# ---------------------------------------------------------------- experiments

def _summ(v) -> dict:
    """(trials, ..., metric) -> {metric: {mean, sd}} over trials."""
    v = np.asarray(v)
    return {m: {"mean": v[..., i].mean(0).tolist(), "sd": v[..., i].std(0, ddof=1).tolist()}
            for i, m in enumerate(METRICS)}


REPL_KEYS = ("lsh", "lsh_sign", "fly_20k", "fly_10d", "random_20k")


def replicate(cfg: Config, trials: int = TRIALS, datasets=DATASETS, log=print) -> dict:
    """The 2017 comparison with random matrices (a reimplementation: several
    details are not specified in the paper, see PROVENANCE).

    Methods: real-valued and sign LSH with k projections; the fly with
    m = 20k (operation-matched, their Fig. 2B) and m = 10d (their Fig. 3);
    random tag selection from the m = 20k expansion (their Fig. 2B control);
    and, as our addition, real-valued LSH given the projection arithmetic of
    the m = 10d fly under their accounting (10d cells x s additions = 2d
    operations per projection x 10s/2 projections).

    Ground truth is the top 2% by Euclidean distance on the raw features (as
    the paper states); the same predictions are also scored against neighbours
    on the row-centred input the hashes see ("truth_centred"), the choice made
    in the FlyLSH code.
    """
    from .experiments import save
    t0 = time.time()
    out = {"protocol": {"n": N_DATA, "queries": N_QUERIES, "top": TOP, "trials": trials,
                        "hash_lengths": list(HASH_LENGTHS), "sampling": SAMPLING,
                        "metrics": list(METRICS)},
           "reported": {"mnist_k4_lsh": 0.160, "mnist_k4_fly_10d": 0.448,
                        "sift_k4_random_20k": 0.177, "sift_k4_wta_20k": 0.324},
           "datasets": {}}
    for name in datasets:
        x = load_benchmark(cfg, name)
        xc = centre(x)
        x32 = xc.astype(np.float32)
        n, d = x.shape
        top = int(TOP * n)
        s = max(1, int(round(SAMPLING * d)))
        ops_bits = int(round(10 * d * s / (2 * d)))
        res = {t: {k: [] for k in REPL_KEYS + ("lsh_ops_10d",)} for t in ("raw", "centred")}
        for t in range(trials):
            rng = np.random.default_rng(1000 + t)
            q = rng.choice(n, N_QUERIES, replace=False)
            truths = {"raw": truth(x, q, top), "centred": truth(xc, q, top)}
            y10 = x32 @ fly_matrix(d, 10 * d, rng)
            preds = {key: [] for key in REPL_KEYS}
            for k in HASH_LENGTHS:
                g = xc @ rng.normal(size=(d, k))
                preds["lsh"].append(rank_dense(g, q, top))
                preds["lsh_sign"].append(rank_binary(sp.csr_matrix(g > 0, dtype=np.float32), q, top, rng))
                y20 = x32 @ fly_matrix(d, 20 * k, rng)
                preds["fly_20k"].append(rank_binary(winners(y20, k), q, top, rng))
                preds["random_20k"].append(rank_dense(y20[:, rng.choice(20 * k, k, replace=False)], q, top))
                preds["fly_10d"].append(rank_binary(winners(y10, k), q, top, rng))
            ops = rank_dense(xc @ rng.normal(size=(d, ops_bits)), q, top)
            for tn, tr in truths.items():
                for key in REPL_KEYS:
                    res[tn][key].append([_score(pk, tr) for pk in preds[key]])
                res[tn]["lsh_ops_10d"].append(_score(ops, tr))
            if (t + 1) % 10 == 0:
                log(f"  {name}: trial {t + 1}/{trials}")
        out["datasets"][name] = {"d": d, "sampled": s, "lsh_ops_10d_bits": ops_bits,
                                 **{key: _summ(v) for key, v in res["raw"].items()},
                                 "truth_centred": {key: _summ(v) for key, v in res["centred"].items()}}
        r = out["datasets"][name]
        i4 = HASH_LENGTHS.index(4)
        log(f"  {name} k=4 AP: LSH {r['lsh']['ap']['mean'][i4]:.3f}  "
            f"fly 20k {r['fly_20k']['ap']['mean'][i4]:.3f}  fly 10d {r['fly_10d']['ap']['mean'][i4]:.3f}  "
            f"random 20k {r['random_20k']['ap']['mean'][i4]:.3f}  "
            f"LSH at 10d ops ({ops_bits}) {r['lsh_ops_10d']['ap']['mean']:.3f}")
        save({**out, "seconds": time.time() - t0}, "replication.json")
    out["dimension_sweep"] = dimension_sweep(cfg, log=log)
    out["seconds"] = time.time() - t0
    save(out, "replication.json")
    return out


def dimension_sweep(cfg: Config, dims=(8, 16, 32, 64, 128, 256, 512), trials: int = 10,
                    ks=(4, 16), random_ties: bool = False, log=print) -> dict:
    """One distribution, varying input dimension: MNIST reduced by PCA to d
    components, ground truth on that reduced input, everything else as in
    `replicate`. Isolates d from dataset differences (it does not isolate it
    from the information PCA discards, which is common to all methods)."""
    x = load_benchmark(cfg, "mnist")
    out = {"dims": list(dims), "ks": list(ks), "trials": trials, "random_ties": random_ties,
           "rows": []}
    for d in dims:
        z = pca(x, d)
        zc = centre(z)
        z32 = zc.astype(np.float32)
        n = len(z)
        top = int(TOP * n)
        s = max(1, int(round(SAMPLING * d)))
        ops_bits = max(1, int(round(10 * d * s / (2 * d))))
        acc = {"lsh": [], "fly_20k": [], "fly_10d": [], "lsh_ops_10d": []}
        for t in range(trials):
            rng = np.random.default_rng(7000 + t)
            q = rng.choice(n, N_QUERIES, replace=False)
            true = truth(z, q, top)
            y10 = z32 @ fly_matrix(d, 10 * d, rng)
            prio = (lambda m: np.random.default_rng(t).random(m)) if random_ties else (lambda m: None)
            acc["lsh"].append([score_dense(zc @ rng.normal(size=(d, k)), q, true) for k in ks])
            acc["fly_20k"].append([score_binary(winners(z32 @ fly_matrix(d, 20 * k, rng), k, prio(20 * k)),
                                                q, true, rng) for k in ks])
            acc["fly_10d"].append([score_binary(winners(y10, k, prio(10 * d)), q, true, rng) for k in ks])
            acc["lsh_ops_10d"].append(score_dense(zc @ rng.normal(size=(d, ops_bits)), q, true))
        row = {"d": d, "sampled": s, "lsh_ops_10d_bits": ops_bits,
               **{key: _summ(v) for key, v in acc.items()}}
        out["rows"].append(row)
        log(f"  dimension {d}: k=4 AP LSH {row['lsh']['ap']['mean'][0]:.3f} "
            f"fly10d {row['fly_10d']['ap']['mean'][0]:.3f} LSH@ops {row['lsh_ops_10d']['ap']['mean']:.3f}")
    return out


def connectome(cfg: Config, trials: int = TRIALS, B: int = 50, datasets=DATASETS,
               B_ablation: int = 20, log=print) -> dict:
    """The same protocol and data, hashed through the MaleCNS connectome (right
    hemisphere). See `score_wiring`. For odours the nulls are the first B of
    the primary analysis's pool. A normalisation ablation repeats each dataset
    without row-centring (B_ablation nulls). Results are merged by dataset
    into results/connectome_benchmarks.json."""
    from .experiments import context, null_pool, save, ROOT
    import json
    path = ROOT / "results" / "connectome_benchmarks.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out.update({"B": B, "trials": trials, "B_ablation": B_ablation,
                "datasets": out.get("datasets", {})})
    ctx = context(cfg)
    for name in datasets:
        p = ctx.proj if name == "odours" else ctx.full
        pool = null_pool(p, 200)[:B] if name == "odours" else \
            null_pool(fh.Projection("b", p.binary, p.glomeruli), B)
        x = load_benchmark(cfg, name)
        res = score_wiring(name, x, p, pool, trials, log)
        res["no_centring"] = score_wiring(name, x, p, pool[:B_ablation], trials, log,
                                          centre_input=False, controls=True)
        out["datasets"][name] = res
        save(out, "connectome_benchmarks.json")
    return out


CONTROLS = {"in_equal": dict(equal_in=True), "out_equal": dict(equal_out=True),
            "both_equal": dict(equal_in=True, equal_out=True)}


def score_wiring(name: str, x: np.ndarray, p: fh.Projection, pool, trials: int,
                 log=print, centre_input: bool = True, controls: bool = True) -> dict:
    """One wiring on one dataset under the 2017 protocol.

    For odours, `x` is already over the glomeruli of `p`. Other datasets are
    reduced by PCA to one component per glomerulus, and every trial assigns
    components to glomeruli by a fresh random permutation (image or word
    features have no glomerulus). Every trial draws new queries and scores
    every matrix on the same input; ground truth is the top 2% by Euclidean
    distance on that input before centring. Each matrix's score is averaged
    over trials and the wiring is ranked among the nulls in `pool`.

    Besides the nulls (both degree sequences kept) every trial draws one
    matrix of each equal-connection control (`CONTROLS`: inputs per cell made
    even, fan-out made even, or both, always nnz(M) connections) and one of
    the six-input construction of 2017 (which also changes nnz).

    Row-centring subtracts mean(x) * inputs(i) from cell i's drive, which
    reorders winners only when inputs per cell differ; `centre_input=False`
    removes it (ablation).
    """
    wiring = p.binary.astype(np.float32)
    g, m = wiring.shape
    nulls = [q.matrix.astype(np.float32) for q in pool]
    nnz = int(wiring.sum())
    sizes = tuple(sorted(set(HASH_LENGTHS) | {int(round(0.05 * m))}))
    ops_bits = max(1, int(round(nnz / (2 * g))))  # 2017 accounting: d mult + d add per projection
    z = x if name == "odours" else pca(x, g)
    n = len(z)
    top = int(TOP * n)
    ctrl = list(CONTROLS) if controls else []
    acc = {k: [] for k in ("real", "null", "random_2017", "lsh", "lsh_sign", "lsh_ops", *ctrl)}
    pb = fh.Projection("b", p.binary, p.glomeruli)
    for t in range(trials):
        rng = np.random.default_rng(5000 + t)
        zp = z if name == "odours" else z[:, rng.permutation(g)]
        zt = centre(zp) if centre_input else zp
        z32 = zt.astype(np.float32)
        q = rng.choice(n, N_QUERIES, replace=False)
        true = truth(zp, q, top)
        tie_seed = int(rng.integers(2**31))

        def fly(w):
            y = z32 @ w
            # the same tie-breaking draws for every matrix within a trial
            tie = np.random.default_rng(tie_seed)
            return [score_binary(winners(y, k), q, true, tie) for k in sizes]
        acc["real"].append(fly(wiring))
        acc["null"].append([fly(w) for w in nulls])
        acc["random_2017"].append(fly(fly_matrix(g, m, rng, sampled=6)))
        for c in ctrl:
            w = fh.margin_control(pb, seed=int(rng.integers(2**31)), **CONTROLS[c])
            acc[c].append(fly(w.matrix.astype(np.float32)))
        acc["lsh"].append([score_dense(zt @ rng.normal(size=(g, k)), q, true) for k in sizes])
        acc["lsh_sign"].append([score_binary(sp.csr_matrix(zt @ rng.normal(size=(g, k)) > 0,
                                                           dtype=np.float32), q, true, rng)
                                for k in sizes])
        acc["lsh_ops"].append(score_dense(zt @ rng.normal(size=(g, ops_bits)), q, true))
        if (t + 1) % 10 == 0:
            log(f"  {name}: trial {t + 1}/{trials}")
    mean = {k: np.mean(v, 0) for k, v in acc.items()}   # null: (B, sizes, metric)
    per_metric = {}
    for j, met in enumerate(METRICS):
        rows = []
        for i, k in enumerate(sizes):
            nd = mean["null"][:, i, j]
            rt = stats.randomization_test(mean["real"][i, j], nd)
            rows.append({"k": k, "real": float(mean["real"][i, j]),
                         "null_mean": float(nd.mean()), "null_sd": float(nd.std(ddof=1)),
                         "relative_difference": float(mean["real"][i, j] / nd.mean() - 1),
                         "z": float((mean["real"][i, j] - nd.mean()) / nd.std(ddof=1)),
                         "p_two_sided": rt["p_two_sided"],
                         "random_2017": float(mean["random_2017"][i, j]),
                         **{c: float(mean[c][i, j]) for c in ctrl},
                         "lsh": float(mean["lsh"][i, j]),
                         "lsh_sign": float(mean["lsh_sign"][i, j])})
        per_metric[met] = {"rows": rows, "lsh_ops": float(mean["lsh_ops"][j])}
    r4 = per_metric["ap"]["rows"][sizes.index(4)]
    log(f"  {name}{'' if centre_input else ' (no centring)'} k=4 AP: real {r4['real']:.3f}  "
        f"null {r4['null_mean']:.3f}  2017 {r4['random_2017']:.3f}  "
        + "  ".join(f"{c} {r4[c]:.3f}" for c in ctrl)
        + f"  LSH {r4['lsh']:.3f}  p {r4['p_two_sided']:.3f}")
    return {**per_metric, "n_glomeruli": g, "n_cells": m, "nnz": nnz,
            "inputs_mean": float(wiring.sum(0).mean()), "sizes": list(sizes),
            "lsh_ops_matched_bits": ops_bits, "centred": centre_input,
            "input": "DoOR mixtures, measured glomeruli" if name == "odours"
            else "PCA, random component-glomerulus assignment per trial"}


# ---------------------------------------------------------------- control uncertainty

def trial_controls(name: str, x: np.ndarray, p: fh.Projection, pool, trials: int,
                   log=print) -> dict:
    """Trial-level scores of the connectome, its nulls and the degree controls
    under the 2017 protocol (same trial seeds, queries and component
    assignments as `score_wiring`), kept so that paired contrasts can be given
    intervals. Per trial: AP@200 of the connectome, the mean over the nulls in
    `pool`, and one fresh draw of each equal-connection control and of the
    six-input construction."""
    wiring = p.binary.astype(np.float32)
    g, m = wiring.shape
    nulls = [q.matrix.astype(np.float32) for q in pool]
    sizes = tuple(sorted(set(HASH_LENGTHS) | {int(round(0.05 * m))}))
    z = x if name == "odours" else pca(x, g)
    n = len(z)
    top = int(TOP * n)
    pb = fh.Projection("b", p.binary, p.glomeruli)
    keys = ("real", "null", *CONTROLS, "random_2017")
    out = {k: [] for k in keys}
    for t in range(trials):
        rng = np.random.default_rng(5000 + t)
        zp = z if name == "odours" else z[:, rng.permutation(g)]
        z32 = centre(zp).astype(np.float32)
        q = rng.choice(n, N_QUERIES, replace=False)
        true = truth(zp, q, top)
        tie_seed = int(rng.integers(2**31))
        crng = np.random.default_rng(80_000 + t)

        def fly(w):
            y = z32 @ w
            tie = np.random.default_rng(tie_seed)
            return [average_precision(rank_binary(winners(y, k), q, top, tie), true) for k in sizes]
        out["real"].append(fly(wiring))
        out["null"].append(np.mean([fly(w) for w in nulls], axis=0).tolist())
        for c, kw in CONTROLS.items():
            out[c].append(fly(fh.margin_control(pb, seed=int(crng.integers(2**31)), **kw)
                              .matrix.astype(np.float32)))
        out["random_2017"].append(fly(fly_matrix(g, m, crng, sampled=6)))
        if (t + 1) % 5 == 0:
            log(f"  {name}: control trial {t + 1}/{trials}")
    return {"sizes": list(sizes), "trials": trials, "B": len(pool), **out}


def contrast(ctrl: np.ndarray, null: np.ndarray, draws: int = 2000, seed: int = 0) -> dict:
    """Relative difference of aggregate means, 100 * (mean ctrl / mean null - 1),
    with a 95% percentile bootstrap over trials (the independent units; queries
    within a trial share their database and matrices and are not resampled).
    Also the mean and SD of the per-trial ratios, a different estimand."""
    ctrl, null = np.asarray(ctrl, float), np.asarray(null, float)
    est = 100 * (ctrl.mean() / null.mean() - 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ctrl), (draws, len(ctrl)))
    boot = 100 * (ctrl[idx].mean(1) / null[idx].mean(1) - 1)
    per = 100 * (ctrl / null - 1)
    return {"estimate": float(est), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "per_trial_mean": float(per.mean()), "per_trial_sd": float(per.std(ddof=1)),
            "n_trials": int(len(ctrl))}


def controls(cfg: Config, only: list[str] | None = None, B: int = 10, trials_primary: int = 20,
             trials_other: int = 10, log=print) -> dict:
    """Trial-level degree controls for every hemisphere (the MaleCNS right
    hemisphere with more trials); results/controls.json, merged by hemisphere."""
    import json
    from . import connectomes as C
    from .experiments import ROOT, null_pool, save
    path = ROOT / "results" / "controls.json"
    out = json.loads(path.read_text()) if path.exists() else {"hemispheres": {}}
    out.update({"B": B, "estimand": "100 * (mean over trials of control AP / mean over trials "
                                   "of null-mean AP - 1); 95% bootstrap over trials"})
    data = {nm: load_benchmark(cfg, nm) for nm in ("sift", "glove", "mnist")}
    for ds, side in C.HEMISPHERES:
        key = f"{ds}_{side}"
        if only and key not in only:
            continue
        p = C.projection(cfg, ds, side)
        pb = fh.Projection("b", p.binary, p.glomeruli, meta=p.meta)
        od = fh.load_odours(cfg, p.glomeruli)
        po = fh.align(pb, od.glomeruli)
        trials = trials_primary if key == "malecns_R" else trials_other
        res = {}
        for nm in DATASETS:
            log(f"  {key}: {nm}")
            if nm == "odours":
                r = trial_controls(nm, load_benchmark(cfg, "odours", odours=od.x), po,
                                   null_pool(po, 100)[:B], trials, log)
            else:
                r = trial_controls(nm, data[nm], pb, null_pool(pb, 100)[:B], trials, log)
            null = np.array(r["null"])
            r["contrasts"] = {c: [contrast(np.array(r[c])[:, i], null[:, i], seed=i)
                                  for i in range(len(r["sizes"]))]
                              for c in ("real", *CONTROLS, "random_2017")}
            res[nm] = r
        # re-read before saving so that parallel runs over different
        # hemispheres do not overwrite each other
        cur = json.loads(path.read_text()) if path.exists() else {"hemispheres": {}}
        cur.update({k: v for k, v in out.items() if k != "hemispheres"})
        cur["hemispheres"][key] = res
        save(cur, "controls.json")
        out = cur
    return out


def tie_sweep(cfg: Config, log=print) -> dict:
    """The dimension sweep again with a seeded random tie priority among
    Kenyon cells; stored next to the original in results/replication.json."""
    import json
    from .experiments import ROOT, save
    res = dimension_sweep(cfg, random_ties=True, log=log)
    path = ROOT / "results" / "replication.json"
    rp = json.loads(path.read_text())
    rp["dimension_sweep_random_ties"] = res
    save(rp, "replication.json")
    return res


# ---------------------------------------------------------------- other fly-hash tasks

def _auc(novel: np.ndarray, familiar: np.ndarray) -> float:
    """Probability that a novel query scores more novel than a familiar one."""
    s = np.concatenate([novel, familiar])
    r = np.argsort(np.argsort(s, kind="stable"), kind="stable") + 1.0
    return float((r[:len(novel)].sum() - len(novel) * (len(novel) + 1) / 2) / (len(novel) * len(familiar)))


def fly_bloom(tags_store: np.ndarray, tags_query: np.ndarray) -> np.ndarray:
    """Novelty of each query under a graded fly Bloom filter (after Dasgupta
    et al. 2018): each Kenyon cell's output synapse is depressed in proportion
    to how often stored items activate it, w_j = 1 - count_j / n_stored; a
    query's novelty is the mean synapse value over its active cells. The
    binary form (w_j = 0 after one activation) saturates once n_stored * k
    exceeds the number of cells, which is the regime tested here."""
    w = 1.0 - tags_store.sum(0) / max(len(tags_store), 1)
    return (tags_query * w).sum(1) / np.maximum(tags_query.sum(1), 1)


def other_tasks(cfg: Config, trials: int = 10, B: int = 10, log=print) -> dict:
    """Novelty detection (fly Bloom filter) and nearest-neighbour
    classification (FlyNN: one Bloom filter per class, predict the least
    novel class; Ram and Sinha) with the MaleCNS right connectome, its
    degree-preserving nulls, the equal-connection controls and the 2017
    construction. MNIST (PCA to one component per glomerulus, random
    component-glomerulus assignment per trial): store digits 0-4, familiar =
    held-out 0-4, novel = 5-9; classification on all ten digits. Odours
    (measured glomeruli): store mixtures of half of the odorants, familiar =
    new mixtures of that half, novel = mixtures of the other half. Results in
    results/tasks.json as AUC / accuracy with 95% bootstrap intervals over
    trials of the difference to each trial's null mean."""
    from .experiments import context, null_pool, save
    ctx = context(cfg)
    d = cfg.raw_dir / "bench"
    mn = np.load(d / "mnist.npz")
    X, y = mn["x_train"].reshape(-1, 784).astype(float), mn["y_train"]
    keys = ("real", "null", *CONTROLS, "random_2017")
    out = {"trials": trials, "B": B, "tasks": {}}
    for task in ("mnist_novelty", "mnist_flynn", "odour_novelty"):
        p = ctx.proj if task.startswith("odour") else ctx.full
        pb = fh.Projection("b", p.binary, p.glomeruli)
        g, m = pb.matrix.shape
        nulls = [q.matrix.astype(np.float32) for q in null_pool(p if task.startswith("odour") else pb, 100)[:B]]
        acc = {k: [] for k in keys}
        for t in range(trials):
            rng = np.random.default_rng(6000 + t)
            crng = np.random.default_rng(90_000 + t)
            if task.startswith("mnist"):
                idx = rng.choice(len(X), 6000, replace=False)
                z = pca(X[idx], g)[:, rng.permutation(g)]
                lab = y[idx]
            else:
                src = ctx.odours.x
                half = rng.permutation(len(src))
                a, b = src[half[: len(src) // 2]], src[half[len(src) // 2:]]
                z = np.vstack([fh.mixtures(a, 3000, seed=t), fh.mixtures(a, 500, seed=100 + t),
                               fh.mixtures(b, 500, seed=200 + t)])
                lab = np.array([0] * 3500 + [1] * 500)
            zc = centre(z).astype(np.float32)
            k = int(round(0.05 * m))

            def score(w):
                T = winners(zc @ w, k).toarray()
                if task == "mnist_novelty":
                    tr, te = np.arange(4000), np.arange(4000, 6000)
                    store = T[tr][lab[tr] < 5]
                    nov = fly_bloom(store, T[te])
                    return _auc(nov[lab[te] >= 5], nov[lab[te] < 5])
                if task == "mnist_flynn":
                    tr, te = np.arange(4000), np.arange(4000, 6000)
                    s = np.stack([fly_bloom(T[tr][lab[tr] == c], T[te]) for c in range(10)], 1)
                    return float((s.argmin(1) == lab[te]).mean())
                nov = fly_bloom(T[:3000], T[3000:])
                return _auc(nov[500:], nov[:500])
            acc["real"].append(score(pb.matrix.astype(np.float32)))
            acc["null"].append(float(np.mean([score(w) for w in nulls])))
            for c, kw in CONTROLS.items():
                acc[c].append(score(fh.margin_control(pb, seed=int(crng.integers(2**31)), **kw)
                                    .matrix.astype(np.float32)))
            acc["random_2017"].append(score(fly_matrix(g, m, crng, sampled=6)))
        null = np.array(acc["null"])
        res = {"k": k, "scores": acc, "mean": {c: float(np.mean(v)) for c, v in acc.items()},
               "difference": {}}
        rng = np.random.default_rng(1)
        for c in ("real", *CONTROLS, "random_2017"):
            dlt = np.array(acc[c]) - null
            boot = dlt[rng.integers(0, trials, (2000, trials))].mean(1)
            res["difference"][c] = {"mean": float(dlt.mean()),
                                    "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        out["tasks"][task] = res
        log(f"  {task}: " + "  ".join(f"{c} {res['mean'][c]:.3f}" for c in keys))
        save(out, "tasks.json")
    return out
