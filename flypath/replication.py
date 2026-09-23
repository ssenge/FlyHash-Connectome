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
"Mean average precision" is taken in the standard IR sense (their ref. 19);
the list-overlap score of the FlyLSH code is reported alongside. Of the two,
the standard definition is the one close to their published values.

`replicate` runs that protocol. `connectome` keeps the protocol and the data
but replaces the random fly matrix with the measured one: each dataset is
reduced by PCA to one component per glomerulus, and the measured wiring is
compared with its curveball nulls, with the 2017 random construction at the
same size, and with LSH.
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
DATASETS = ("sift", "glove", "mnist")
SOURCES = {
    "mnist.npz": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
    "siftsmall.tar.gz": "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
    "glove.6B.zip": "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
}


# ---------------------------------------------------------------- data

def load_benchmark(cfg: Config, name: str, n: int = N_DATA, seed: int = 0) -> np.ndarray:
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


def winners(y: np.ndarray, k: int) -> sp.csr_matrix:
    """Binary top-k tag per row, as a sparse matrix (ties: by partition order)."""
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
    """Mean average precision, the standard IR definition (their ref. 19):
    for each query, the mean of precision@i over the ranks i of the predicted
    list at which a true neighbour appears (0 if none does)."""
    hit = _hits(pred, true)
    prec = np.cumsum(hit, axis=1) / np.arange(1, hit.shape[1] + 1)
    return float(((prec * hit).sum(1) / np.maximum(hit.sum(1), 1)).mean())


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


METRICS = ("map", "overlap")


def _score(pred, true) -> np.ndarray:
    return np.array([average_precision(pred, true), list_overlap(pred, true)])


def score_binary(tags: sp.csr_matrix, queries, true, rng) -> np.ndarray:
    """Euclidean distance between binary tags (= Hamming); ties at random."""
    ones = np.asarray(tags.sum(1)).ravel()
    overlap = (tags[queries] @ tags.T).toarray()
    dist = ones[queries][:, None] + ones[None, :] - 2 * overlap
    return _score(_nearest(dist, queries, true.shape[1], rng), true)


def score_dense(h: np.ndarray, queries, true) -> np.ndarray:
    return _score(_nearest(_sqdist(h[queries], h), queries, true.shape[1], None), true)


# ---------------------------------------------------------------- experiments

def _summ(v) -> dict:
    """(trials, ..., metric) -> {metric: {mean, sd}} over trials."""
    v = np.asarray(v)
    return {m: {"mean": v[..., i].mean(0).tolist(), "sd": v[..., i].std(0, ddof=1).tolist()}
            for i, m in enumerate(METRICS)}


def replicate(cfg: Config, trials: int = TRIALS, datasets=DATASETS, log=print) -> dict:
    """The 2017 comparison with random matrices, on their three datasets.

    Methods: dense and sign LSH with k projections; the fly with m = 20k
    (operation-matched, their Fig. 2B) and m = 10d (their Fig. 3); random tag
    selection from the m = 20k expansion (their Fig. 2B control); and, as our
    addition, dense LSH given the operation count of the m = 10d fly under
    their accounting (10d cells x 0.1d additions = 2d operations x d/2 bits).
    """
    from .experiments import save
    t0 = time.time()
    out = {"protocol": {"n": N_DATA, "queries": N_QUERIES, "top": TOP, "trials": trials,
                        "hash_lengths": list(HASH_LENGTHS), "sampling": SAMPLING},
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
        keys = ("lsh", "lsh_sign", "fly_20k", "fly_10d", "random_20k")
        res = {k: [] for k in keys + ("lsh_ops_10d",)}
        for t in range(trials):
            rng = np.random.default_rng(1000 + t)
            q = rng.choice(n, N_QUERIES, replace=False)
            true = truth(x, q, top)
            y10 = x32 @ fly_matrix(d, 10 * d, rng)
            row = {key: [] for key in keys}
            for k in HASH_LENGTHS:
                g = xc @ rng.normal(size=(d, k))
                row["lsh"].append(score_dense(g, q, true))
                row["lsh_sign"].append(score_binary(sp.csr_matrix(g > 0, dtype=np.float32), q, true, rng))
                y20 = x32 @ fly_matrix(d, 20 * k, rng)
                row["fly_20k"].append(score_binary(winners(y20, k), q, true, rng))
                row["random_20k"].append(score_dense(y20[:, rng.choice(20 * k, k, replace=False)], q, true))
                row["fly_10d"].append(score_binary(winners(y10, k), q, true, rng))
            for key in keys:
                res[key].append(row[key])
            res["lsh_ops_10d"].append(score_dense(xc @ rng.normal(size=(d, ops_bits)), q, true))
            if (t + 1) % 10 == 0:
                log(f"  {name}: trial {t + 1}/{trials}")
        out["datasets"][name] = {"d": d, "sampled": s, "lsh_ops_10d_bits": ops_bits,
                                 **{key: _summ(v) for key, v in res.items()}}
        r = out["datasets"][name]
        i4 = HASH_LENGTHS.index(4)
        log(f"  {name} k=4 mAP: LSH {r['lsh']['map']['mean'][i4]:.3f}  "
            f"fly 20k {r['fly_20k']['map']['mean'][i4]:.3f}  fly 10d {r['fly_10d']['map']['mean'][i4]:.3f}  "
            f"random 20k {r['random_20k']['map']['mean'][i4]:.3f}  "
            f"LSH at 10d ops ({ops_bits} bits) {r['lsh_ops_10d']['map']['mean']:.3f}")
        save({**out, "seconds": time.time() - t0}, "replication.json")
    out["seconds"] = time.time() - t0
    save(out, "replication.json")
    return out


def connectome(cfg: Config, trials: int = TRIALS, B: int = 50, datasets=DATASETS, log=print) -> dict:
    """The same protocol and data, hashed through the measured wiring.

    Each dataset is reduced by PCA to one component per glomerulus; every
    trial assigns components to glomeruli by a fresh random permutation (the
    pairing of image or word features with glomeruli has no meaning), draws
    new queries, and scores every matrix on that same input. Ground truth is
    the top 2% by Euclidean distance on the reduced input, which is what all
    hashes see. Each matrix's score is averaged over trials; the measured
    wiring is then ranked among its B curveball nulls.
    """
    from .experiments import context, null_pool, save
    t0 = time.time()
    real = context(cfg).full                       # all olfactory glomeruli, right hemisphere
    wiring = real.binary.astype(np.float32)
    g, m = wiring.shape
    nulls = [q.matrix.astype(np.float32)
             for q in null_pool(fh.Projection("b", real.binary, real.glomeruli), B)]
    nnz = int(wiring.sum())
    sizes = tuple(sorted(set(HASH_LENGTHS) | {int(round(0.05 * m))}))
    ops_bits = max(1, int(round(nnz / (2 * g))))  # 2017 accounting: d mult + d add per projection
    out = {"n_glomeruli": g, "n_cells": m, "nnz": nnz, "inputs_mean": float(wiring.sum(0).mean()),
           "sizes": list(sizes), "B": B, "trials": trials, "lsh_ops_matched_bits": ops_bits,
           "reduction": "PCA to n_glomeruli components, random component-glomerulus assignment per trial",
           "datasets": {}}
    for name in datasets:
        z = pca(load_benchmark(cfg, name), g)
        n = len(z)
        top = int(TOP * n)
        acc = {k: [] for k in ("real", "null", "random_2017", "lsh", "lsh_sign", "lsh_ops")}
        for t in range(trials):
            rng = np.random.default_rng(5000 + t)
            zt = centre(z[:, rng.permutation(g)])
            z32 = zt.astype(np.float32)
            q = rng.choice(n, N_QUERIES, replace=False)
            true = truth(zt, q, top)
            tie_seed = int(rng.integers(2**31))

            def fly(w):
                y = z32 @ w
                # the same tie-breaking draws for every matrix within a trial
                tie = np.random.default_rng(tie_seed)
                return [score_binary(winners(y, k), q, true, tie) for k in sizes]
            acc["real"].append(fly(wiring))
            acc["null"].append([fly(w) for w in nulls])
            acc["random_2017"].append(fly(fly_matrix(g, m, rng, sampled=6)))
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
                             "p_two_sided": rt["p_two_sided"],
                             "random_2017": float(mean["random_2017"][i, j]),
                             "lsh": float(mean["lsh"][i, j]),
                             "lsh_sign": float(mean["lsh_sign"][i, j])})
            per_metric[met] = {"rows": rows, "lsh_ops": float(mean["lsh_ops"][j])}
        out["datasets"][name] = per_metric
        r4 = per_metric["map"]["rows"][sizes.index(4)]
        log(f"  {name} k=4 mAP: real {r4['real']:.3f}  null {r4['null_mean']:.3f}  "
            f"2017-random {r4['random_2017']:.3f}  LSH {r4['lsh']:.3f}  p {r4['p_two_sided']:.3f}")
        save({**out, "seconds": time.time() - t0}, "connectome_benchmarks.json")
    out["seconds"] = time.time() - t0
    save(out, "connectome_benchmarks.json")
    return out
