"""E4 stages 1 and 2: retrieval (AP@200) of the frozen spectral fan-out
allocation f*(Lambda, k) against even fan-out, the power grid f ~ lambda^alpha
and input pre-scaling.

Pre-registration: experiments/PREREGISTRATION_2026-09-28.md, E4 stages 1 and 2.
f* comes from `flypath.spectral`, which is frozen (stage 0). This module does
not modify it. At start it checks that the SHA-256 of flypath/spectral.py
equals the one recorded in results/e4_frozen.json and aborts if not.

Only the runner is here. It writes per-trial AP@200 and recall@200 for every
arm, dataset and k to results/e4_spectral.json, so that the pre-registered
paired intervals and criteria can be computed from the stored trials. It
computes no interval and no verdict.

    VECLIB_MAXIMUM_THREADS=3 PYTHONPATH=. python -m flypath.spectral_run --budget-minutes 8

The run checkpoints after every trial and after every f* model evaluation.
Call the same command again to resume. Exit status 0 means the run is
complete; 3 means call again. `--smoke --out <scratch file>` runs a tiny
configuration that only checks the code paths.


Protocol (both stages)
----------------------
  items     n = 10,000 fixed vectors per dataset, d = 51 inputs.
  trials    each trial draws 1,000 queries from the items and new matrices.
            Truth is the 200 nearest items (top 2%) by Euclidean distance on
            the unscaled, uncentred input (the query itself excluded). The
            score is AP@200 (`replication.average_precision`, all 200 true
            neighbours in the denominator). Recall@200 is stored as well.
  hash      the (possibly pre-scaled) input is row-centred
            (`replication.centre`), then y = centre(x) @ M. The tag is exactly
            the k cells with the largest drive, and items are ranked by Hamming
            distance between tags (`replication.rank_binary`).
  matrices  binary d x m, m = 1,886 cells. Column sums are the MaleCNS right
            hemisphere's inputs per cell (nnz = 9,967) and row sums are the
            arm's fan-out f. The margins are realised greedily and the pairing
            is randomised with 30 curveball sweeps (`spectral.realise`, which
            uses `flyhash.curveball`).
  arms      per k:
              even           f = power_allocation(lambda, 0), even fan-out
              grid:<a>       f ~ lambda^a, a in {0, .25, .5, .75, 1, 1.25, 1.5, 2}
                             (`spectral.power_allocation`, the frozen family);
                             grid:0 is the even allocation
              fstar          f*(Lambda, k)
              prescale:<a>   input x_j lambda_j^(a/2), even fan-out, same a grid;
                             prescale:0 is the even arm
              prescale_best  prescale:<a_pre(k)>, where a_pre(k) is chosen on
                             the validation split before any test trial
  ks        4, 16 and 94 (94 = 5% of 1,886) in both stages.

Stage 1 (synthetic): beta in {0, 0.5, 1, 1.5, 2}, lambda_j = j^-beta, x ~
N(0, Lambda). The items are `spectral.synthetic(beta)` (seed 0), the data set
from which the frozen f* and theta were computed. f* is the frozen record
(`spectral.frozen_fstar`). 10 trials.

Stage 2 (real): PCA-51 SIFT, MNIST and GloVe. PCA, Lambda and theta are
fitted on a training split disjoint from the items (and hence from the
queries). f* = `spectral.fstar_all(spectrum(train), (4, 16, 94), c, 9967,
median_nn_angle(train))` with the frozen defaults. 20 trials.


Unspecified details fixed before running
----------------------------------------
 1. Common random numbers. Within a trial, every arm uses the same queries,
    the same wiring seed for `spectral.realise`, the same random priority over
    cells for ties in drive, and, per k, the same random stream for Hamming
    ties. The wiring seed depends only on (stage, dataset, split, trial), not
    on the arm. So arms with the same integer allocation hash with the
    identical matrix and get identical AP. This is the case for even, grid:0
    and prescale:0, and for fstar whenever f* is the same integer vector as a
    grid member or as even fan-out. Such coincidences are recorded under
    "coincident_arms". At beta = 0, Lambda is isotropic, so every arm
    coincides with even fan-out: the stage-1 comparison of f* with even at
    beta = 0 is exact by construction and is not a noise-level test.
 2. Ties in drive. Cells with identical input sets get bitwise-identical
    drives: the drive is computed once per distinct column and copied. The
    tag is exactly k cells: every cell above the k-th largest drive, plus the
    cells tied at it taken in order of a fixed random priority over cells.
    The priority is drawn per trial and shared by all arms.
 3. Allocations. Even fan-out is `power_allocation(lambda, 0, c)`:
    9,967 = 51 x 195 + 22, and the 22 extra connections go to the 22
    highest-variance inputs. The grid arms are `power_allocation(lambda, a,
    c)`: the frozen family, with cap floor(m/2), Gale-Ryser projection and
    largest-remainder rounding, and with larger f on larger lambda for
    a > 0. f* is used exactly as the frozen code returns it. That includes
    alpha* < 0 (more fan-out on low-variance inputs) and alpha* = 6 or inf
    (fan-out concentrated on the top inputs at the cap). Each allocation is
    stored with its alpha* and key.
 4. Pre-scaling multiplies input j by lambda_j^(a/2) before row-centring, and
    hashes with the trial's even-fan-out matrix. The ground truth is
    unchanged. lambda is the nominal j^-beta in stage 1 and the training
    spectrum in stage 2: the same Lambda that f* uses.
 5. Best pre-scaling. a_pre(k) is the grid value with the largest mean AP
    over 10 validation trials, and exact ties go to the smallest a. The
    validation trials follow the same protocol (1,000 queries, top 200, fresh
    matrices) on validation data. In stage 1 that is `spectral.synthetic(beta,
    seed=1)`, a new draw of the same distribution. In stage 2 it is a
    validation split of 10,000 vectors, disjoint from the items and from the
    training split. a_pre(k) is chosen and stored per dataset and k before
    the first test trial of that dataset runs. It is never chosen on test AP.
 6. Stage-2 splits. The items are `replication.load_benchmark(cfg, name)`
    (10,000, seed 0). The training and validation splits are 10,000 vectors
    each, drawn by a seeded permutation of the vectors that are not items:
      MNIST  the rest of x_train;
      GloVe  the rest of the 400,000 words;
      SIFT   siftsmall_base holds exactly the 10,000 items, so the splits come
             from siftsmall_learn (25,000), the texmex learning set.
    The runner checks that the items it reproduces equal `load_benchmark`'s.
    Disjointness is by index (or by file for SIFT). Exact duplicate vectors
    across splits are not searched for.
 7. PCA. It is fitted on the training split: the training mean and the top
    51 right singular vectors of the centred training matrix. Each axis's
    sign is fixed so that its largest-magnitude loading is positive. The
    same map is applied to the items, the training split and the validation
    split. Lambda = `spectral.spectrum(train)`, the variance of each PCA
    coordinate (ddof 0). theta = `spectral.median_nn_angle(train)`, which
    uses 200 neighbours at n = 10,000. f*, Lambda and theta are stored at
    first computation and reused on resume. On resume the split indices must
    be identical and the recomputed Lambda must equal the stored one
    (relative tolerance 1e-8).
 8. Seeds. numpy default_rng([0, crc32 keys...]) keyed by stage, dataset,
    split, trial and purpose (queries, wiring, priority, Hamming ties).
 9. Order. Stage 1 runs before stage 2, betas in increasing order, then the
    datasets SIFT, MNIST, GloVe. Within a dataset the order is PCA, Lambda,
    theta and f* (stage 2), then the validation trials, then the choice of
    a_pre, then the test trials.
10. Stored per trial: AP and recall for every arm and k. The arms are even,
    grid:<a> (8, grid:0 included), prescale:<a> (8, prescale:0 included),
    fstar and prescale_best. The integer allocations and their keys are stored
    per dataset.


Deviations
----------
None.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

from . import replication as rp
from . import spectral as spc
from .config import ROOT, Config

RESULTS = ROOT / "results" / "e4_spectral.json"
FROZEN = ROOT / "results" / "e4_frozen.json"
SEED = 0
GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _label(a: float) -> str:
    return f"{float(a):g}"


DESIGN = {
    "n_items": rp.N_DATA,
    "n_queries": rp.N_QUERIES,
    "top_fraction": rp.TOP,
    "d": spc.D_SYNTH,
    "nnz": spc.NNZ,
    "inputs_per_cell": "MaleCNS R, experiments.context(cfg).full.inputs()",
    "ks": [4, 16, 94],
    "grid_alphas": [_label(a) for a in GRID],
    "prescale_alphas": [_label(a) for a in GRID],
    "sweeps": spc.SWEEPS,
    "seed": SEED,
    "stage1": {"betas": list(spc.BETAS), "trials": 10, "validation_trials": 10,
               "item_seed": spc.SEED, "validation_seed": 1},
    "stage2": {"datasets": ["sift", "mnist", "glove"], "trials": 20, "validation_trials": 10,
               "pca_dims": 51, "n_train": 10_000, "n_validation": 10_000},
    "fstar_model": {"pairs": spc.N_PAIRS, "wirings": spc.N_WIRINGS,
                    "alphas": [spc._alpha_label(a) for a in spc.ALPHAS],
                    "half_width_deg": spc.HALF_WIDTH_DEG, "seed": spc.SEED},
}

# A tiny configuration that only checks that every code path runs.
SMOKE = copy.deepcopy(DESIGN)
SMOKE.update({"n_items": 2000, "n_queries": 100, "ks": [4, 16]})
SMOKE["stage1"].update({"betas": [0.0, 1.0], "trials": 2, "validation_trials": 1})
SMOKE["stage2"].update({"datasets": ["sift"], "trials": 2, "validation_trials": 1,
                        "n_train": 2000, "n_validation": 2000})
SMOKE["fstar_model"].update({"pairs": 1000, "wirings": 2, "alphas": ["0", "1", "inf"]})


# ---------------------------------------------------------------- frozen stage 0

def check_frozen(path: Path | str = FROZEN) -> dict:
    """Abort unless flypath/spectral.py is byte-identical to the frozen code:
    its SHA-256 must equal the one in results/e4_frozen.json, and the record
    must be a complete freeze."""
    path = Path(path)
    rec = json.loads(path.read_text())
    now = hashlib.sha256(Path(spc.__file__).read_bytes()).hexdigest()
    frozen = rec.get("code", {}).get("sha256")
    if not rec.get("complete") or frozen != now:
        raise SystemExit(f"flypath/spectral.py (sha256 {now[:12]}) is not the frozen code "
                         f"recorded in {path} (sha256 {str(frozen)[:12]}, complete="
                         f"{rec.get('complete')}); refusing to run E4 stages 1-2")
    return rec


def inputs_per_cell(cfg: Config, frozen: dict | None = None) -> np.ndarray:
    """MaleCNS right hemisphere, distinct glomerular inputs per Kenyon cell;
    checked against nnz and against the sequence recorded in the freeze."""
    from . import experiments as ex
    c = ex.context(cfg).full.inputs().astype(int)
    if int(c.sum()) != spc.NNZ:
        raise SystemExit(f"MaleCNS R inputs per cell sum to {int(c.sum())}, not {spc.NNZ}")
    if frozen is not None:
        sha1 = hashlib.sha1(c.astype(np.int64).tobytes()).hexdigest()
        if sha1 != frozen["inputs_per_cell"]["sha1"]:
            raise SystemExit("inputs per cell differ from those of the frozen f*")
    return c


# ---------------------------------------------------------------- seeds

def _seed(*keys) -> list[int]:
    return [SEED, *[k if isinstance(k, int) else zlib.crc32(str(k).encode()) for k in keys]]


def _key(f) -> str:
    return hashlib.sha1(np.asarray(f, np.int64).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- inputs

def prescale(z: np.ndarray, lam: np.ndarray, a: float) -> np.ndarray:
    """x_j * lambda_j^(a/2) (before row-centring); a = 0 returns z unchanged."""
    if float(a) == 0.0:
        return z
    return z * np.asarray(lam, float) ** (float(a) / 2.0)


def pca_fit(train: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Training mean and top `dims` principal axes (d x dims) of the training
    split. The sign of each axis is fixed so that its largest-magnitude
    loading is positive."""
    train = np.asarray(train, float)
    mu = train.mean(0)
    _, _, vt = np.linalg.svd(train - mu, full_matrices=False)
    v = vt[:dims].T.copy()
    s = np.sign(v[np.abs(v).argmax(0), np.arange(v.shape[1])])
    s[s == 0] = 1.0
    return mu, v * s


def pca_apply(x: np.ndarray, mu: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (np.asarray(x, float) - mu) @ v


def _glove_lines(cfg: Config) -> list[str]:
    with zipfile.ZipFile(cfg.raw_dir / "bench" / "glove.6B.zip") as z:
        return z.read("glove.6B.300d.txt").decode("utf8").splitlines()


def split_indices(name: str, n_pool: int, n_items: int, n_train: int, n_val: int,
                  seed: int = SEED) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """(item, training, validation) indices. The item indices are those
    `replication.load_benchmark` draws (seed 0; sorted for GloVe). Training
    and validation come from a seeded permutation of the remaining indices.
    For SIFT, whose items are the whole siftsmall_base, the item indices are
    None and the other two index siftsmall_learn (`n_pool` vectors)."""
    rng = np.random.default_rng(_seed("stage2", "split", name, seed))
    if name == "sift":
        items, pool = None, np.arange(n_pool)
    else:
        items = np.random.default_rng(0).choice(n_pool, n_items, replace=False)
        if name == "glove":
            items = np.sort(items)
        pool = np.setdiff1d(np.arange(n_pool), items)
    if len(pool) < n_train + n_val:
        raise RuntimeError(f"{name}: {len(pool)} vectors cannot hold the splits")
    pick = rng.permutation(pool)[: n_train + n_val]
    return items, pick[:n_train], pick[n_train:]


def load_splits(cfg: Config, name: str, n_items: int, n_train: int, n_val: int,
                seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(items, train, validation, meta) in raw feature space. The items are
    `replication.load_benchmark(cfg, name, n_items)`; train and validation are
    disjoint from them and from each other (see "Unspecified details" 6)."""
    items = rp.load_benchmark(cfg, name, n=n_items)          # also fetches the files
    d = cfg.raw_dir / "bench"
    if name == "mnist":
        x = np.load(d / "mnist.npz")["x_train"].reshape(-1, 784).astype(float)
        idx, tr, va = split_indices(name, len(x), n_items, n_train, n_val, seed)
        if not np.array_equal(x[idx], items):
            raise RuntimeError("MNIST items differ from load_benchmark's")
        train, val = x[tr], x[va]
        src = f"MNIST x_train minus the {n_items} items ({len(x) - n_items} vectors)"
    elif name == "sift":
        raw = np.fromfile(d / "siftsmall" / "siftsmall_learn.fvecs", dtype=np.float32)
        x = raw.reshape(-1, 129)[:, 1:].astype(float)
        _, tr, va = split_indices(name, len(x), n_items, n_train, n_val, seed)
        train, val = x[tr], x[va]
        src = f"siftsmall_learn ({len(x)} vectors); the items are siftsmall_base"
    elif name == "glove":
        lines = _glove_lines(cfg)
        idx, tr, va = split_indices(name, len(lines), n_items, n_train, n_val, seed)

        def parse(ii):
            return np.array([[float(v) for v in lines[i].split(" ")[1:]] for i in ii])
        if not np.array_equal(parse(idx[:50]), items[:50]):
            raise RuntimeError("GloVe items differ from load_benchmark's")
        train, val = parse(tr), parse(va)
        src = f"GloVe 6B 300d minus the {n_items} items ({len(lines) - n_items} words)"
    else:
        raise ValueError(f"stage 2 has no dataset {name!r}")
    meta = {"source": src, "n_items": int(len(items)), "n_train": int(len(train)),
            "n_validation": int(len(val)), "raw_dim": int(items.shape[1]),
            "split_sha1": hashlib.sha1(np.concatenate([tr, va]).astype(np.int64)
                                       .tobytes()).hexdigest()}
    return items, train, val, meta


# ---------------------------------------------------------------- the hash

def top_k_exact(y: np.ndarray, k: int, order: np.ndarray) -> sparse.csr_matrix:
    """Exactly k winners per row: every cell above the row's k-th largest
    drive, then the cells tied at it in the order `order` (cells by
    decreasing priority)."""
    n, m = y.shape
    kth = -np.partition(-y, k - 1, axis=1)[:, k - 1:k]
    win = y > kth
    tied = y == kth
    need = k - win.sum(axis=1)
    easy = tied.sum(axis=1) == need
    win[easy] |= tied[easy]
    hard = np.flatnonzero(~easy)
    if len(hard):
        t = tied[hard][:, order]
        take = np.zeros_like(t)
        take[:, order] = t & (np.cumsum(t, axis=1) <= need[hard, None])
        win[hard] |= take
    rows, cols = np.nonzero(win)
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, m))


def tags(xc: np.ndarray, M: np.ndarray, ks, order: np.ndarray,
         chunk: int = 2000) -> dict[int, sparse.csr_matrix]:
    """Top-k tags of the row-centred input `xc` through the binary matrix M,
    for every k. The drive is computed once per distinct column of M and
    copied, so identical cells have bitwise-identical drives."""
    U, inv = np.unique(np.asarray(M, float).T, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    ut = U.T
    parts = {int(k): [] for k in ks}
    for i in range(0, len(xc), chunk):
        y = (xc[i:i + chunk] @ ut)[:, inv]
        for k in parts:
            parts[k].append(top_k_exact(y, k, order))
    return {k: sparse.vstack(v).tocsr() for k, v in parts.items()}


# ---------------------------------------------------------------- arms and trials

def allocations(lam, c, fstar: dict, grid) -> dict[str, np.ndarray]:
    """Integer fan-out per arm allocation (aligned with lam)."""
    out = {"even": spc.power_allocation(lam, 0.0, c)}
    for a in grid:
        out[f"grid:{_label(a)}"] = spc.power_allocation(lam, float(a), c)
    for k, f in fstar.items():
        out[f"fstar:{int(k)}"] = np.asarray(f, int)
    return out


def arm_source(arm: str, k: int, choice: dict | None = None) -> tuple[str, float]:
    """(allocation name, pre-scaling exponent) of an arm at hash length k."""
    if arm == "even":
        return "even", 0.0
    if arm == "fstar":
        return f"fstar:{int(k)}", 0.0
    if arm == "prescale_best":
        return "even", float(choice[str(int(k))]["alpha"])
    kind, a = arm.split(":")
    if kind == "grid":
        return arm, 0.0
    if kind == "prescale":
        return "even", float(a)
    raise ValueError(f"unknown arm {arm!r}")


def test_arms(D: dict) -> list[str]:
    return (["even", "fstar"] + [f"grid:{a}" for a in D["grid_alphas"]]
            + [f"prescale:{a}" for a in D["prescale_alphas"]] + ["prescale_best"])


def validation_arms(D: dict) -> list[str]:
    return [f"prescale:{a}" for a in D["prescale_alphas"]]


def run_trial(z: np.ndarray, lam: np.ndarray, c: np.ndarray, allocs: dict, arms, ks,
              seeds: list[int], n_queries: int, top: int, choice: dict | None = None,
              sweeps: int = spc.SWEEPS) -> dict:
    """One trial of the 2017 protocol for every arm and k with common random
    numbers (see "Unspecified details" 1). Returns {"ap": {arm: {k: .}},
    "recall": {...}, "distinct_hashes": n}."""
    z = np.asarray(z, float)
    n = len(z)
    m = len(c)
    q = np.random.default_rng(seeds + [1]).choice(n, n_queries, replace=False)
    true = rp.truth(z, q, top)
    order = np.argsort(-np.random.default_rng(seeds + [2]).random(m), kind="stable")
    wiring_seed = seeds + [4]
    jobs = {}
    for arm in arms:
        for k in ks:
            alloc, pre = arm_source(arm, k, choice)
            key = (_key(allocs[alloc]), pre)
            j = jobs.setdefault(key, {"alloc": alloc, "pre": pre, "ks": set(), "arms": []})
            j["ks"].add(int(k))
            j["arms"].append((arm, int(k)))
    mats, scored = {}, {}
    for key, j in jobs.items():
        if key[0] not in mats:
            mats[key[0]] = spc.realise(c, allocs[j["alloc"]], seed=wiring_seed, sweeps=sweeps)
        xc = rp.centre(prescale(z, lam, j["pre"]))
        tg = tags(xc, mats[key[0]], sorted(j["ks"]), order)
        for k in sorted(j["ks"]):
            pred = rp.rank_binary(tg[k], q, top, np.random.default_rng(seeds + [3, k]))
            scored[key, k] = (rp.average_precision(pred, true), rp.recall(pred, true))
    out = {"ap": {}, "recall": {}}
    for key, j in jobs.items():
        for arm, k in j["arms"]:
            out["ap"].setdefault(arm, {})[str(k)] = scored[key, k][0]
            out["recall"].setdefault(arm, {})[str(k)] = scored[key, k][1]
    out["distinct_hashes"] = len(scored)
    out["distinct_matrices"] = len(mats)
    return out


def choose_prescale(trials: dict, alphas, ks) -> dict:
    """a_pre(k): the largest mean validation AP; exact ties to the smallest a."""
    out = {}
    for k in ks:
        means = {a: float(np.mean([t["ap"][f"prescale:{a}"][str(k)] for t in trials.values()]))
                 for a in alphas}
        best = max(means.values())
        a = min((a for a, v in means.items() if v == best), key=float)
        out[str(k)] = {"alpha": a, "validation_mean_ap": means,
                       "validation_trials": len(trials)}
    return out


def coincident(allocs: dict, ks, lam, prescale_alphas) -> dict:
    """Arms that hash with the identical matrix and input as even fan-out, and
    as f* at each k (common random numbers make their AP identical)."""
    keys = {name: _key(f) for name, f in allocs.items()}
    grid = [n for n in allocs if n.startswith("grid:")]
    lam = np.asarray(lam, float)
    iso = bool(lam.max() == lam.min())
    pre = [f"prescale:{a}" for a in prescale_alphas if iso or float(a) == 0.0]
    same_even = [g for g in grid if keys[g] == keys["even"]] + pre
    out = {"even": same_even, "isotropic_lambda": iso}
    for k in ks:
        fk = keys[f"fstar:{int(k)}"]
        out[f"fstar@{int(k)}"] = (["even"] + same_even if fk == keys["even"] else
                                  [g for g in grid if keys[g] == fk])
    return out


def summarise(trials: dict, arms, ks) -> dict:
    """Mean AP over test trials per arm and k (descriptive only)."""
    return {arm: {str(k): float(np.mean([t["ap"][arm][str(k)] for t in trials.values()]))
                  for k in ks} for arm in arms}


# ---------------------------------------------------------------- persistence

def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _save(res: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(res), indent=1, allow_nan=False))
    os.replace(tmp, path)


def _sha256_self() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Stop(Exception):
    pass


# ---------------------------------------------------------------- the runner

def _store_allocations(e: dict, allocs: dict, lam, D: dict) -> None:
    e["allocations"] = {name: {"fanout": f.tolist(), "key": _key(f),
                               "cv": float(f.std() / f.mean())} for name, f in allocs.items()}
    e["coincident_arms"] = coincident(allocs, D["ks"], lam, D["prescale_alphas"])


def _trials(e: dict, split: str, z, lam, c, allocs, arms, D: dict, stage: str, name: str,
            n_trials: int, time_left, save, log, choice=None) -> bool:
    """Run the missing trials of one split; False if the budget ran out."""
    store = e.setdefault(split, {}).setdefault("trials", {})
    n = len(z)
    top = int(D["top_fraction"] * n)
    for t in range(n_trials):
        if str(t) in store:
            continue
        if not time_left():
            return False
        t0 = time.time()
        r = run_trial(z, lam, c, allocs, arms, D["ks"], _seed(stage, name, split, t),
                      D["n_queries"], top, choice, D["sweeps"])
        r["seconds"] = time.time() - t0
        store[str(t)] = _clean(r)
        save()
        ap = r["ap"]
        if split == "test":
            msg = "  ".join(
                f"k={k}: even {ap['even'][str(k)]:.4f} f* {ap['fstar'][str(k)]:.4f} "
                f"pre* {ap['prescale_best'][str(k)]:.4f}" for k in D["ks"])
        else:
            msg = "  ".join(f"k={k}: " + " ".join(
                f"{a}:{ap[f'prescale:{a}'][str(k)]:.4f}" for a in D["prescale_alphas"])
                for k in D["ks"][:2])
        log(f"  {stage} {name} {split} trial {t + 1}/{n_trials} ({r['seconds']:.0f}s): {msg}")
    return True


def _finish_config(e: dict, z_test, z_val, lam, c, allocs, D: dict, stage: str, name: str,
                   n_trials: int, n_val: int, time_left, save, log) -> bool:
    """Validation trials, the pre-scaling choice, then the test trials."""
    if "prescale_choice" not in e:
        if not _trials(e, "validation", z_val, lam, c, allocs, validation_arms(D), D, stage,
                       name, n_val, time_left, save, log):
            return False
        e["prescale_choice"] = choose_prescale(e["validation"]["trials"], D["prescale_alphas"],
                                               D["ks"])
        e["prescale_choice_utc"] = _utc()
        save()
        log(f"  {stage} {name}: pre-scaling chosen on validation: " + ", ".join(
            f"k={k} a={v['alpha']}" for k, v in e["prescale_choice"].items()))
    if not _trials(e, "test", z_test, lam, c, allocs, test_arms(D), D, stage, name, n_trials,
                   time_left, save, log, choice=e["prescale_choice"]):
        return False
    if "summary" not in e:
        e["summary"] = {"mean_ap": summarise(e["test"]["trials"], test_arms(D), D["ks"])}
        e["completed_utc"] = _utc()
        save()
    return True


def _stage1(res: dict, D: dict, c: np.ndarray, frozen_path: Path, time_left, save, log) -> bool:
    S = D["stage1"]
    frozen = json.loads(Path(frozen_path).read_text())
    for beta in S["betas"]:
        label = _label(beta)
        e = res["stage1"].setdefault(label, {"beta": float(beta)})
        if "summary" in e:
            continue
        if not time_left():
            return False
        z_test, lam = spc.synthetic(beta, D["n_items"], D["d"], S["item_seed"])
        z_val, _ = spc.synthetic(beta, D["n_items"], D["d"], S["validation_seed"])
        fr = frozen["synthetic"][label]
        if not np.allclose(fr["lambda"], lam, rtol=1e-12, atol=0):
            raise RuntimeError(f"stage 1 beta {label}: lambda differs from the frozen record")
        fs = {int(k): spc.frozen_fstar(beta, int(k), frozen_path) for k in D["ks"]}
        allocs = allocations(lam, c, fs, [float(a) for a in D["grid_alphas"]])
        if "allocations" not in e:
            e.update({"lambda": lam.tolist(), "theta_deg_frozen": fr["theta_deg"],
                      "fstar": {str(k): {"fanout": f.tolist(), "key": _key(f),
                                         **{kk: fr["selection"][str(k)][kk]
                                            for kk in ("alpha_star", "cv", "slope")}}
                                for k, f in fs.items()}})
            _store_allocations(e, allocs, lam, D)
            save()
        log(f"stage 1 beta {label}: alpha* " + ", ".join(
            f"k={k} {e['fstar'][str(k)]['alpha_star']}" for k in D["ks"]))
        if not _finish_config(e, z_test, z_val, lam, c, allocs, D, "stage1", label,
                              S["trials"], S["validation_trials"], time_left, save, log):
            return False
    return True


def _stage2(res: dict, cfg: Config, D: dict, c: np.ndarray, time_left, save, log) -> bool:
    S = D["stage2"]
    F = D["fstar_model"]
    for name in S["datasets"]:
        e = res["stage2"].setdefault(name, {"dataset": name})
        if "summary" in e:
            continue
        if not time_left():
            return False
        t0 = time.time()
        items, train, val, meta = load_splits(cfg, name, D["n_items"], S["n_train"],
                                              S["n_validation"])
        mu, v = pca_fit(train, S["pca_dims"])
        z_test, z_train, z_val = (pca_apply(x, mu, v) for x in (items, train, val))
        lam = spc.spectrum(z_train)
        if "lambda" not in e:
            theta = spc.median_nn_angle(z_train)
            e.update({"splits": meta, "pca_dims": S["pca_dims"], "lambda": lam.tolist(),
                      "theta_deg": theta,
                      "pca_variance_fraction": float(lam.sum() / train.var(0).sum()),
                      "lambda_items": spc.spectrum(z_test).tolist()})
            save()
        else:
            if e["splits"]["split_sha1"] != meta["split_sha1"]:
                raise RuntimeError(f"{name}: the training/validation split changed")
            if not np.allclose(e["lambda"], lam, rtol=1e-8, atol=0):
                raise RuntimeError(f"{name}: Lambda differs from the stored training spectrum")
        log(f"stage 2 {name}: loaded, PCA-{S['pca_dims']} on the training split "
            f"({time.time() - t0:.0f}s), theta {e['theta_deg']:.2f} deg")
        if "fstar" not in e:
            cache = e.setdefault("fstar_evaluations", {})

            def on_eval(key, rec):
                save()
                if not time_left():
                    raise _Stop
            alphas = tuple(math.inf if a == "inf" else float(a) for a in F["alphas"])
            try:
                fs, info = spc.fstar_all(np.array(e["lambda"]), D["ks"], c, D["nnz"],
                                         e["theta_deg"], alphas=alphas, pairs=F["pairs"],
                                         wirings=F["wirings"], seed=F["seed"],
                                         half_width=F["half_width_deg"], cache=cache,
                                         on_eval=on_eval, log=log)
            except _Stop:
                save()
                return False
            e["fstar"] = {str(k): {"fanout": f.tolist(), "key": _key(f),
                                   **{kk: info["k"][str(k)][kk] for kk in
                                      ("alpha_star", "cv", "slope", "se",
                                       "alphas_with_same_allocation")}}
                          for k, f in fs.items()}
            e["fstar_setup"] = info["setup"]
            e["fstar_curve"] = info["curve"]
            e["fstar_utc"] = _utc()
            save()
            log(f"  f* for {name}: " + ", ".join(
                f"k={k} alpha* {e['fstar'][str(k)]['alpha_star']}" for k in D["ks"]))
        fs = {int(k): np.array(e["fstar"][str(k)]["fanout"], int) for k in D["ks"]}
        lam = np.array(e["lambda"])
        allocs = allocations(lam, c, fs, [float(a) for a in D["grid_alphas"]])
        if "allocations" not in e:
            _store_allocations(e, allocs, lam, D)
            save()
        if not _finish_config(e, z_test, z_val, lam, c, allocs, D, "stage2", name,
                              S["trials"], S["validation_trials"], time_left, save, log):
            return False
    return True


def run(cfg: Config, out: Path | str = RESULTS, design: dict | None = None,
        budget_minutes: float | None = None, frozen_path: Path | str = FROZEN,
        log=print) -> dict:
    """Run E4 stages 1 and 2, resuming from `out` if it exists. Stops cleanly
    before starting a new unit (a trial or an f* model evaluation) once
    `budget_minutes` have passed; call again to continue."""
    D = json.loads(json.dumps(design or DESIGN))
    out = Path(out)
    t0 = time.time()
    deadline = None if budget_minutes is None else t0 + 60 * budget_minutes
    frozen = check_frozen(frozen_path)
    fsha = frozen["code"]["sha256"]

    if out.exists():
        res = json.loads(out.read_text())
        if res["design"] != D:
            raise SystemExit(f"{out} was produced with a different design; refusing to mix")
        if res["frozen"]["spectral_sha256"] != fsha:
            raise SystemExit(f"{out} was produced against a different freeze of spectral.py")
        if res.get("complete"):
            log(f"{out} is already complete")
            return res
        if res["code_sha256_at_creation"] != _sha256_self():
            log("  note: flypath/spectral_run.py changed since this file was created "
                "(recorded in invocations)")
        log(f"resuming {out}")
    else:
        res = {"experiment": "E4 stages 1 and 2: spectral fan-out allocation, retrieval",
               "preregistration": "experiments/PREREGISTRATION_2026-09-28.md, E4 stages 1 and 2",
               "note": "per-trial AP@200 and recall@200 of every arm; no intervals or verdicts",
               "design": D, "created_utc": _utc(), "code_sha256_at_creation": _sha256_self(),
               "frozen": {"path": str(Path(frozen_path).resolve().relative_to(ROOT))
                          if Path(frozen_path).resolve().is_relative_to(ROOT)
                          else str(frozen_path),
                          "spectral_sha256": fsha, "frozen_utc": frozen.get("frozen_utc")},
               "complete": False, "invocations": [], "stage1": {}, "stage2": {}}
    res["invocations"].append({"start_utc": _utc(), "code_sha256": _sha256_self(),
                               "spectral_sha256_verified": fsha,
                               "python": platform.python_version(), "numpy": np.__version__})

    def save():
        _save(res, out)

    def time_left() -> bool:
        return deadline is None or time.time() < deadline

    save()
    c = inputs_per_cell(cfg, frozen)
    res["inputs_per_cell"] = {"m": int(len(c)), "nnz": int(c.sum()),
                              "sha1": hashlib.sha1(c.astype(np.int64).tobytes()).hexdigest()}
    finished = (_stage1(res, D, c, Path(frozen_path), time_left, save, log)
                and _stage2(res, cfg, D, c, time_left, save, log))
    res["complete"] = bool(finished)
    res["invocations"][-1]["seconds"] = time.time() - t0
    if finished:
        res["completed_utc"] = _utc()
    save()
    log("E4 stages 1-2 complete" if finished else
        "E4 stages 1-2 incomplete: run the same command again to resume")
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m flypath.spectral_run",
                                description="E4 stages 1 and 2 (pre-registered): AP of f*")
    p.add_argument("--config", help="a YAML config overriding config.example.yaml")
    p.add_argument("--out", default=str(RESULTS), help="results JSON (checkpoint and resume)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny configuration that only checks the code runs; needs --out")
    p.add_argument("--budget-minutes", type=float, default=None,
                   help="stop cleanly before starting a new unit after this many minutes")
    args = p.parse_args(argv)
    if args.smoke and Path(args.out).resolve() == RESULTS.resolve():
        p.error("--smoke must not write to results/e4_spectral.json; pass --out")
    from . import config
    cfg = config.load(args.config)
    res = run(cfg, out=args.out, design=SMOKE if args.smoke else DESIGN,
              budget_minutes=args.budget_minutes, log=lambda s: print(s, flush=True))
    return 0 if res["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
