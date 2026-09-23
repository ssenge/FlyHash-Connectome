"""Every number in the paper is produced here and written to results/*.json.

  primary       measured wiring against B curveball nulls on the baseline
                benchmark: randomization tests, Holm adjustment, shift power,
                two-stage bootstrap interval, pre-specified equivalence margin
  architecture  fly hash against Gaussian sign codes at three budgets
  convergence   curveball mixing diagnostics and a pairing-structure test
  robustness    one-factor-at-a-time sensitivity grid
  coverage      simulation check of the two-stage bootstrap (and of the
                procedure it replaced)

Baseline, fixed before the revised analysis was run:
  right hemisphere, all synapses (no threshold), DoOR glomeruli with >= 40
  odorants and odorants with >= 12 glomeruli, unmeasured entries set to zero,
  4000 Dirichlet mixtures of 2-5 sources (seed 0), raw Euclidean ground truth,
  kappa = 10, random tie-breaking (seed 0) for winner-take-all and retrieval,
  binary projection.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import flyhash as fh
from . import stats
from .config import ROOT, Config
from .data import load_graph

NEIGHBOURS = 10
N_ITEMS = 4000
SECONDARY_SIZES = (8, 16, 32, 64)
SPARSITY = 0.05            # primary hash size: 5% of Kenyon cells, as in Dasgupta et al.
MARGINS = (0.025, 0.05, 0.10)
PRIMARY_MARGIN = 0.05      # pre-specified; see paper, Section III-F


def primary_k(p: fh.Projection) -> int:
    return int(round(SPARSITY * p.matrix.shape[1]))


def sizes_for(p: fh.Projection) -> tuple[int, ...]:
    return tuple(sorted(set(SECONDARY_SIZES) | {primary_k(p)}))


# ---------------------------------------------------------------- building blocks

@dataclass
class Context:
    proj: fh.Projection
    full: fh.Projection
    odours: fh.Odours


@dataclass
class Bench:
    x: np.ndarray
    truth: np.ndarray
    priority: np.ndarray


def context(cfg: Config, side: str = "R", graph_dir: str | None = None,
            missing: str = "zero", min_odours: int = 40,
            min_glomeruli: int = 12, _cache: dict = {}) -> Context:
    key = (side, graph_dir, missing, min_odours, min_glomeruli)
    if key not in _cache:
        gkey = ("graph", graph_dir)
        if gkey not in _cache:
            c = cfg
            if graph_dir:
                d = json.loads(json.dumps(cfg.raw))
                d["data"]["graph_dir"] = graph_dir
                c = Config(d)
            _cache[gkey] = load_graph(c)
        full = fh.mushroom_body(_cache[gkey], side=side)
        od = fh.load_odours(cfg, full.glomeruli, min_glomeruli=min_glomeruli,
                            min_odours=min_odours, missing=missing)
        _cache[key] = Context(fh.align(full, od.glomeruli), full, od)
    return _cache[key]


def bench(x: np.ndarray, metric: str = "euclidean", tie_rule: str = "random",
          tie_seed: int = 0) -> Bench:
    return Bench(x, fh.true_neighbours(x, NEIGHBOURS, metric),
                 fh.retrieval_priority(len(x), tie_seed, tie_rule))


def scores(b: Bench, p: fh.Projection, sizes, tie: str = "random",
           tie_seed: int = 0, weighted: bool = False) -> np.ndarray:
    """mAP of one wiring at every hash size (the drive is computed once)."""
    y = fh.drive(b.x, p, weighted)
    return np.array([fh.mean_average_precision(
        fh.tags(b.x, p, k, weighted, tie, tie_seed, y=y), b.truth, b.priority)
        for k in sizes])


def null_pool(p: fh.Projection, B: int, seed0: int = 10_000,
              sweeps: int = 30) -> list[fh.Projection]:
    """B curveball draws of the binary projection, cached on disk."""
    key = hashlib.sha1(p.binary.astype(np.uint8).tobytes()
                       + f"{B}-{seed0}-{sweeps}".encode()).hexdigest()[:16]
    path = ROOT / "data" / "cache" / f"curveball-{key}.npz"
    if path.exists():
        mats = np.load(path)["m"]
    else:
        mats = np.stack([fh.curveball(fh.Projection("b", p.binary, p.glomeruli),
                                      seed=seed0 + s, sweeps=sweeps).matrix > 0
                         for s in range(B)]).astype(np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, m=mats)
        tmp.rename(path)
    return [fh.Projection("curveball", m.astype(float), p.glomeruli) for m in mats]


def projection_summary(ctx: Context) -> dict:
    p, full = ctx.proj, ctx.full
    return {
        "n_glomeruli_full": int(full.matrix.shape[0]),
        "n_cells_full": int(full.matrix.shape[1]),
        "inputs_mean_full": float(full.inputs().mean()),
        "n_glomeruli": int(p.matrix.shape[0]),
        "n_cells": int(p.matrix.shape[1]),
        "nnz": p.nnz,
        "inputs_mean": float(p.inputs().mean()),
        "inputs_median": float(np.median(p.inputs())),
        "inputs_max": int(p.inputs().max()),
        "pn_partners_mean": float(np.mean(p.meta["pn_partners"])),
        "pn_partners_median": float(np.median(p.meta["pn_partners"])),
        "fan_out_min": int(p.fan_out().min()),
        "fan_out_max": int(p.fan_out().max()),
        "fan_out_max_glomerulus": p.glomeruli[int(np.argmax(p.fan_out()))],
        "n_odorants": int(ctx.odours.x.shape[0]),
        "missing_fraction": ctx.odours.missing_fraction,
        "door_commit": fh.DOOR_COMMIT,
        "primary_k": primary_k(p),
    }


def save(obj: dict, name: str) -> Path:
    out = ROOT / "results" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=1, default=float))
    return out


# ---------------------------------------------------------------- primary analysis

def primary(cfg: Config, B: int = 200, R: int = 200, B_sub: int = 20,
            others: int = 40, log=print) -> dict:
    t0 = time.time()
    ctx = context(cfg)
    p, src = ctx.proj, ctx.odours.x
    sizes = sizes_for(p)
    kP = primary_k(p)
    iP = sizes.index(kP)
    b0 = bench(fh.mixtures(src, N_ITEMS, seed=0))

    real = scores(b0, p, sizes)
    pool = null_pool(p, B)
    log(f"  scoring {B} curveball nulls ...")
    null = np.array([scores(b0, q, sizes) for q in pool])

    alt = {}
    for name, make, n in (("shuffle_inputs", fh.shuffle_inputs, others),
                          ("uniform_inputs_6", lambda q, seed: fh.uniform_inputs(q, seed, 6), others),
                          ("balanced_fanout", fh.balanced_fanout, others),
                          ("degenerate", fh.degenerate, 5)):
        log(f"  scoring {n} x {name} ...")
        s = np.array([scores(b0, make(p, seed=20_000 + i), sizes) for i in range(n)])
        alt[name] = {"mean": s.mean(0).tolist(), "sd": s.std(0, ddof=1).tolist(),
                     "n": n}

    y = fh.drive(b0.x, p)
    rows = []
    for i, k in enumerate(sizes):
        nd = null[:, i]
        mu, sd = float(nd.mean()), float(nd.std(ddof=1))
        inc = fh.tags(b0.x, p, k, tie="include", y=y).sum(1)
        rows.append({
            "k": k, "primary": k == kP,
            "real": float(real[i]), "null_mean": mu, "null_sd": sd,
            "null_se": sd / np.sqrt(B),
            "null_cv": sd / mu,
            "relative_difference": float(real[i] / mu - 1),
            "z": float((real[i] - mu) / sd),
            "randomization": stats.randomization_test(real[i], nd),
            "tie_conflict_fraction": float((inc > k).mean()),
            "alternatives": {a: {"mean": v["mean"][i], "sd": v["sd"][i],
                                 "relative_to_null": v["mean"][i] / mu - 1}
                             for a, v in alt.items()},
        })
    # The primary size is a single pre-specified test; only the secondary
    # sizes are adjusted, among themselves.
    sec = [r for r in rows if not r["primary"]]
    for r, a in zip(sec, stats.holm([r["randomization"]["p_two_sided"] for r in sec])):
        r["randomization"]["p_two_sided_holm"] = a
    for r in rows:
        if r["primary"]:
            r["randomization"]["p_two_sided_holm"] = None

    deltas = [-0.10, -0.075, -0.05, -0.04, -0.03, -0.025, -0.02, -0.015, -0.01,
              -0.005, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10]
    up = stats.shift_power(null[:, iP], [d for d in deltas if d > 0], direction="upper")
    dn = stats.shift_power(null[:, iP], [d for d in deltas if d < 0], direction="lower")
    power = {"upper": up, "lower": dn,
             "mde80_upper": stats.smallest_detectable(up),
             "mde80_lower": stats.smallest_detectable(dn)}

    log(f"  two-stage bootstrap, R={R}, {B_sub} nulls per replicate ...")

    def replicate(rng):
        s = src[rng.integers(0, len(src), len(src))]
        b = bench(fh.mixtures(s, N_ITEMS, seed=int(rng.integers(2**31))),
                  tie_seed=int(rng.integers(2**31)))
        r = scores(b, p, (kP,))[0]
        sub = rng.choice(B, B_sub, replace=False)
        return r, np.array([scores(b, pool[j], (kP,))[0] for j in sub])

    boot = stats.two_stage_bootstrap(replicate, R, seed=77)
    boot["equivalent"] = {f"{m:.3f}": stats.equivalent(boot["relative_ci"], m)
                          for m in MARGINS}
    boot["primary_margin"] = PRIMARY_MARGIN
    res = {"setting": "baseline", "B": B, "sizes": list(sizes), "primary_k": kP,
           "n_items": N_ITEMS, "neighbours": NEIGHBOURS,
           "projection": projection_summary(ctx), "rows": rows,
           "null_scores": null.tolist(), "real_scores": real.tolist(),
           "power": power, "bootstrap": boot,
           "seconds": time.time() - t0}
    save(res, "primary.json")
    return res


# ---------------------------------------------------------------- architecture

def architecture(cfg: Config, seeds: int = 20, log=print) -> dict:
    t0 = time.time()
    ctx = context(cfg)
    p = ctx.proj
    sizes = sizes_for(p)
    x = fh.mixtures(ctx.odours.x, N_ITEMS, seed=0)
    m = p.matrix.shape[1]
    budgets = {"k_bits": {k: k for k in sizes},
               "storage_matched": {k: fh.storage_bits(m, k) for k in sizes},
               "computation_matched": {k: fh.computation_bits(p) for k in sizes}}
    out = {"budgets": {b: {str(k): v for k, v in d.items()} for b, d in budgets.items()},
           "computation_model": "the 2017 accounting: nnz(M) additions for the fly "
                                "expansion; d multiplications plus d additions per "
                                "Gaussian projection",
           "nnz": p.nnz, "d": int(p.matrix.shape[0]), "metrics": {}}
    for metric in ("euclidean", "normalised", "angular"):
        log(f"  architecture, {metric} ground truth ...")
        b = bench(x, metric)
        fly = scores(b, p, sizes)
        rec = {"sizes": list(sizes), "fly": fly.tolist(), "gaussian": {}}
        for name, per_k in budgets.items():
            vals = np.array([[fh.mean_average_precision(
                fh.gaussian_sign(x, per_k[k], seed=s), b.truth, b.priority)
                for k in sizes] for s in range(seeds)])
            rec["gaussian"][name] = {"mean": vals.mean(0).tolist(),
                                     "sd": vals.std(0, ddof=1).tolist(),
                                     "ratio_fly_over": (fly / vals.mean(0)).tolist()}
        out["metrics"][metric] = rec
    out["seconds"] = time.time() - t0
    save(out, "architecture.json")
    return out


# ---------------------------------------------------------------- convergence

def _cooccurrence_q(p: fh.Projection) -> float:
    """Sum of squared off-diagonal glomerulus co-occurrence counts (how often
    two glomeruli share a Kenyon cell). Invariant to the degrees only up to
    pairing, so it tracks the pairing structure the chain randomises."""
    a = p.binary
    c = a @ a.T
    np.fill_diagonal(c, 0)
    return float((c ** 2).sum() / 2)


def _jaccard(p: fh.Projection, q: fh.Projection) -> float:
    a, b = p.matrix > 0, q.matrix > 0
    return float((a & b).sum() / (a | b).sum())


def convergence(cfg: Config, chains: int = 5, log=print) -> dict:
    t0 = time.time()
    ctx = context(cfg)
    p = ctx.proj
    base = fh.Projection("real", p.binary, p.glomeruli)
    kP = primary_k(p)
    b0 = bench(fh.mixtures(ctx.odours.x, N_ITEMS, seed=0))
    marks = (0, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100)
    score_marks = {0, 1, 5, 30, 100}
    traj = []
    for c in range(chains):
        log(f"  chain {c + 1}/{chains} ...")
        snaps = fh.curveball(base, seed=50_000 + c, checkpoints=marks)
        traj.append([{"sweeps": s.meta["sweeps"], "jaccard_to_real": _jaccard(base, s),
                      "cooccurrence_q": _cooccurrence_q(s),
                      "map": (float(scores(b0, s, (kP,))[0])
                              if round(s.meta["sweeps"], 6) in score_marks else None)}
                     for s in snaps])
    pool = null_pool(p, 200)
    q_null = np.array([_cooccurrence_q(q) for q in pool])
    q_real = _cooccurrence_q(base)
    out = {"marks": list(marks), "chains": traj,
           "pool_q_mean": float(q_null.mean()), "pool_q_sd": float(q_null.std(ddof=1)),
           "real_q": q_real,
           "real_q_test": stats.randomization_test(q_real, q_null),
           "pool_jaccard_mean": float(np.mean([_jaccard(base, q) for q in pool[:40]])),
           "seconds": time.time() - t0}
    save(out, "convergence.json")
    return out


# ---------------------------------------------------------------- robustness

CONDITIONS = [
    ("baseline", "baseline", {}),
    ("hemisphere_L", "left hemisphere", {"side": "L"}),
    ("synapse_threshold_5", "connections of >= 5 synapses only", {"graph_dir": "data/graph_w5"}),
    ("missing_glomerulus_mean", "unmeasured -> glomerulus mean", {"missing": "glomerulus_mean"}),
    ("missing_odour_mean", "unmeasured -> odorant mean", {"missing": "odour_mean"}),
    ("missing_lowrank", "unmeasured -> rank-5 SVD imputation", {"missing": "lowrank"}),
    ("coverage_strict", "glomeruli >= 80 odorants, odorants >= 20 glomeruli",
     {"min_odours": 80, "min_glomeruli": 20}),
    ("coverage_loose", "glomeruli >= 20 odorants, odorants >= 12 glomeruli",
     {"min_odours": 20, "min_glomeruli": 12}),
    ("all_glomeruli", "every glomerulus with DoOR data (46 of 51), unmeasured -> rank-5 imputation",
     {"min_odours": 1, "missing": "lowrank"}),
    ("mixture_seed_1", "mixture seed 1", {"mix_seed": 1}),
    ("mixture_seed_2", "mixture seed 2", {"mix_seed": 2}),
    ("mixture_seed_3", "mixture seed 3", {"mix_seed": 3}),
    ("mixture_seed_4", "mixture seed 4", {"mix_seed": 4}),
    ("mixtures_pairs", "mixtures of exactly 2 sources", {"lo": 2, "hi": 2}),
    ("mixtures_wide", "mixtures of 2-10 sources", {"lo": 2, "hi": 10}),
    ("measured_only", "measured odorants only, no mixtures", {"items": "odours"}),
    ("metric_normalised", "ground truth: Euclidean on normalised input", {"metric": "normalised"}),
    ("metric_angular", "ground truth: angular distance", {"metric": "angular"}),
    ("ties_index", "ties broken by index (WTA and retrieval)", {"tie": "index"}),
    ("ties_include", "WTA includes all tied cells", {"tie": "include"}),
    ("tie_seed_1", "random tie-breaking, seed 1", {"tie_seed": 1}),
    ("tie_seed_2", "random tie-breaking, seed 2", {"tie_seed": 2}),
    ("weighted", "synapse-weighted projection", {"weighted": True}),
    ("synthetic_lowrank", "non-biological: rank-6 Gaussian inputs", {"items": "synthetic_lowrank"}),
    ("synthetic_gaussian", "non-biological: |N(0,1)| inputs", {"items": "synthetic_gaussian"}),
    ("synthetic_sparse", "non-biological: sparse uniform inputs", {"items": "synthetic_sparse"}),
]


def _items(kind: str, src: np.ndarray, seed: int, lo: int, hi: int) -> np.ndarray:
    d = src.shape[1]
    rng = np.random.default_rng(900 + seed)
    if kind == "mixtures":
        return fh.mixtures(src, N_ITEMS, seed=seed, lo=lo, hi=hi)
    if kind == "odours":
        return src
    if kind == "synthetic_lowrank":
        return np.abs(rng.normal(size=(N_ITEMS, 6)) @ rng.normal(size=(6, d)))
    if kind == "synthetic_gaussian":
        return np.abs(rng.normal(size=(N_ITEMS, d)))
    if kind == "synthetic_sparse":
        return (rng.random((N_ITEMS, d)) < 0.2) * rng.random((N_ITEMS, d))
    raise ValueError(kind)


def robustness(cfg: Config, B: int = 40, only: list[str] | None = None, log=print) -> dict:
    """With `only`, the listed conditions are recomputed and merged into the
    existing results/robustness.json; the other rows are kept."""
    t0 = time.time()
    path = ROOT / "results" / "robustness.json"
    kept = json.loads(path.read_text())["rows"] if only and path.exists() else []
    kept = [r for r in kept if r["id"] not in only] if only else []
    rows = []
    for cid, label, opt in CONDITIONS:
        if only and cid not in only:
            continue
        ts = time.time()
        ctx = context(cfg, side=opt.get("side", "R"), graph_dir=opt.get("graph_dir"),
                      missing=opt.get("missing", "zero"),
                      min_odours=opt.get("min_odours", 40),
                      min_glomeruli=opt.get("min_glomeruli", 12))
        p = ctx.proj
        sizes = sizes_for(p)
        x = _items(opt.get("items", "mixtures"), ctx.odours.x, opt.get("mix_seed", 0),
                   opt.get("lo", 2), opt.get("hi", 5))
        tie = opt.get("tie", "random")
        b = bench(x, opt.get("metric", "euclidean"),
                  tie_rule="index" if tie == "index" else "random",
                  tie_seed=opt.get("tie_seed", 0))
        weighted = opt.get("weighted", False)
        kw = dict(tie="index" if tie == "index" else tie,
                  tie_seed=opt.get("tie_seed", 0), weighted=weighted)
        real = scores(b, p, sizes, **kw)
        if weighted:
            pool = [fh.curveball(p, seed=10_000 + s) for s in range(B)]
        else:
            pool = null_pool(p, B)
        null = np.array([scores(b, q, sizes, **kw) for q in pool])
        per_k = []
        for i, k in enumerate(sizes):
            rt = stats.randomization_test(real[i], null[:, i])
            per_k.append({"k": k, "primary": k == primary_k(p), "real": float(real[i]),
                          "null_mean": float(null[:, i].mean()),
                          "null_sd": float(null[:, i].std(ddof=1)),
                          "relative_difference": float(real[i] / null[:, i].mean() - 1),
                          "p_two_sided": rt["p_two_sided"], "p_lower": rt["p_lower"],
                          "p_upper": rt["p_upper"]})
        rows.append({"id": cid, "label": label, "options": opt,
                     "n_items": int(len(x)), "n_glomeruli": int(p.matrix.shape[0]),
                     "n_cells": int(p.matrix.shape[1]),
                     "inputs_mean": float(p.inputs().mean()),
                     "missing_fraction": ctx.odours.missing_fraction,
                     "per_k": per_k, "seconds": time.time() - ts})
        log(f"  {cid:24s} primary k: rel {per_k[sizes.index(primary_k(p))]['relative_difference']:+.4f}"
            f"  p {per_k[sizes.index(primary_k(p))]['p_two_sided']:.3f}   ({time.time() - ts:.0f}s)")
        order = [c[0] for c in CONDITIONS]
        merged = sorted(kept + rows, key=lambda r: order.index(r["id"]))
        save({"B": B, "rows": merged, "seconds": time.time() - t0}, "robustness.json")
    return {"B": B, "rows": rows}


# ---------------------------------------------------------------- coverage

def _dgp(src: np.ndarray, rank: int = 6):
    """A generative model fitted to the source profiles, used as a known
    population: rank-`rank` Gaussian latent structure plus isotropic residual
    noise, clipped at zero."""
    mu = src.mean(0)
    u, s, vt = np.linalg.svd(src - mu, full_matrices=False)
    load = (s[:rank, None] * vt[:rank]) / np.sqrt(len(src) - 1)
    resid = (src - mu) - (u[:, :rank] * s[:rank]) @ vt[:rank]
    sigma = float(resid.std())

    def draw(rng, n):
        while True:
            z = mu + rng.normal(size=(n, rank)) @ load + sigma * rng.normal(size=(n, src.shape[1]))
            z = np.clip(z, 0, None)
            if (z.sum(1) > 0).all():
                return z
    return draw


def coverage(cfg: Config, datasets: int = 100, R: int = 100, B_sub: int = 10,
             truth_sets: int = 300, n_items: int = 600, log=print) -> dict:
    """Does the two-stage bootstrap cover the estimand at its nominal level?

    Uses a known population (`_dgp`) so the true relative difference can be
    computed by Monte Carlo. Same wiring, null pool, hash size and generator as
    the primary analysis, at a reduced benchmark size to keep the run to about
    half an hour. The superseded group bootstrap is evaluated on the same
    datasets.
    """
    t0 = time.time()
    ctx = context(cfg)
    p = ctx.proj
    kP = primary_k(p)
    pool = null_pool(p, 200)
    draw = _dgp(ctx.odours.x)
    n_src = len(ctx.odours.x)
    rng = np.random.default_rng(424242)

    def one(src, mix_seed, tie_seed, subset):
        x = fh.mixtures(src, n_items, seed=mix_seed)
        b = bench(x, tie_seed=tie_seed)
        r = scores(b, p, (kP,))[0]
        return r, np.array([scores(b, pool[j], (kP,))[0] for j in subset]), b, x

    log("  estimating the true value ...")
    num = den = 0.0
    for _ in range(truth_sets):
        src = draw(rng, n_src)
        r, nl, _, _ = one(src, int(rng.integers(2**31)), int(rng.integers(2**31)),
                          rng.choice(len(pool), 20, replace=False))
        num += r - nl.mean()
        den += nl.mean()
    theta = num / den

    new_cov = old_cov = 0
    widths_new, widths_old = [], []
    for d in range(datasets):
        src = draw(rng, n_src)
        mseed = int(rng.integers(2**31))

        def replicate(r_rng, src=src):
            s = src[r_rng.integers(0, n_src, n_src)]
            r, nl, _, _ = one(s, int(r_rng.integers(2**31)), int(r_rng.integers(2**31)),
                              r_rng.choice(len(pool), B_sub, replace=False))
            return r, nl

        ci = stats.two_stage_bootstrap(replicate, R, seed=int(rng.integers(2**31)))["relative_ci"]
        new_cov += ci[0] <= theta <= ci[1]
        widths_new.append(ci[1] - ci[0])

        # the superseded procedure on the same dataset
        x = fh.mixtures(src, n_items, seed=mseed)
        b = bench(x)
        real_ap = fh.mean_average_precision(fh.tags(x, p, kP), b.truth, b.priority, per_item=True)
        sub = rng.choice(len(pool), 20, replace=False)
        null_ap = np.array([fh.mean_average_precision(fh.tags(x, pool[j], kP), b.truth,
                                                      b.priority, per_item=True) for j in sub])
        cl = _dominant_component(src, n_items, mseed)
        oci = stats.legacy_group_bootstrap(real_ap, null_ap, cl, draws=1000, seed=d)
        old_cov += oci[0] <= theta <= oci[1]
        widths_old.append(oci[1] - oci[0])
        if (d + 1) % 10 == 0:
            log(f"  {d + 1}/{datasets}: new {new_cov}/{d + 1}, superseded {old_cov}/{d + 1}")

    def wilson(k, n, z=1.96):
        ph = k / n
        c = (ph + z * z / (2 * n)) / (1 + z * z / n)
        h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return [float(c - h), float(c + h)]

    out = {"theta_relative": float(theta), "nominal": 0.90, "datasets": datasets,
           "R": R, "B_sub": B_sub, "n_items": n_items, "k": kP,
           "two_stage": {"coverage": new_cov / datasets, "wilson95": wilson(new_cov, datasets),
                         "median_width": float(np.median(widths_new))},
           "superseded_group_bootstrap": {"coverage": old_cov / datasets,
                                          "wilson95": wilson(old_cov, datasets),
                                          "median_width": float(np.median(widths_old))},
           "seconds": time.time() - t0}
    save(out, "coverage.json")
    return out


def _dominant_component(src: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Reproduce `fh.mixtures`' draws to recover each item's largest-weight
    source, which is how the superseded procedure grouped items."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n, np.int64)
    for i in range(n):
        k = int(rng.integers(2, 6))
        pick = rng.choice(len(src), k, replace=False)
        w = rng.dirichlet(np.ones(k))
        out[i] = pick[int(np.argmax(w))]
    return out
