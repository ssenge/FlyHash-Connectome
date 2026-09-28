"""F2 of the pre-registration: larger null ensembles for the odour endpoint.

Pre-registration: experiments/PREREGISTRATION_2026-09-28.md, section 14, "F2.
Larger null ensembles (paper revision)":

    Odour primary endpoint for all 7 hemispheres with B = 2,000 curveball
    nulls (p floor 0.001), Holm across the 7. Replaces the B = 100 results in
    the paper whatever the outcome.

The endpoint is the "odours" row at the primary size of
`connectomes._odour_test` as `connectomes.compare` calls it, unchanged except
for the number of nulls:

  projection  `connectomes.projection` of the hemisphere, restricted with
              `flyhash.align` to the DoOR glomeruli that connectome has
              (`flyhash.load_odours` defaults), binary
  benchmark   `experiments.bench` on 4,000 mixtures of the DoOR odorants
              (`flyhash.mixtures`, seed 0): raw Euclidean truth, kappa = 10,
              random retrieval tie-breaking (seed 0)
  score       mAP at kappa = 10 (AP@10) via `experiments.scores`, binary
              drive, random WTA tie-breaking (seed 0)
  size        k = `experiments.primary_k` (5% of the projection's cells)
  nulls       B = 2,000 curveball draws of the binary projection
              (`flyhash.curveball`, 30 sweeps, seeds 10,000 + b), made exactly
              as `experiments.null_pool` makes them
  test        `stats.randomization_test`, two-sided; Holm (`stats.holm`) over
              the seven hemispheres

Run (repeat until it prints "complete"; every call checkpoints and stops
before its time budget, so each fits in a 10-minute shell call):

    VECLIB_MAXIMUM_THREADS=3 PYTHONPATH=. python -m flypath.f2_audit

Writes results/f2_nulls.json once all seven hemispheres are done. A subset
(--only) or another B (--nulls) needs an explicit --out. Checkpoints live in data/cache/f2_audit/ (gitignored).

Unspecified details fixed before running
----------------------------------------
Fixed on 2026-09-28, before any full-run result was seen. Only a timing probe
and a smoke run preceded them: B = 100 (chunks of 50) on MaleCNS R and FlyWire
R into a scratch file, which reproduced the paper's B = 100 numbers from
results/connectomes.json exactly (real score, null mean, null SD, p) and
confirmed that the draws equal the cached B = 100 `null_pool` ensembles.

1. Null seeds. Null b (b = 0 .. 1,999) is `flyhash.curveball` of the binary
   odour projection with seed 10,000 + b and 30 sweeps, exactly the draw
   `experiments.null_pool(po, B)` makes (its default seed0 and sweeps). The
   ensembles are therefore nested: nulls 0..99 are the B = 100 nulls behind
   the paper's current numbers. This is checked, not assumed: the first 100
   matrices are compared with the cached B = 100 pool, and the real score and
   the first-100 null mean and p are compared with results/connectomes.json.
2. Draws are made and scored in chunks of 100 so the run can checkpoint.
   When a hemisphere is complete, the 2,000 matrices are written to the
   `experiments.null_pool` cache in its own format and key (B = 2000,
   seed0 = 10000, sweeps = 30), then read back through `experiments.null_pool`
   and compared with the chunks, so `null_pool(po, 2000)` returns this
   ensemble afterwards.
3. Only the primary size is scored (k = `experiments.primary_k(po)`: 92, 91,
   86, 118, 115, 98, 101). WTA tags at one k do not depend on which other
   sizes are scored, so the real score equals the B = 100 run's.
4. Only the "odours" endpoint (the hemisphere's own DoOR glomeruli) is rerun.
   "odours_matched", the six-input and equal-connection controls (10 draws
   each, not tested against the null), the Q structure test and the 2017
   protocol are not part of F2 and are not rerun.
5. Summary statistics as in `_odour_test`: null mean, null SD (ddof = 1),
   relative difference real / null mean - 1, two-sided p =
   min(1, 2 min(p_lower, p_upper)) with the +1 correction and ties counted
   against the claim. With B = 2,000 the smallest attainable two-sided p is
   2 / 2,001 = 0.0009995 (the pre-registered "p floor 0.001"). One-sided p,
   rank, z = (real - null mean) / null SD and the Monte Carlo SE of the null
   mean (SD / sqrt(B)) are reported descriptively.
6. Holm family: the seven primary-size two-sided p-values, one per
   hemisphere, in `connectomes.HEMISPHERES` order. A hemisphere counts as
   significant if its Holm-adjusted p < 0.05 (the paper's `< 0.05`
   convention). Holm is valid under any dependence, which matters because
   hemispheres of one animal are not independent.
7. Nothing is decided by F2 beyond reporting: the pre-registration makes these
   numbers replace the B = 100 odour results whatever they are.

Deviations
----------
None.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import connectomes as C
from . import experiments as ex
from . import flyhash as fh
from . import stats

B = 2000
SEED0 = 10_000              # experiments.null_pool default
SWEEPS = 30                 # experiments.null_pool default
CHUNK = 100
MIX_SEED = 0
ALPHA = 0.05
BUDGET = 540.0              # seconds per call; leaves margin under a 10-minute shell limit
FIRST_CHUNK_GUESS = 150.0   # seconds assumed for a chunk before one has been timed
PREREG = "experiments/PREREGISTRATION_2026-09-28.md, section 14, F2"


def default_checkpoint_dir() -> Path:
    return ex.ROOT / "data" / "cache" / "f2_audit"


def default_out() -> Path:
    return ex.ROOT / "results" / "f2_nulls.json"


def keys() -> list[str]:
    return [f"{ds}_{sd}" for ds, sd in C.HEMISPHERES]


# ---------------------------------------------------------------- building blocks

def odour_projection(cfg, dataset: str, side: str) -> tuple[fh.Projection, np.ndarray]:
    """The binary odour projection and DoOR responses, as in `connectomes.compare`."""
    p = C.projection(cfg, dataset, side)
    od = fh.load_odours(cfg, p.glomeruli)
    po_w = fh.align(p, od.glomeruli)
    return fh.Projection("b", po_w.binary, po_w.glomeruli, meta=po_w.meta), od.x


def draw(po: fh.Projection, b: int, seed0: int = SEED0, sweeps: int = SWEEPS) -> np.ndarray:
    """Null b, exactly as `experiments.null_pool` makes it (0/1 uint8)."""
    q = fh.curveball(fh.Projection("b", po.binary, po.glomeruli), seed=seed0 + b, sweeps=sweeps)
    return (q.matrix > 0).astype(np.uint8)


def pool_cache_path(po: fh.Projection, n: int, seed0: int = SEED0, sweeps: int = SWEEPS) -> Path:
    """The file `experiments.null_pool(po, n, seed0, sweeps)` reads and writes."""
    key = hashlib.sha1(po.binary.astype(np.uint8).tobytes()
                       + f"{n}-{seed0}-{sweeps}".encode()).hexdigest()[:16]
    return ex.ROOT / "data" / "cache" / f"curveball-{key}.npz"


def write_pool_cache(po: fh.Projection, mats: np.ndarray, seed0: int = SEED0,
                     sweeps: int = SWEEPS) -> Path:
    """Store `mats` where `experiments.null_pool(po, len(mats))` will find them."""
    path = pool_cache_path(po, len(mats), seed0, sweeps)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, m=mats.astype(np.uint8))
        tmp.rename(path)
    return path


def summarise(real: float, null: np.ndarray) -> dict:
    null = np.asarray(null, float)
    n = len(null)
    rt = stats.randomization_test(real, null)
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    return {"B": n, "real": float(real), "null_mean": mu, "null_sd": sd,
            "null_se": sd / math.sqrt(n), "relative_difference": float(real / mu - 1),
            "z": float((real - mu) / sd), "p_two_sided": rt["p_two_sided"],
            "p_lower": rt["p_lower"], "p_upper": rt["p_upper"],
            "rank_from_top": rt["rank_from_top"], "p_floor": 2 / (n + 1)}


def holm_family(records: dict[str, dict], alpha: float = ALPHA) -> dict:
    ks = list(records)
    adj = stats.holm([records[k]["p_two_sided"] for k in ks])
    return {"family": ks, "alpha": alpha,
            "p_holm": dict(zip(ks, adj)),
            "significant": {k: bool(a < alpha) for k, a in zip(ks, adj)},
            "n_significant": int(sum(a < alpha for a in adj)),
            "min_p_holm": float(min(adj))}


def _save_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name[:-4] + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)


def _write_json(path: Path, obj) -> None:
    """Atomic: a call killed mid-write must not leave a truncated file that
    every later (resumed) call would fail to parse."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def _setup(cfg, key: str, cdir: Path) -> tuple[fh.Projection, np.ndarray]:
    """The odour projection and responses, cached so resumed calls skip the
    connectome loaders."""
    path = cdir / f"{key}-setup.npz"
    if path.exists():
        z = np.load(path)
        return fh.Projection("b", z["binary"].astype(np.float64),
                             [str(g) for g in z["glomeruli"]]), z["x"]
    dataset, side = key.rsplit("_", 1)
    po, x = odour_projection(cfg, dataset, side)
    _save_npz(path, binary=po.binary.astype(np.uint8), glomeruli=np.array(po.glomeruli), x=x)
    return po, x


def _b100_reference(key: str) -> dict | None:
    path = ex.ROOT / "results" / "connectomes.json"
    if not path.exists():
        return None
    h = json.loads(path.read_text()).get("hemispheres", {}).get(key)
    if h is None:
        return None
    r = next(r for r in h["odours"] if r["primary"])
    return {"k": r["k"], "real": r["real"], "null_mean": r["null_mean"],
            "null_sd": r["null_sd"], "relative_difference": r["relative_difference"],
            "p_two_sided": r["p_two_sided"]}


def advance(key: str, po: fh.Projection, x: np.ndarray, cdir: Path, n_null: int = B,
            chunk: int = CHUNK, deadline: float = math.inf, pool_cache: bool = True,
            log=print) -> dict | None:
    """Draw and score this hemisphere's nulls chunk by chunk until all are
    done or the next chunk would pass `deadline`. Returns the finished record,
    or None if stopped early (call again to resume)."""
    done = cdir / f"{key}-B{n_null}.json"
    if done.exists():
        return json.loads(done.read_text())
    kP = ex.primary_k(po)
    b0 = ex.bench(fh.mixtures(x, ex.N_ITEMS, seed=MIX_SEED))
    n_chunks = -(-n_null // chunk)
    paths = [cdir / f"{key}-B{n_null}-c{chunk}-{c:04d}.npz" for c in range(n_chunks)]
    est = FIRST_CHUNK_GUESS
    for c, path in enumerate(paths):
        if path.exists():
            continue
        if time.time() + est > deadline:
            return None
        t = time.time()
        lo, hi = c * chunk, min(n_null, (c + 1) * chunk)
        mats = np.stack([draw(po, b) for b in range(lo, hi)])
        sc = np.array([ex.scores(b0, fh.Projection("curveball", m.astype(float), po.glomeruli),
                                 (kP,))[0] for m in mats])
        sec = time.time() - t
        _save_npz(path, m=mats, s=sc, seeds=np.arange(lo, hi) + SEED0, sec=np.array(sec))
        est = 1.25 * sec
        log(f"  {key}: nulls {hi}/{n_null} ({sec:.0f}s)")

    if time.time() + est > deadline:          # finalising reads and writes every matrix
        return None
    t = time.time()
    zs = [np.load(p) for p in paths]
    mats = np.concatenate([z["m"] for z in zs])
    null = np.concatenate([z["s"] for z in zs])
    compute = float(sum(float(z["sec"]) for z in zs))
    real = float(ex.scores(b0, po, (kP,))[0])
    checks = {}
    if pool_cache:
        cpath = write_pool_cache(po, mats)
        pool = ex.null_pool(po, n_null)
        checks["pool_cache"] = str(cpath.relative_to(ex.ROOT))
        checks["pool_cache_equals_chunks"] = bool(
            len(pool) == len(mats) and all(np.array_equal(q.matrix > 0, m > 0)
                                           for q, m in zip(pool, mats)))
        del pool
    p100 = pool_cache_path(po, 100)
    if n_null >= 100 and p100.exists():
        checks["first_100_equal_B100_pool"] = bool(np.array_equal(np.load(p100)["m"], mats[:100]))
    ref = _b100_reference(key)
    first = summarise(real, null[:100]) if n_null >= 100 else None
    if ref is not None and first is not None:
        checks["b100_reproduced"] = bool(
            ref["k"] == kP and abs(ref["real"] - real) < 1e-12
            and abs(ref["null_mean"] - first["null_mean"]) < 1e-12
            and ref["p_two_sided"] == first["p_two_sided"])
    rec = {"key": key, "k": kP, "n_glomeruli": int(po.matrix.shape[0]),
           "n_cells": int(po.matrix.shape[1]), "nnz": po.nnz, "n_odorants": int(len(x)),
           **summarise(real, null),
           "b100_paper": ref,
           "b100_recomputed_first_100": first,
           "checks": checks, "seeds": [SEED0, SEED0 + n_null - 1], "sweeps": SWEEPS,
           "seconds": compute + time.time() - t,
           "null_scores": null.tolist()}
    failed = [c for c, v in checks.items() if v is False]
    if failed:
        log(f"  {key}: WARNING, check failed: {', '.join(failed)}")
    _write_json(done, rec)
    if pool_cache and checks.get("pool_cache_equals_chunks"):
        for p in paths:                        # the matrices now live in the null_pool cache
            p.unlink()
    log(f"  {key}: real {real:.4f}, null {rec['null_mean']:.4f} +- {rec['null_sd']:.4f}, "
        f"rel {100 * rec['relative_difference']:+.2f}%, p {rec['p_two_sided']:.4f}")
    return rec


# ---------------------------------------------------------------- runner

def run(cfg=None, n_null: int = B, chunk: int = CHUNK, budget: float = BUDGET,
        only: list[str] | None = None, out: str | Path | None = None,
        checkpoint_dir: str | Path | None = None, pool_cache: bool = True,
        log=print) -> dict:
    """Advance the audit within `budget` seconds; write `out` when every
    requested hemisphere is done. Returns {"complete": bool, ...}.

    The default `out` (results/f2_nulls.json) holds the pre-registered run
    only: all seven hemispheres at n_null = B. A subset or a different B needs
    an explicit `out`, so a smoke run cannot overwrite the F2 result."""
    unknown = sorted(set(only or ()) - set(keys()))
    if unknown:
        raise ValueError(f"unknown hemispheres {unknown}; choose from {keys()}")
    wanted = [k for k in keys() if not only or k in only]
    if out is None and (len(wanted) != len(keys()) or n_null != B):
        raise ValueError(f"a run on {len(wanted)} of {len(keys())} hemispheres with B = {n_null} "
                         f"needs an explicit out; {default_out().name} holds only the "
                         f"pre-registered run (all seven, B = {B})")
    if cfg is None:
        from . import config
        cfg = config.load()
    t0 = time.time()
    deadline = t0 + budget
    cdir = Path(checkpoint_dir) if checkpoint_dir else default_checkpoint_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    records = {}
    for key in wanted:
        done = cdir / f"{key}-B{n_null}.json"
        if done.exists():
            records[key] = json.loads(done.read_text())
            continue
        # loading a connectome from scratch can take a few minutes
        if time.time() + (30 if (cdir / f"{key}-setup.npz").exists() else 180) > deadline:
            break
        po, x = _setup(cfg, key, cdir)
        rec = advance(key, po, x, cdir, n_null, chunk, deadline, pool_cache, log)
        if rec is None:
            break
        records[key] = rec
    complete = len(records) == len(wanted)
    status = {"complete": complete, "done": list(records), "pending": [k for k in wanted if k not in records],
              "seconds": time.time() - t0}
    if not complete:
        log(f"incomplete: {len(records)}/{len(wanted)} hemispheres done; run again")
        return status

    result = {
        "experiment": "F2 larger null ensembles: odour primary endpoint, every hemisphere",
        "preregistration": PREREG,
        "B": n_null, "seed0": SEED0, "sweeps": SWEEPS, "chunk": chunk,
        "n_items": ex.N_ITEMS, "neighbours": ex.NEIGHBOURS, "mixture_seed": MIX_SEED,
        "score": "mAP at kappa = 10 (AP@10), experiments.scores, binary drive, random ties seed 0",
        "test": "stats.randomization_test, two-sided, +1 correction",
        "p_floor": 2 / (n_null + 1),
        "hemispheres": records,
        "holm": holm_family({k: records[k] for k in wanted}) if len(wanted) == len(keys()) else None,
        "seconds": float(sum(r["seconds"] for r in records.values())),
    }
    path = Path(out) if out else default_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, result)
    log(f"complete: wrote {path}")
    if result["holm"]:
        for k in wanted:
            r = records[k]
            log(f"  {k:12s} k {r['k']:3d}  rel {100 * r['relative_difference']:+6.2f}%  "
                f"p {r['p_two_sided']:.4f}  Holm {result['holm']['p_holm'][k]:.4f}")
    status["out"] = str(path)
    return status


def main(argv: list[str] | None = None) -> int:
    a = argparse.ArgumentParser(prog="python -m flypath.f2_audit", description=__doc__.splitlines()[0])
    a.add_argument("--nulls", type=int, default=B)
    a.add_argument("--chunk", type=int, default=CHUNK)
    a.add_argument("--budget", type=float, default=BUDGET, help="seconds for this call")
    a.add_argument("--only", help="comma-separated hemispheres, e.g. malecns_R,banc_L")
    a.add_argument("--out", help="output JSON (default results/f2_nulls.json)")
    a.add_argument("--checkpoint-dir", help="default data/cache/f2_audit")
    a.add_argument("--no-pool-cache", action="store_true",
                   help="do not write the matrices to the experiments.null_pool cache")
    a.add_argument("--config")
    args = a.parse_args(argv)
    from . import config
    status = run(config.load(args.config), n_null=args.nulls, chunk=args.chunk,
                 budget=args.budget,
                 only=[s.strip() for s in args.only.split(",") if s.strip()] if args.only else None,
                 out=args.out, checkpoint_dir=args.checkpoint_dir,
                 pool_cache=not args.no_pool_cache)
    print("complete" if status["complete"] else "incomplete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
