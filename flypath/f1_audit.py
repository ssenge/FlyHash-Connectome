"""F1 weak-connection audit (pre-registration 2026-09-28, section F1).

Pre-registered (experiments/PREREGISTRATION_2026-09-28.md, F1, frozen):

  - Q z-score of every hemisphere against >= 100 curveball nulls at
    connection-weight thresholds w >= 1, 2, 3, 5 synapses; descriptive.
  - Fixes the paper's hemibrain statement if its Q departs from null at w >= 2.

For each of the 7 hemispheres (`connectomes.HEMISPHERES`) the synapse-weighted
glomerulus x Kenyon-cell matrix from `connectomes.projection` is binarised at
each threshold w, cells left without input are dropped, and the glomerulus
co-occurrence statistic Q (`experiments._cooccurrence_q`) of the measured matrix
is compared with Q of B curveball draws (`flyhash.curveball`) of that same
thresholded binary matrix. Reported per hemisphere and threshold: matrix size,
nnz, inputs per cell, Q, the null mean and SD, z, and the one-sided upper
randomization p = (1 + #{null >= real}) / (B + 1). Output: results/f1_audit.json.

Unspecified details fixed before running
----------------------------------------
Fixed on 2026-09-28, after reading the pre-registration, the code and the
existing w >= 1 results in results/connectomes.json (B = 100, already in the
paper), and before any thresholded Q or null draw was computed.

1. Threshold unit. w applies to the glomerulus-level weight M[j, c] built by
   `connectomes.projection`: synapses summed over every uniglomerular
   projection neuron of glomerulus j onto Kenyon cell c. The binary matrix is
   M >= w. w >= 1 is therefore exactly the paper's binary matrix. (This is not
   the neuron-level threshold of the robustness condition
   `synapse_threshold_5`, which uses data/graph_w5.)
2. Glomerulus set. Every olfactory glomerulus of the hemisphere's projection,
   as in the structure test of `connectomes.compare` (not restricted to DoOR
   glomeruli). A glomerulus left with no connection at a threshold is kept as
   an empty row. Empty rows add 0 to Q and curveball trades never change them.
   The number of innervated glomeruli is reported.
3. Cells. Cells left with no connection >= w are dropped, as the task says.
   Cells with one input are kept: they add 0 to Q and still take part in
   curveball trades.
4. Null ensemble size. B = 1000 per hemisphere and threshold. The
   pre-registration requires at least 100. 1000 draws give a p floor of about
   0.001, the order of F2's floor, and a precise null SD for z.
5. Null seeds. Draw b uses seed 10_000 + b (b = 0 .. B-1) and 30 sweeps, for
   every hemisphere and threshold. This is the seeding of
   `experiments.null_pool` (`fh.curveball(Projection("b", binary), seed=10_000
   + b, sweeps=30)`), so at w >= 1 the first 100 draws are the null behind
   results/connectomes.json. Q and the first-100 null mean and SD at w >= 1 are
   checked against that file (`reproduction_check`). The draws are made by
   calling `fh.curveball` directly instead of `ex.null_pool`, so nothing is
   written to data/cache. The draws are identical.
6. Statistics. z = (Q_real - mean(Q_null)) / SD(Q_null, ddof=1), as in
   `connectomes.compare`. p_upper comes from `stats.randomization_test` (ties
   count against the claim). p_lower and the two-sided p are recorded and are
   not used by any rule. The relative difference Q_real / mean(Q_null) - 1 is
   also recorded. Holm across the 7 hemispheres within each threshold
   (`stats.holm` on p_upper) is reported descriptively, mirroring the paper's
   AnimQHolm.
7. "Departs from null" (hemibrain statement). The hemibrain's Q departs from
   its null at w >= 2 iff its one-sided upper p < 0.05 on the w >= 2 matrix.
   This is the rule the paper already uses to count departing hemispheres
   (`report.py`, AnimQNSig: p_upper < 0.05). The result at w >= 3 and w >= 5
   and a significant lower tail (p_lower < 0.05) are reported alongside and do
   not change the rule's outcome. The output records whether the paper's
   sentence "the hemibrain is the exception, indistinguishable from its null by
   Q" needs revision. The paper is not edited here.
8. Descriptives per threshold: number of glomeruli (rows) and innervated
   glomeruli, cells kept and dropped, nnz, mean, median and max inputs per
   cell, cells with a single input, fan-out CV over all rows (as in
   `connectomes.summary`), and the fraction of w >= 1 connections and of
   synapses retained.
9. Execution. Null draws run in worker processes (3 by default). Each draw
   depends only on its seed, so results do not depend on the number of
   workers or on where a run was interrupted. The runner checkpoints its null
   Q values into the output JSON after every chunk and resumes from it. The
   SHA-1 of each thresholded binary matrix is stored, and resuming against a
   different matrix raises. (Added in code review after the full run; it
   changes no stored number:) resuming also raises if the checkpoint holds a
   hemisphere that was not requested, so a rerun with --only cannot shrink the
   Holm family in place, and a resumed file is marked incomplete, without the
   derived table and hemibrain statement, until the run finishes.

Deviations
----------
None.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import connectomes as C
from . import experiments as ex
from . import flyhash as fh
from . import stats
from .config import ROOT

THRESHOLDS = (1, 2, 3, 5)
B_DEFAULT = 1000
SEED0 = 10_000
SWEEPS = 30
DEPART_ALPHA = 0.05
STATEMENT_THRESHOLD = 2
OUT = ROOT / "results" / "f1_audit.json"
PREREG = "experiments/PREREGISTRATION_2026-09-28.md, section F1"


# ---------------------------------------------------------------- building blocks

def threshold(p: fh.Projection, w: int) -> fh.Projection:
    """Binary matrix of glomerulus-level weights >= w, cells without input dropped."""
    b = (np.asarray(p.matrix) >= w)
    keep = b.sum(0) > 0
    return fh.Projection(f"w>={w}", b[:, keep].astype(np.float64), list(p.glomeruli),
                         meta={"threshold": w, "cells_kept": keep})


def digest(b: fh.Projection) -> str:
    return hashlib.sha1(b.binary.astype(np.uint8).tobytes()
                        + repr(b.matrix.shape).encode()).hexdigest()


def describe(p: fh.Projection, b: fh.Projection) -> dict:
    """Size and degree statistics of the thresholded binary matrix `b` built from
    the weighted projection `p`."""
    w = b.meta["threshold"]
    inp, fan = b.inputs(), b.fan_out()
    m = np.asarray(p.matrix)
    return {"threshold": w, "n_glomeruli": int(b.matrix.shape[0]),
            "n_glomeruli_innervated": int((fan > 0).sum()),
            "n_cells": int(b.matrix.shape[1]),
            "n_cells_dropped": int(m.shape[1] - b.matrix.shape[1]),
            "nnz": b.nnz,
            "inputs_mean": float(inp.mean()), "inputs_median": float(np.median(inp)),
            "inputs_max": int(inp.max()), "single_input_cells": int((inp == 1).sum()),
            "fan_out_cv": float(fan.std() / fan.mean()),
            "connections_retained": float(b.nnz / (m > 0).sum()),
            "synapses_retained": float(m[m >= w].sum() / m.sum()),
            "matrix_sha1": digest(b)}


def null_q(args) -> tuple[int, float]:
    """Q of one curveball draw (worker entry point): args = (binary uint8, seed, sweeps)."""
    a, seed, sweeps = args
    q = fh.curveball(fh.Projection("b", a.astype(np.float64), [str(i) for i in range(len(a))]),
                     seed=seed, sweeps=sweeps)
    return seed, ex._cooccurrence_q(fh.Projection("curveball", (q.matrix > 0).astype(np.float64),
                                                  q.glomeruli))


def summarise(q_real: float, q_null) -> dict:
    q_null = np.asarray(q_null, float)
    mu, sd = float(q_null.mean()), float(q_null.std(ddof=1))
    rt = stats.randomization_test(q_real, q_null)
    return {"B": int(len(q_null)), "q_null_mean": mu, "q_null_sd": sd,
            "z": float((q_real - mu) / sd), "relative_difference": float(q_real / mu - 1),
            "p_upper": rt["p_upper"], "p_lower": rt["p_lower"],
            "p_two_sided": rt["p_two_sided"], "rank_from_top": rt["rank_from_top"]}


# ---------------------------------------------------------------- checkpoint i/o

def _write(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def _fresh(B: int, thresholds, hemispheres) -> dict:
    return {"experiment": "F1 weak-connection audit", "preregistration": PREREG,
            "B": B, "seed0": SEED0, "sweeps": SWEEPS, "thresholds": list(thresholds),
            "hemisphere_order": [f"{d}_{s}" for d, s in hemispheres],
            "statistic": "Q = sum_{j<l} (M M^T)_{jl}^2 on the thresholded binary matrix "
                         "(experiments._cooccurrence_q)",
            "p_rule": "p_upper = (1 + #{Q_null >= Q_real}) / (B + 1)",
            "complete": False, "hemispheres": {}, "seconds": 0.0}


def _load(path: Path, B: int, thresholds, hemispheres) -> dict:
    if not path.exists():
        return _fresh(B, thresholds, hemispheres)
    out = json.loads(path.read_text())
    for k, v in (("seed0", SEED0), ("sweeps", SWEEPS), ("thresholds", list(thresholds))):
        if out.get(k) != v:
            raise ValueError(f"checkpoint {path} has {k}={out.get(k)!r}, expected {v!r}")
    if B < out["B"]:
        raise ValueError(f"checkpoint {path} has B={out['B']} > requested {B}")
    want = [f"{d}_{s}" for d, s in hemispheres]
    extra = sorted(set(out.get("hemispheres", {})) - set(want))
    if extra:
        # finish() would otherwise redo Holm, the table and holm_family over the
        # requested subset only and overwrite the full-family values in place.
        raise ValueError(f"checkpoint {path} holds hemispheres {extra} that were not requested; "
                         "request them too or use another --out")
    out["B"] = B
    out["hemisphere_order"] = want
    # Until this run finishes, the file must not claim completion or carry
    # conclusions computed from a smaller ensemble.
    out["complete"] = False
    for k in ("table", "holm_family", "hemibrain_statement", "reproduction_check"):
        out.pop(k, None)
    return out


# ---------------------------------------------------------------- runner

def run(cfg=None, out: Path | str = OUT, B: int = B_DEFAULT, thresholds=THRESHOLDS,
        hemispheres=None, load=None, workers: int = 3, chunk: int = 60,
        max_seconds: float = 500.0, log=print) -> dict:
    """Compute (or resume) the audit. Stops starting new chunks once
    `max_seconds` have passed; call again until the result has complete=True.

    `load(cfg, dataset, side)` returns the weighted projection (default
    `connectomes.projection`); `hemispheres` defaults to `connectomes.HEMISPHERES`.
    """
    t0 = time.time()
    out = Path(out)
    hemispheres = list(hemispheres or C.HEMISPHERES)
    load = load or C.projection
    if cfg is None and load is C.projection:
        from .config import load as load_cfg
        cfg = load_cfg()
    res = _load(out, B, thresholds, hemispheres)
    pool = None
    if workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(workers)
    stopped = False
    chunks = 0
    try:
        for ds, side in hemispheres:
            key = f"{ds}_{side}"
            h = res["hemispheres"].setdefault(key, {"dataset": ds, "side": side,
                                                    "label": f"{C.LABELS.get(ds, ds)} {side}",
                                                    "thresholds": {}})
            if all(_complete(h["thresholds"].get(str(w)), B) for w in thresholds):
                continue
            p = load(cfg, ds, side)
            h["glomeruli"] = list(p.glomeruli)
            h["synapses"] = float(np.asarray(p.matrix).sum())
            for w in thresholds:
                b = threshold(p, w)
                cell = h["thresholds"].get(str(w))
                d = describe(p, b)
                if cell is None:
                    cell = {**d, "q_real": ex._cooccurrence_q(b), "q_null": []}
                    h["thresholds"][str(w)] = cell
                elif cell["matrix_sha1"] != d["matrix_sha1"]:
                    raise ValueError(f"{key} w>={w}: matrix changed since the checkpoint")
                a = b.binary.astype(np.uint8)
                while len(cell["q_null"]) < B:
                    if chunks and time.time() - t0 > max_seconds:
                        stopped = True
                        break
                    lo = len(cell["q_null"])
                    jobs = [(a, SEED0 + s, SWEEPS) for s in range(lo, min(B, lo + chunk))]
                    got = (pool.map(null_q, jobs, chunksize=max(1, len(jobs) // (4 * workers)))
                           if pool else [null_q(j) for j in jobs])
                    assert [s for s, _ in got] == [j[1] for j in jobs]
                    cell["q_null"].extend(q for _, q in got)
                    chunks += 1
                    _write(res, out)
                    log(f"  {key} w>={w}: {len(cell['q_null'])}/{B} nulls "
                        f"({time.time() - t0:.0f}s)")
                if stopped:
                    break
                if "z" not in cell or cell.get("B") != len(cell["q_null"]):
                    cell.update(summarise(cell["q_real"], cell["q_null"]))
                    _write(res, out)
            if stopped:
                break
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    res["seconds"] = res.get("seconds", 0.0) + (time.time() - t0)
    done = (not stopped) and all(
        _complete(res["hemispheres"].get(f"{d}_{s}", {}).get("thresholds", {}).get(str(w)), B)
        for d, s in hemispheres for w in thresholds)
    res["complete"] = bool(done)
    if done:
        finish(res, hemispheres, thresholds)
        for w in thresholds:
            log(f"  w>={w}: " + "; ".join(
                f"{k} z {v['z']:+.1f} p {v['p_upper']:.3f}"
                for k, v in res["table"][str(w)].items() if isinstance(v, dict)))
    _write(res, out)
    log(f"  f1_audit: {'complete' if done else 'checkpointed, call again'} "
        f"({time.time() - t0:.0f}s this call)")
    return res


def _complete(cell: dict | None, B: int) -> bool:
    return bool(cell) and len(cell["q_null"]) >= B and cell.get("B") == len(cell["q_null"])


def finish(res: dict, hemispheres, thresholds) -> None:
    """Holm per threshold, the hemibrain rule and the w >= 1 reproduction check."""
    keys = [f"{d}_{s}" for d, s in hemispheres]
    table = {}
    for w in thresholds:
        cells = [res["hemispheres"][k]["thresholds"][str(w)] for k in keys]
        adj = stats.holm([c["p_upper"] for c in cells])
        for c, a in zip(cells, adj):
            c["p_upper_holm"] = a
        table[str(w)] = {k: {"z": c["z"], "p_upper": c["p_upper"], "p_upper_holm": c["p_upper_holm"],
                             "departs": c["p_upper"] < DEPART_ALPHA}
                         for k, c in zip(keys, cells)}
        table[str(w)]["n_departing"] = int(sum(c["p_upper"] < DEPART_ALPHA for c in cells))
    res["table"] = table
    res["holm_family"] = keys

    if "hemibrain_R" in res["hemispheres"] and str(STATEMENT_THRESHOLD) in res["hemispheres"]["hemibrain_R"]["thresholds"]:
        hb = res["hemispheres"]["hemibrain_R"]["thresholds"]
        at = hb[str(STATEMENT_THRESHOLD)]
        departs = at["p_upper"] < DEPART_ALPHA
        res["hemibrain_statement"] = {
            "rule": f"hemibrain Q departs from its null iff p_upper < {DEPART_ALPHA} on the "
                    f"w >= {STATEMENT_THRESHOLD} matrix (the paper's AnimQNSig rule)",
            "paper_sentence": "the hemibrain is the exception, indistinguishable from its null by Q",
            "departs_at_w2": bool(departs),
            "z_at_w2": at["z"], "p_upper_at_w2": at["p_upper"],
            "revise_paper_statement": bool(departs),
            "reported_not_rule": {w: {"z": hb[w]["z"], "p_upper": hb[w]["p_upper"],
                                      "p_lower": hb[w]["p_lower"],
                                      "lower_tail_below_alpha": hb[w]["p_lower"] < DEPART_ALPHA}
                                  for w in hb},
        }
    res["reproduction_check"] = reproduction_check(res)


def reproduction_check(res: dict) -> dict | None:
    """w >= 1 against results/connectomes.json (same matrices, same first 100 seeds)."""
    path = ROOT / "results" / "connectomes.json"
    if not path.exists():
        return None
    old = json.loads(path.read_text())
    n = int(old.get("B", 100))
    out = {"source": "results/connectomes.json", "B_compared": n, "hemispheres": {}}
    for k, h in res["hemispheres"].items():
        o = old.get("hemispheres", {}).get(k)
        c = h["thresholds"].get("1")
        if o is None or c is None or len(c["q_null"]) < n:
            continue
        s = o["structure"]
        sub = np.asarray(c["q_null"][:n])
        out["hemispheres"][k] = {
            "q_real_equal": bool(c["q_real"] == s["q_real"]),
            "null_mean_equal": bool(np.isclose(sub.mean(), s["q_null_mean"], rtol=1e-12, atol=0)),
            "null_sd_equal": bool(np.isclose(sub.std(ddof=1), s["q_null_sd"], rtol=1e-9, atol=0)),
            "nnz_equal": bool(c["nnz"] == o["full"]["nnz"]),
            "n_cells_equal": bool(c["n_cells"] == o["full"]["n_cells"])}
    out["all_equal"] = bool(out["hemispheres"]) and all(
        all(v.values()) for v in out["hemispheres"].values())
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="flypath.f1_audit", description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--max-seconds", type=float, default=500.0)
    ap.add_argument("--only", help="comma-separated hemispheres, e.g. hemibrain_R")
    ap.add_argument("--thresholds", help="comma-separated, default 1,2,3,5")
    a = ap.parse_args(argv)
    hs = C.HEMISPHERES
    if a.only:
        want = set(a.only.split(","))
        hs = [h for h in hs if f"{h[0]}_{h[1]}" in want]
    th = tuple(int(t) for t in a.thresholds.split(",")) if a.thresholds else THRESHOLDS
    r = run(out=a.out, B=a.B, thresholds=th, hemispheres=hs, workers=a.workers,
            chunk=a.chunk, max_seconds=a.max_seconds)
    return 0 if r["complete"] else 3


if __name__ == "__main__":
    import sys

    from flypath import f1_audit
    sys.exit(f1_audit.main())
