"""Standalone evaluation of F2 (larger null ensembles) as pre-registered.

Pre-registration: experiments/PREREGISTRATION_2026-09-28.md, section 14, F2
(also UPDATES.md section 14), frozen 2026-09-28 before any run.

Inputs:  results/f2_nulls.json   written by `python -m flypath.f2_audit`
         results/connectomes.json  the paper's B = 100 odour results
         the experiments.null_pool cache files the runner wrote
         results/f1_verdict.json   (optional; F1 pointer only)
Output:  results/f2_verdict.json and a printed report.

    VECLIB_MAXIMUM_THREADS=3 PYTHONPATH=. \\
        /opt/homebrew/anaconda3/envs/flybrain/bin/python experiments/eval_f2.py

Runtime is under a minute (about 40 s). The seven odour projections are
rebuilt, the real scores are recomputed, every stored null matrix is checked
for its margins, and six null draws per hemisphere are redrawn and rescored.

Criteria, copied from the pre-registration
------------------------------------------
F1. Weak-connection audit (paper revision)
  - Q z-score of every hemisphere against >= 100 curveball nulls at
    connection-weight thresholds w >= 1, 2, 3, 5 synapses; descriptive.
  - Fixes the paper's hemibrain statement if its Q departs from null at
    w >= 2.
F2. Larger null ensembles (paper revision)
  - Odour primary endpoint for all 7 hemispheres with B = 2,000 curveball
    nulls (p floor 0.001), Holm across the 7. Replaces the B = 100 results in
    the paper whatever the outcome.

F2 is a reporting item, not a hypothesis with a pass criterion. Its numbers
replace the paper's B = 100 odour results whatever they are. So
`primary.run_completed_as_registered` records only whether the registered run
exists in full and survives the integrity checks. The substantive outcome
(per-hemisphere relative difference, two-sided p, Holm) is reported under
`f2`. F1 is descriptive and is evaluated by experiments/eval_f1.py into
results/f1_verdict.json. This script only points to that file (point 13).

Unspecified details fixed before running
----------------------------------------
Timing, stated plainly. This script was written after the F2 full run had
finished, and after the runner's stored per-hemisphere summaries in
results/f2_nulls.json (relative difference, p, Holm) had been read. So it
adds no decision rule of its own. The analysis rules it applies were fixed in
the runner's docstring (flypath/f2_audit.py, "Unspecified details fixed
before running", points 1 to 7) before the run, and this script applies them
as written there. The points below were fixed before this script was first
run. They only choose what is recomputed, checked or reported. None of them
can change any registered F2 number.

 1. Registered statistics (runner points 5 and 6). Per hemisphere:
    relative difference = real / null mean - 1; two-sided p =
    min(1, 2 min(p_lower, p_upper)) from `stats.randomization_test` on the
    2,000 stored null scores (+1 correction, ties against the claim); Holm
    (`stats.holm`) over the seven two-sided p-values in
    `connectomes.HEMISPHERES` order; a hemisphere is significant iff its
    Holm-adjusted p < 0.05. The p floor is 2 / 2,001 = 0.0009995.
 2. Recomputation. Everything in point 1 is recomputed from the raw
    `null_scores` and the recomputed values are reported. The runner's stored
    summary and Holm block are a cross-check: p, rank and Holm must match
    exactly, means, SDs, z and relative differences to a relative 1e-12.
    The randomization p and Holm are also computed a second time with plain
    numpy, not through `flypath.stats`, and must agree exactly.
 3. Registered design checks: B = 2,000 in the file and for every
    hemisphere, exactly 2,000 null scores each, the hemisphere set is exactly
    the 7 of `connectomes.HEMISPHERES`, the Holm family is those 7 in order,
    seeds 10,000 to 11,999 with 30 sweeps, 4,000 mixtures (seed 0), kappa = 10,
    k = `experiments.primary_k` of the rebuilt projection.
 4. Rebuild. Each odour projection is rebuilt as `connectomes.compare`
    builds it (`connectomes.projection`, `flyhash.load_odours` defaults,
    `flyhash.align`, binary). Its binary matrix, glomerulus list and DoOR
    responses must equal the runner's setup cache, and its glomerulus, cell,
    nnz, odorant counts and k must equal the stored ones. The real score is
    recomputed (`experiments.bench` on `flyhash.mixtures(x, 4000, seed 0)`,
    `experiments.scores` at k) and must equal the stored one to 1e-12
    absolute. Whether it is bitwise equal is reported.
 5. Null ensemble. The 2,000 matrices are read from the
    `experiments.null_pool` cache file. Its name is derived here from the
    rebuilt matrix with `null_pool`'s key formula and must equal the path
    the runner recorded. There must be exactly 2,000 of them. Every one must
    be 0/1 with the real matrix's shape, row sums (glomerulus fan-out) and
    column sums (inputs per cell). The first 100 must equal the B = 100
    pool the paper used, if that cache file exists. The number of distinct
    matrices and the number equal to the real matrix are reported, not
    checked, since duplicates are legal draws.
 6. Spot nulls. For every hemisphere, draws b = 0, 99, 100, 1,000 and 1,999,
    plus the draw whose stored score is nearest the real score (smallest b
    on ties; this draw sets the rank boundary), are redrawn with
    `flyhash.curveball` (seed 10,000 + b, 30 sweeps). Each must equal the
    cached matrix and, rescored with `experiments.scores`, must reproduce the
    stored null score to 1e-12 absolute. 35 to 42 draws keep the runtime
    under a minute. A full redraw would repeat the 2.3-hour run.
 7. Paper reproduction at B = 100. For the primary odour row of each
    hemisphere in results/connectomes.json: k equal; real equal to 1e-12
    absolute; mean, SD and relative difference of null scores 0..99 equal to
    a relative 1e-9 (different code path, so summation order may differ);
    two-sided p equal exactly.
 8. Cross-evaluator. `f2_evaluate` of experiments/eval_f1.py is run on the
    same file (it writes nothing). It must return status "evaluated" with no
    integrity failure, and the same relative differences (relative 1e-12)
    and exactly the same p and Holm values per hemisphere.
 9. Descriptive table. Per hemisphere: k, real, null mean, null SD, Monte
    Carlo SE of the null mean (SD / sqrt(B)), z, relative difference,
    p_lower, p_upper, two-sided p, rank, whether p sits at the floor, Holm p,
    significance. The paper's B = 100 relative difference, p and the Holm
    adjustment of the seven B = 100 p-values are shown beside it. Summary:
    number Holm-significant, number with raw p < 0.05, number negative,
    relative-difference range, smallest Holm p, and the per-animal mean
    relative difference (hemispheres grouped by dataset as flypath/report.py
    groups them) with the number of animals whose mean is negative.
10. Paper consequences, descriptive only. The report.py macros that carry
    the own-glomeruli odour endpoint (AnimOdourPFloor, AnimOdourMin,
    AnimOdourMax, AnimOdourNNeg, AnimOdourNSig, AnimOdourHolmMin,
    AnimalsOdourNeg, and the "odours Delta% (p)" cell of every AnimalRows
    row) are computed with report.py's own formatting (`pct`, `pval`) from
    results/connectomes.json (B = 100) and from the recomputed B = 2,000
    values. The sentence in paper/flyhash.tex "after Holm adjustment across
    the seven hemispheres none is" is flagged for revision iff at least one
    hemisphere is Holm-significant at B = 2,000. Nothing is written to the
    paper. The shared-glomeruli analysis (odours_matched, AnimMatched*) is not
    part of F2, stays at B = 100, and is marked as such.
11. Monte Carlo uncertainty, descriptive only (not a rule, never changes
    point 1). Per hemisphere, the tail count c = min(#{null <= real},
    #{null >= real}) out of B gives a 95% Clopper-Pearson interval for the
    one-sided tail probability, doubled (capped at 1) for the two-sided p.
    Holm is recomputed with every hemisphere's p at its upper bound at once
    (conservative), and the hemispheres that stay below 0.05 are reported.
12. primary.run_completed_as_registered is true iff the design checks of
    point 3 hold and no integrity check in points 2 and 4 to 8 fails. It
    does not depend on any outcome.
13. F1 is not re-evaluated here. If results/f1_verdict.json exists, its
    primary flag, its hemibrain statement and its per-threshold counts are
    copied for reference. That file's F2 block was written before
    results/f2_nulls.json existed and reads "pending". For F2, this file
    supersedes it. eval_f1.py is not rerun here, because f1_verdict.json is
    not this script's output.

Deviations
----------
None.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flypath import connectomes as C  # noqa: E402
from flypath import experiments as ex  # noqa: E402
from flypath import flyhash as fh  # noqa: E402
from flypath import stats  # noqa: E402
from flypath.report import pct, pval  # noqa: E402

F2_IN = ROOT / "results" / "f2_nulls.json"
PAPER_IN = ROOT / "results" / "connectomes.json"
F1_VERDICT = ROOT / "results" / "f1_verdict.json"
PREREG_FILE = ROOT / "experiments" / "PREREGISTRATION_2026-09-28.md"
RUNNER = ROOT / "flypath" / "f2_audit.py"
SETUP_DIR = ROOT / "data" / "cache" / "f2_audit"
OUT = ROOT / "results" / "f2_verdict.json"
PREREG = "experiments/PREREGISTRATION_2026-09-28.md, section 14, F2"

B = 2000
SEED0 = 10_000
SWEEPS = 30
MIX_SEED = 0
ALPHA = 0.05
P_FLOOR = 2 / (B + 1)
REL_TOL = 1e-12
PAPER_TOL = 1e-9
ABS_SCORE_TOL = 1e-12
SPOT_B = (0, 99, 100, 1000, 1999)
KEYS = [f"{d}_{s}" for d, s in C.HEMISPHERES]
HOLM_SENTENCE = ("after Holm adjustment across the seven hemispheres none is "
                 "(smallest adjusted $p=\\AnimOdourHolmMin$ on own and "
                 "$\\AnimMatchedHolmMin$ on shared glomeruli)")
REGISTERED_TEXT = ("Odour primary endpoint for all 7 hemispheres with B = 2,000 curveball "
                   "nulls (p floor 0.001), Holm across the 7. Replaces the B = 100 results "
                   "in the paper whatever the outcome.")


# ---------------------------------------------------------------- helpers

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, tol: float = REL_TOL) -> bool:
    return bool(a == b or abs(a - b) <= tol * max(abs(a), abs(b)))


class Integrity:
    def __init__(self):
        self.failures: list[str] = []
        self.n_checks = 0

    def check(self, ok: bool, what: str) -> bool:
        self.n_checks += 1
        if not ok:
            self.failures.append(what)
        return bool(ok)


def plain_two_sided(real: float, null: np.ndarray) -> float:
    """Point 2: the randomization p again, without flypath.stats."""
    n = len(null)
    hi = (1 + np.count_nonzero(null >= real)) / (n + 1)
    lo = (1 + np.count_nonzero(null <= real)) / (n + 1)
    return min(1.0, 2 * min(hi, lo))


def plain_holm(p: list[float]) -> list[float]:
    """Point 2: Holm step-down again, without flypath.stats."""
    m = len(p)
    order = sorted(range(m), key=lambda i: (p[i], i))
    out = [0.0] * m
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        out[i] = min(1.0, run)
    return out


def pool_path(binary: np.ndarray, n: int) -> Path:
    """The file `experiments.null_pool(p, n)` reads (same key formula)."""
    key = hashlib.sha1(binary.astype(np.uint8).tobytes()
                       + f"{n}-{SEED0}-{SWEEPS}".encode()).hexdigest()[:16]
    return ex.ROOT / "data" / "cache" / f"curveball-{key}.npz"


def clopper_pearson(c: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo = 0.0 if c == 0 else float(sps.beta.ppf(a, c, n - c + 1))
    hi = 1.0 if c == n else float(sps.beta.ppf(1 - a, c + 1, n - c))
    return lo, hi


def paper_row(paper: dict, key: str) -> dict:
    return next(r for r in paper["hemispheres"][key]["odours"] if r["primary"])


# ---------------------------------------------------------------- checks

def design(res: dict, ig: Integrity) -> dict:
    """Point 3 (the parts that need only the file)."""
    recs = res.get("hemispheres", {})
    ig.check(res.get("B") == B, f"file B = {res.get('B')} != {B}")
    ig.check(sorted(recs) == sorted(KEYS), f"hemisphere set {sorted(recs)} != {sorted(KEYS)}")
    ig.check((res.get("holm") or {}).get("family") == KEYS, "runner's Holm family is not the 7 in order")
    ig.check(res.get("seed0") == SEED0 and res.get("sweeps") == SWEEPS,
             f"seed0/sweeps {res.get('seed0')}/{res.get('sweeps')} != {SEED0}/{SWEEPS}")
    ig.check(res.get("n_items") == ex.N_ITEMS and res.get("neighbours") == ex.NEIGHBOURS
             and res.get("mixture_seed") == MIX_SEED,
             "benchmark settings (items, kappa, mixture seed) not as registered")
    ig.check(close(res.get("p_floor", -1), P_FLOOR), "stored p floor != 2 / (B + 1)")
    for k in KEYS:
        r = recs.get(k)
        if r is None:
            continue
        ig.check(r.get("B") == B and len(r.get("null_scores", ())) == B,
                 f"{k}: null ensemble is not complete at B = {B}")
        ig.check(r.get("seeds") == [SEED0, SEED0 + B - 1] and r.get("sweeps") == SWEEPS,
                 f"{k}: seeds or sweeps not as registered")
    return {"B": res.get("B"), "hemispheres": list(recs), "seed0": res.get("seed0"),
            "sweeps": res.get("sweeps"), "n_items": res.get("n_items"),
            "neighbours": res.get("neighbours"), "mixture_seed": res.get("mixture_seed"),
            "p_floor": P_FLOOR, "runner_compute_hours": res.get("seconds", 0) / 3600}


def recompute(res: dict, ig: Integrity) -> dict:
    """Points 1 and 2."""
    rows = {}
    for k in KEYS:
        r = res["hemispheres"][k]
        null = np.asarray(r["null_scores"], float)
        real = float(r["real"])
        rt = stats.randomization_test(real, null)
        mu, sd = float(null.mean()), float(null.std(ddof=1))
        row = {"k": r["k"], "real": real, "null_mean": mu, "null_sd": sd,
               "null_mean_se": sd / math.sqrt(len(null)), "z": (real - mu) / sd,
               "relative_difference": real / mu - 1,
               "p_lower": rt["p_lower"], "p_upper": rt["p_upper"],
               "p_two_sided": rt["p_two_sided"], "rank_from_top": rt["rank_from_top"],
               "at_p_floor": rt["p_two_sided"] <= P_FLOOR + 1e-15,
               "direction": "below null" if real < mu else "above null",
               "n_null_le_real": int(np.count_nonzero(null <= real)),
               "n_null_ge_real": int(np.count_nonzero(null >= real))}
        tag = f"{k}"
        for name, tol_key in (("null_mean", "null_mean"), ("null_sd", "null_sd"), ("z", "z"),
                              ("relative_difference", "relative_difference")):
            ig.check(close(row[name], r[tol_key]), f"{tag}: {name} differs from stored")
        ig.check(close(row["null_mean_se"], r["null_se"]), f"{tag}: null SE differs from stored")
        for name in ("p_two_sided", "p_lower", "p_upper", "rank_from_top"):
            ig.check(row[name] == r[name], f"{tag}: {name} differs from stored")
        ig.check(plain_two_sided(real, null) == row["p_two_sided"],
                 f"{tag}: plain-numpy two-sided p differs from flypath.stats")
        rows[k] = row
    ps = [rows[k]["p_two_sided"] for k in KEYS]
    adj = stats.holm(ps)
    adj_plain = plain_holm(ps)
    ig.check(adj == adj_plain, "plain-numpy Holm differs from flypath.stats.holm")
    for k, a in zip(KEYS, adj):
        rows[k]["p_holm"] = a
        rows[k]["significant_holm_lt_0.05"] = a < ALPHA
        ig.check(a == res["holm"]["p_holm"][k], f"{k}: Holm p differs from stored")
        ig.check((a < ALPHA) == res["holm"]["significant"][k], f"{k}: stored significance differs")
    ig.check(sum(a < ALPHA for a in adj) == res["holm"]["n_significant"],
             "stored Holm n_significant differs")
    return rows


def rebuild_and_spot(res: dict, cfg, ig: Integrity, spot: bool, log=print) -> dict:
    """Points 3 (k), 4, 5 and 6."""
    out = {}
    t0 = time.time()
    for ds, side in C.HEMISPHERES:
        k = f"{ds}_{side}"
        r = res["hemispheres"][k]
        tag = k
        p = C.projection(cfg, ds, side)
        od = fh.load_odours(cfg, p.glomeruli)
        po_w = fh.align(p, od.glomeruli)
        po = fh.Projection("b", po_w.binary, po_w.glomeruli, meta=po_w.meta)
        a = po.binary.astype(np.uint8)
        kP = ex.primary_k(po)
        # point 4: against the runner's setup cache and stored counts
        su = SETUP_DIR / f"{k}-setup.npz"
        if ig.check(su.exists(), f"{tag}: runner setup cache missing"):
            z = np.load(su)
            ig.check(np.array_equal(z["binary"], a), f"{tag}: rebuilt binary matrix != setup cache")
            ig.check([str(g) for g in z["glomeruli"]] == list(po.glomeruli),
                     f"{tag}: glomerulus list != setup cache")
            ig.check(np.array_equal(z["x"], od.x), f"{tag}: DoOR responses != setup cache")
        ig.check(kP == r["k"], f"{tag}: primary k {kP} != stored {r['k']}")
        ig.check(a.shape == (r["n_glomeruli"], r["n_cells"]), f"{tag}: shape != stored")
        ig.check(int(a.sum()) == r["nnz"], f"{tag}: nnz != stored")
        ig.check(len(od.x) == r["n_odorants"], f"{tag}: odorant count != stored")
        b0 = ex.bench(fh.mixtures(od.x, ex.N_ITEMS, seed=MIX_SEED))
        real = float(ex.scores(b0, po, (kP,))[0])
        ig.check(abs(real - r["real"]) <= ABS_SCORE_TOL, f"{tag}: recomputed real score != stored")
        # point 5: the null ensemble itself
        path = pool_path(a, B)
        ig.check(str(path.relative_to(ex.ROOT)) == r["checks"].get("pool_cache"),
                 f"{tag}: derived pool cache path != runner's")
        info = {"k": kP, "shape": list(a.shape), "nnz": int(a.sum()),
                "real_recomputed": real, "real_bitwise_equal": real == r["real"],
                "pool_cache": str(path.relative_to(ex.ROOT))}
        if not ig.check(path.exists(), f"{tag}: null_pool cache file missing"):
            out[k] = info
            continue
        mats = np.load(path)["m"]
        ig.check(mats.shape == (B, *a.shape), f"{tag}: cached pool shape {mats.shape}")
        ig.check(bool(((mats == 0) | (mats == 1)).all()), f"{tag}: cached pool not 0/1")
        rows_ok = bool((mats.sum(axis=2) == a.sum(axis=1)[None, :]).all())
        cols_ok = bool((mats.sum(axis=1) == a.sum(axis=0)[None, :]).all())
        ig.check(rows_ok, f"{tag}: some null breaks the glomerulus fan-out (row sums)")
        ig.check(cols_ok, f"{tag}: some null breaks the inputs per cell (column sums)")
        digests = [hashlib.sha1(m.tobytes()).hexdigest() for m in mats]
        info.update({"n_matrices": int(len(mats)), "margins_preserved": rows_ok and cols_ok,
                     "n_distinct": len(set(digests)),
                     "n_equal_to_real": digests.count(hashlib.sha1(a.tobytes()).hexdigest())})
        p100 = pool_path(a, 100)
        if p100.exists():
            eq = bool(np.array_equal(np.load(p100)["m"], mats[:100]))
            ig.check(eq, f"{tag}: first 100 nulls != the paper's B = 100 pool")
            info["first_100_equal_B100_pool"] = eq
        # point 6: spot redraws and rescoring
        null = np.asarray(r["null_scores"], float)
        near = int(np.argmin(np.abs(null - r["real"])))      # argmin returns the smallest b on ties
        spots = []
        if spot:
            for b in sorted(set(SPOT_B) | {near}):
                q = fh.curveball(fh.Projection("b", po.binary, po.glomeruli),
                                 seed=SEED0 + b, sweeps=SWEEPS)
                m = (q.matrix > 0).astype(np.uint8)
                same = bool(np.array_equal(m, mats[b]))
                s = float(ex.scores(b0, fh.Projection("curveball", m.astype(float), po.glomeruli),
                                    (kP,))[0])
                ok = abs(s - null[b]) <= ABS_SCORE_TOL
                ig.check(same, f"{tag}: redrawn null b = {b} != cached matrix")
                ig.check(ok, f"{tag}: rescored null b = {b} gives {s}, stored {null[b]}")
                spots.append({"b": b, "nearest_to_real": b == near, "matrix_equal": same,
                              "score": s, "stored": float(null[b]), "score_equal": bool(ok)})
        info.update({"nearest_null_b": near, "nearest_null_score": float(null[near]),
                     "spot_nulls": spots})
        del mats
        out[k] = info
        log(f"  checked {k} ({time.time() - t0:.0f}s)")
    return out


def paper_reproduction(res: dict, paper: dict, ig: Integrity) -> dict:
    """Point 7."""
    rows = {}
    for k in KEYS:
        r = res["hemispheres"][k]
        o = paper_row(paper, k)
        first = np.asarray(r["null_scores"][:100], float)
        mu, sd = float(first.mean()), float(first.std(ddof=1))
        rt = stats.randomization_test(r["real"], first)
        got = {"k_equal": o["k"] == r["k"],
               "real_equal": abs(o["real"] - r["real"]) <= ABS_SCORE_TOL,
               "null_mean_equal": close(o["null_mean"], mu, PAPER_TOL),
               "null_sd_equal": close(o["null_sd"], sd, PAPER_TOL),
               "relative_difference_equal": close(o["relative_difference"], r["real"] / mu - 1, PAPER_TOL),
               "p_two_sided_equal": o["p_two_sided"] == rt["p_two_sided"]}
        for name, ok in got.items():
            ig.check(bool(ok), f"{k}: B = 100 paper reproduction failed ({name})")
        for name, v in (r.get("checks") or {}).items():
            if isinstance(v, bool):
                ig.check(v, f"{k}: runner check {name} is false")
        rows[k] = {kk: bool(v) for kk, v in got.items()}
    return {"source": "results/connectomes.json (B = 100)", "hemispheres": rows,
            "all_equal": all(all(v.values()) for v in rows.values())}


def cross_evaluator(rows: dict, ig: Integrity) -> dict:
    """Point 8."""
    spec = importlib.util.spec_from_file_location("eval_f1", ROOT / "experiments" / "eval_f1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ig1 = mod.Integrity()
    f2 = mod.f2_evaluate(ig1)
    ok_status = f2.get("status") == "evaluated"
    ig.check(ok_status, f"eval_f1.f2_evaluate status is {f2.get('status')}")
    ig.check(not ig1.failures, f"eval_f1.f2_evaluate integrity failures: {ig1.failures}")
    agree = {}
    if ok_status:
        for k in KEYS:
            o = f2["hemispheres"][k]
            a = (close(o["relative_difference"], rows[k]["relative_difference"])
                 and o["p_two_sided"] == rows[k]["p_two_sided"] and o["p_holm"] == rows[k]["p_holm"])
            ig.check(a, f"{k}: eval_f1.f2_evaluate disagrees")
            agree[k] = bool(a)
    return {"evaluator": "experiments/eval_f1.py f2_evaluate", "status": f2.get("status"),
            "its_integrity_checks": ig1.n_checks, "its_integrity_failures": ig1.failures,
            "agrees": agree, "all_agree": bool(agree) and all(agree.values())}


# ---------------------------------------------------------------- outcome

def outcome(rows: dict, paper: dict) -> dict:
    """Points 9 and 10."""
    p100 = [paper_row(paper, k)["p_two_sided"] for k in KEYS]
    h100 = stats.holm(p100)
    table = {}
    for k, ph in zip(KEYS, h100):
        o = paper_row(paper, k)
        r = rows[k]
        table[k] = {**r,
                    "paper_B100": {"relative_difference": o["relative_difference"],
                                   "null_mean": o["null_mean"], "null_sd": o["null_sd"],
                                   "p_two_sided": o["p_two_sided"], "p_holm": ph,
                                   "significant_holm_lt_0.05": ph < ALPHA},
                    "change": {"raw_p_lt_0.05": [o["p_two_sided"] < ALPHA, r["p_two_sided"] < ALPHA],
                               "holm_lt_0.05": [ph < ALPHA, r["p_holm"] < ALPHA]}}
    rel = [rows[k]["relative_difference"] for k in KEYS]
    animals: dict[str, list[float]] = {}
    for k in KEYS:
        animals.setdefault(k.rsplit("_", 1)[0], []).append(rows[k]["relative_difference"])
    an_mean = {a: float(np.mean(v)) for a, v in animals.items()}
    sig = [k for k in KEYS if rows[k]["significant_holm_lt_0.05"]]
    summary = {"n_significant_holm": len(sig), "significant_holm": sig,
               "n_raw_p_lt_0.05": sum(rows[k]["p_two_sided"] < ALPHA for k in KEYS),
               "raw_p_lt_0.05": [k for k in KEYS if rows[k]["p_two_sided"] < ALPHA],
               "n_at_p_floor": sum(rows[k]["at_p_floor"] for k in KEYS),
               "at_p_floor": [k for k in KEYS if rows[k]["at_p_floor"]],
               "n_negative": sum(v < 0 for v in rel),
               "relative_difference_min": min(rel), "relative_difference_max": max(rel),
               "min_p_holm": min(rows[k]["p_holm"] for k in KEYS),
               "animal_mean_relative_difference": an_mean,
               "n_animals_negative": sum(v < 0 for v in an_mean.values()),
               "paper_B100": {"n_significant_holm": sum(h < ALPHA for h in h100),
                              "n_raw_p_lt_0.05": sum(p < ALPHA for p in p100),
                              "min_p_holm": min(h100)}}

    def macros(rel_d, ps, floor, an_means):
        adj = stats.holm(ps)
        return {"AnimOdourPFloor": pval(floor),
                "AnimOdourMin": f"{pct(min(rel_d), 1)}\\%", "AnimOdourMax": f"{pct(max(rel_d), 1)}\\%",
                "AnimOdourNNeg": sum(v < 0 for v in rel_d),
                "AnimOdourNSig": sum(p < ALPHA for p in ps),
                "AnimOdourHolmMin": pval(min(adj)),
                "AnimalsOdourNeg": sum(v < 0 for v in an_means)}

    rel100 = [paper_row(paper, k)["relative_difference"] for k in KEYS]
    an100: dict[str, list[float]] = {}
    for k, v in zip(KEYS, rel100):
        an100.setdefault(k.rsplit("_", 1)[0], []).append(v)
    old = macros(rel100, p100, 2 / (paper["B"] + 1), [np.mean(v) for v in an100.values()])
    new = macros(rel, [rows[k]["p_two_sided"] for k in KEYS], P_FLOOR, list(an_mean.values()))
    cells = {k: {"paper_B100": f"{pct(paper_row(paper, k)['relative_difference'], 1)} "
                               f"({pval(paper_row(paper, k)['p_two_sided'])})",
                 "B2000": f"{pct(rows[k]['relative_difference'], 1)} ({pval(rows[k]['p_two_sided'])})"}
             for k in KEYS}
    consequences = {
        "note": "descriptive; F2 replaces the paper's B = 100 odour results whatever the outcome. "
                "Recomputed with flypath.report's pct/pval formatting; nothing is written to the paper.",
        "macros": {m: {"paper_B100": old[m], "B2000": new[m], "changes": old[m] != new[m]} for m in old},
        "table_animals_odours_cell": cells,
        "holm_sentence": {"location": "paper/flyhash.tex, paragraph starting 'On odours'",
                          "text": HOLM_SENTENCE,
                          "holds_at_B2000": len(sig) == 0,
                          "revise": len(sig) > 0,
                          "significant_holm_B2000": sig},
        "not_in_F2": "odours_matched (shared glomeruli; AnimMatchedMin/Max/NSig/HolmMin) is not part "
                     "of F2 and stays at B = 100 (flypath/f2_audit.py point 4)",
    }
    return {"table": table, "summary": summary, "paper_consequences": consequences}


def mc_uncertainty(rows: dict) -> dict:
    """Point 11 (descriptive only)."""
    per, upper = {}, []
    for k in KEYS:
        r = rows[k]
        c = min(r["n_null_le_real"], r["n_null_ge_real"])
        lo, hi = clopper_pearson(c, B)
        p_lo, p_hi = min(1.0, 2 * lo), min(1.0, 2 * hi)
        per[k] = {"tail_count": c, "p_two_sided": r["p_two_sided"],
                  "p_two_sided_95ci": [p_lo, p_hi]}
        upper.append(p_hi)
    adj = stats.holm(upper)
    for k, a in zip(KEYS, adj):
        per[k]["p_holm_all_at_upper_bound"] = a
    return {"note": "descriptive; Clopper-Pearson 95% for the one-sided tail probability from the "
                    "tail count, doubled; Holm with every p at its upper bound at once (conservative). "
                    "Never changes the registered p of point 1.",
            "hemispheres": per,
            "significant_holm_at_upper_bounds": [k for k, a in zip(KEYS, adj) if a < ALPHA]}


def f1_pointer() -> dict:
    """Point 13."""
    if not F1_VERDICT.exists():
        return {"status": "not found", "file": "results/f1_verdict.json"}
    v = json.loads(F1_VERDICT.read_text())
    f1 = v.get("f1", {})
    hb = f1.get("hemibrain_statement", {})
    return {"file": "results/f1_verdict.json", "evaluator": "experiments/eval_f1.py",
            "sha256": sha256(F1_VERDICT),
            "note": "F1 is descriptive and evaluated there, not here. That file's F2 block was written "
                    "before results/f2_nulls.json existed and reads "
                    f"'{v.get('f2', {}).get('status')}'; for F2 this file supersedes it.",
            "run_completed_as_registered": v.get("primary", {}).get("run_completed_as_registered"),
            "hemibrain_departs_at_w2": hb.get("departs_at_w2"),
            "revise_hemibrain_statement": hb.get("revise_paper_statement"),
            "per_threshold": {w: {kk: s.get(kk) for kk in ("n_departing", "n_holm_lt_0.05",
                                                           "not_departing", "z_min", "z_max")}
                              for w, s in f1.get("summary", {}).get("per_threshold", {}).items()},
            "its_f2_status": v.get("f2", {}).get("status")}


# ---------------------------------------------------------------- report

def report(v: dict, log=print) -> None:
    f2 = v["f2"]
    log(f"F2 odour primary endpoint, B = {B} curveball nulls, Holm across 7 "
        f"(p floor {P_FLOOR:.5f})")
    log(f"  {'hemisphere':12s} {'k':>3s} {'real':>7s} {'null mean':>9s} {'rel':>7s} "
        f"{'p':>7s} {'Holm':>7s}   paper B=100: rel, p, Holm")
    for k in KEYS:
        r = f2["table"][k]
        o = r["paper_B100"]
        log(f"  {k:12s} {r['k']:3d} {r['real']:.4f} {r['null_mean']:9.4f} "
            f"{100 * r['relative_difference']:+6.2f}% {r['p_two_sided']:7.4f} {r['p_holm']:7.4f}"
            f"{'*' if r['significant_holm_lt_0.05'] else ' '}  "
            f"{100 * o['relative_difference']:+6.2f}% {o['p_two_sided']:.4f} {o['p_holm']:.4f}")
    s = f2["summary"]
    log(f"  Holm < 0.05: {s['n_significant_holm']}/7 {s['significant_holm']} "
        f"(B=100: {s['paper_B100']['n_significant_holm']}/7); raw p < 0.05: "
        f"{s['n_raw_p_lt_0.05']}/7 (B=100: {s['paper_B100']['n_raw_p_lt_0.05']}/7); "
        f"negative {s['n_negative']}/7, {100 * s['relative_difference_min']:+.2f}% to "
        f"{100 * s['relative_difference_max']:+.2f}%; at floor {s['at_p_floor']}")
    pc = f2["paper_consequences"]
    for m, d in pc["macros"].items():
        log(f"  {m:18s} {str(d['paper_B100']):>9s} -> {str(d['B2000']):>9s}"
            f"{'  (changes)' if d['changes'] else ''}")
    log(f"  paper Holm sentence ('none is') holds at B=2000: {pc['holm_sentence']['holds_at_B2000']}")
    mc = f2["mc_uncertainty"]
    log(f"  Holm-significant with every p at its 95% MC upper bound: "
        f"{mc['significant_holm_at_upper_bounds']}")
    p = v["primary"]
    log(f"integrity: {p['integrity_checks']} checks, {len(p['integrity_failures'])} failures")
    for f in p["integrity_failures"]:
        log(f"  FAIL {f}")
    log(f"run completed as registered: {p['run_completed_as_registered']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-spot", action="store_true", help="skip the spot redraws (point 6)")
    a = ap.parse_args(argv)
    t0 = time.time()
    from flypath.config import load as load_cfg
    ig = Integrity()
    if not F2_IN.exists():
        print("results/f2_nulls.json not found; run `python -m flypath.f2_audit` until it prints 'complete'")
        return 1
    res = json.loads(F2_IN.read_text())
    paper = json.loads(PAPER_IN.read_text())
    des = design(res, ig)
    if ig.failures:
        v = {"primary": {"run_completed_as_registered": False, "integrity_checks": ig.n_checks,
                         "integrity_failures": ig.failures}}
        Path(a.out).write_text(json.dumps(v, indent=1))
        print("\n".join(ig.failures))
        return 1
    rows = recompute(res, ig)
    checks = rebuild_and_spot(res, load_cfg(), ig, spot=not a.no_spot)
    repro = paper_reproduction(res, paper, ig)
    cross = cross_evaluator(rows, ig)
    out = outcome(rows, paper)
    out["mc_uncertainty"] = mc_uncertainty(rows)
    completed = not ig.failures
    v = {"experiment": "F2 larger null ensembles: odour primary endpoint, every hemisphere",
         "preregistration": PREREG,
         "registered_text": REGISTERED_TEXT,
         "evaluator": "experiments/eval_f2.py",
         "inputs": {"f2_nulls.json_sha256": sha256(F2_IN),
                    "f2_audit.py_sha256": sha256(RUNNER),
                    "connectomes.json_sha256": sha256(PAPER_IN),
                    "preregistration_sha256": sha256(PREREG_FILE)},
         "primary": {"run_completed_as_registered": completed,
                     "definition": "point 12: registered design present in full (B = 2,000, all 7 "
                                   "hemispheres, primary size, Holm family of 7) and no integrity "
                                   "failure; F2 is a reporting item with no pass/fail outcome",
                     "integrity_checks": ig.n_checks, "integrity_failures": ig.failures},
         "registered_items": {
             "odour_primary_endpoint_all_7": completed,
             "B_2000_curveball_nulls": des["B"] == B,
             "p_floor": P_FLOOR,
             "holm_across_7": True,
             "replaces_B100_results": "yes, whatever the outcome (see f2.paper_consequences)"},
         "design": des,
         "f2": {"status": "evaluated",
                "rule": "two-sided randomization p on 2,000 curveball nulls; Holm across the 7 "
                        "hemispheres; significant iff Holm p < 0.05 (flypath/f2_audit.py points 5-6)",
                **out},
         "verification": {"rebuild_and_nulls": checks, "paper_B100_reproduction": repro,
                          "cross_evaluator": cross,
                          "spot_nulls_checked": sum(len(c.get("spot_nulls", [])) for c in checks.values()),
                          "spot_nulls_all_equal": all(s["matrix_equal"] and s["score_equal"]
                                                      for c in checks.values()
                                                      for s in c.get("spot_nulls", []))},
         "f1": f1_pointer(),
         "seconds": time.time() - t0}
    Path(a.out).write_text(json.dumps(v, indent=1, default=float))
    report(v)
    print(f"wrote {a.out} ({v['seconds']:.0f}s)")
    return 0 if completed else 1


if __name__ == "__main__":
    sys.exit(main())
