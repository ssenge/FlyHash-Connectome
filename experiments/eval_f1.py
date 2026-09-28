"""Standalone evaluation of F1 (weak-connection audit) as pre-registered, with
the F2 summary (larger null ensembles) added once its results file exists.

Pre-registration: experiments/PREREGISTRATION_2026-09-28.md, sections F1 and
F2 (also UPDATES.md section 14), frozen 2026-09-28 before any run.

Inputs:  results/f1_audit.json   written by `python -m flypath.f1_audit`
         results/f2_nulls.json   written by `python -m flypath.f2_audit`
                                 (optional; F2 is reported as pending without it)
Output:  results/f1_verdict.json and a printed report.

    VECLIB_MAXIMUM_THREADS=3 PYTHONPATH=. \\
        /opt/homebrew/anaconda3/envs/flybrain/bin/python experiments/eval_f1.py

Runtime is under a minute (about 25 s): the thresholded matrices are rebuilt
from the connectome projections and three null draws per cell are redrawn.

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

Neither item is a hypothesis with a pass criterion. F1 is descriptive plus one
conditional rule, the hemibrain statement. F2 is a reporting item whose numbers
replace the paper's B = 100 odour results whatever they are. So
`primary.run_completed_as_registered` records only whether the registered run
exists in full, and the substantive outcome is reported separately under
`f1` and `f2`.

Unspecified details fixed before running
----------------------------------------
Timing, stated plainly: this script was written after the F1 full run had
finished and after its stored summary table (z, p_upper, Holm, hemibrain
statement) had been read. So it adds no decision rule of its own. Every rule
it applies was fixed in the runner's docstring (flypath/f1_audit.py,
"Unspecified details fixed before running", points 1 to 9) before the run,
and this script applies those rules as written there. For F2 it applies
flypath/f2_audit.py, points 1 to 7, fixed there before the F2 run. The points
fixed here (3 to 11) can only change what is reported or flagged as an
integrity failure. None of them can change the hemibrain rule's outcome or
any F2 number.

 1. F1 rule for the hemibrain statement (runner point 7). The hemibrain's Q
    departs from its null at w >= 2 iff its one-sided upper randomization
    p_upper = (1 + #{Q_null >= Q_real}) / (B + 1) < 0.05 on the w >= 2 matrix.
    If it departs, the paper's sentence "the hemibrain is the exception,
    indistinguishable from its null by Q" needs revision. w >= 3 and w >= 5
    and the lower tail are reported and do not enter the rule.
 2. F1 statistics (runner point 6): z = (Q_real - mean) / SD(ddof = 1) over
    the B null Q values; relative difference Q_real / mean - 1; p from
    `flypath.stats.randomization_test` (ties count against the claim); Holm
    (`flypath.stats.holm`) on p_upper across the 7 hemispheres within each
    threshold, in `connectomes.HEMISPHERES` order, reported descriptively.
 3. Recomputation. Every F1 statistic above is recomputed here from the
    stored raw null Q values (`q_null`), and the recomputed values are the
    ones reported. The runner's stored statistics, table and hemibrain block
    are a cross-check only. p-values and Holm must match exactly. Means, SDs,
    z and relative differences must match to a relative 1e-12. Any mismatch is
    an integrity failure.
 4. Independent rebuild of the matrices. For every hemisphere and threshold
    the binary matrix is rebuilt from `connectomes.projection` with plain
    numpy (weights >= w, empty Kenyon-cell columns dropped, every glomerulus
    row kept). Its SHA-1, computed as flypath/f1_audit.py computes it, its
    shape, nnz, dropped cells, retained connections and synapses, and
    Q = sum_{j<l} (A A^T)_{jl}^2 computed in exact integer arithmetic must
    equal the stored values. The glomerulus list must equal the stored one.
 5. Null spot check. For every cell, draws b = 0, B // 2 and B - 1 are
    redrawn with `flyhash.curveball` (seed 10,000 + b, 30 sweeps, the
    runner's point 5) and their Q must equal the stored q_null[b] exactly.
    84 draws keep the runtime under a minute. A full redraw would repeat the
    40-minute run.
 6. Reproduction of the paper (runner point 5). At w >= 1 the stored Q_real
    and the mean and SD of the first 100 null draws must equal
    results/connectomes.json (the B = 100 structure test in the paper); p_upper
    of those 100 draws must equal the paper's p_upper; nnz and cell count
    must equal the paper's full matrix. Recomputed here, with the runner's
    stored `reproduction_check` as cross-check.
 7. Registered design checks: the file is marked complete; B >= 100 (the
    registered minimum) and every cell holds exactly B nulls with B equal
    across cells; thresholds are exactly {1, 2, 3, 5}; the hemisphere set is
    exactly the 7 of `connectomes.HEMISPHERES`; seeds start at 10,000 with 30
    sweeps; the Holm family recorded by the runner is the 7 in order.
 8. Descriptive summary per threshold: number of hemispheres with
    p_upper < 0.05 (the paper's AnimQNSig count), number with Holm-adjusted
    p_upper < 0.05, z range, relative-difference range, and the Monte Carlo SE
    of each null mean (SD / sqrt(B)). Per hemisphere: what dropping
    single-synapse connections (w >= 1 to w >= 2) removes (connections,
    synapses, cells) and the change in z.
 9. The paper's current Q sentence ("departs from its null in 6 of 7
    hemispheres, z from -0.0 to 17.5, raw p = 0.010 each, Holm-adjusted over
    seven 0.069", B = 100, w >= 1) is recomputed from results/connectomes.json
    the way flypath/report.py computes those macros. It is set beside the same
    quantities at B = 1,000, w >= 1 from this audit. This is descriptive and
    is not a registered rule. F1 registers only the hemibrain rule.
10. F2 (only if results/f2_nulls.json exists and holds all 7 hemispheres
    with a Holm block). Per hemisphere: relative difference = real / null
    mean - 1 and the two-sided p from `stats.randomization_test` on the
    stored 2,000 null scores. Holm (`stats.holm`) over the 7 two-sided
    p-values in `connectomes.HEMISPHERES` order. A hemisphere counts as
    significant iff its Holm-adjusted p < 0.05 (flypath/f2_audit.py points 5
    and 6). All of this is recomputed from `null_scores` and cross-checked
    against the stored summary and Holm block (exact for p, relative 1e-12
    otherwise). The runner's own checks (B = 100 reproduction, nesting,
    pool cache) must all be true. The paper's B = 100 values are shown beside
    the new ones. If the file is missing, F2 is reported as pending, and no
    number is taken from the runner's partial checkpoints.
11. primary.run_completed_as_registered is true iff point 7 holds and no
    integrity failure occurred in points 3 to 6. It does not depend on any
    outcome. F2 has its own `f2.status` ("evaluated", "pending" or
    "invalid") and its own integrity list, and does not enter the F1 flag.

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flypath import connectomes as C  # noqa: E402
from flypath import flyhash as fh  # noqa: E402
from flypath import stats  # noqa: E402

F1_IN = ROOT / "results" / "f1_audit.json"
F2_IN = ROOT / "results" / "f2_nulls.json"
PAPER_IN = ROOT / "results" / "connectomes.json"
OUT = ROOT / "results" / "f1_verdict.json"
PREREG = "experiments/PREREGISTRATION_2026-09-28.md, sections F1 and F2"

THRESHOLDS = (1, 2, 3, 5)
B_MIN = 100
SEED0 = 10_000
SWEEPS = 30
ALPHA = 0.05
STATEMENT_W = 2
F2_B = 2000
REL_TOL = 1e-12
KEYS = [f"{d}_{s}" for d, s in C.HEMISPHERES]
PAPER_SENTENCE = "the hemibrain is the exception, indistinguishable from its null by Q"


# ---------------------------------------------------------------- helpers

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, tol: float = REL_TOL) -> bool:
    return bool(a == b or abs(a - b) <= tol * max(abs(a), abs(b)))


def q_stat(a: np.ndarray) -> float:
    """Q = sum_{j<l} (A A^T)_{jl}^2, exact in int64."""
    a = a.astype(np.int64)
    c = a @ a.T
    np.fill_diagonal(c, 0)
    return float((c ** 2).sum() // 2)


def sha1_binary(a: np.ndarray) -> str:
    """As flypath/f1_audit.py `digest`."""
    return hashlib.sha1(a.astype(np.uint8).tobytes() + repr(a.shape).encode()).hexdigest()


def summarise_null(real: float, null) -> dict:
    null = np.asarray(null, float)
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    rt = stats.randomization_test(real, null)
    return {"B": int(len(null)), "mean": mu, "sd": sd, "se_mean": sd / math.sqrt(len(null)),
            "z": float((real - mu) / sd), "relative_difference": float(real / mu - 1),
            "p_upper": rt["p_upper"], "p_lower": rt["p_lower"],
            "p_two_sided": rt["p_two_sided"], "rank_from_top": rt["rank_from_top"]}


class Integrity:
    def __init__(self):
        self.failures: list[str] = []
        self.n_checks = 0

    def check(self, ok: bool, what: str) -> bool:
        self.n_checks += 1
        if not ok:
            self.failures.append(what)
        return bool(ok)


# ---------------------------------------------------------------- F1

def f1_design(res: dict, ig: Integrity) -> dict:
    B = res.get("B")
    ig.check(res.get("complete") is True, "f1_audit.json is not marked complete")
    ig.check(isinstance(B, int) and B >= B_MIN, f"B = {B} is below the registered minimum {B_MIN}")
    ig.check(res.get("thresholds") == list(THRESHOLDS), f"thresholds {res.get('thresholds')} != {THRESHOLDS}")
    ig.check(sorted(res.get("hemispheres", {})) == sorted(KEYS),
             f"hemisphere set {sorted(res.get('hemispheres', {}))} != {sorted(KEYS)}")
    ig.check(res.get("seed0") == SEED0 and res.get("sweeps") == SWEEPS,
             f"seed0/sweeps {res.get('seed0')}/{res.get('sweeps')} != {SEED0}/{SWEEPS}")
    ig.check(res.get("holm_family") == KEYS, "runner's Holm family is not the 7 hemispheres in order")
    for k in KEYS:
        for w in THRESHOLDS:
            cell = res["hemispheres"].get(k, {}).get("thresholds", {}).get(str(w))
            ig.check(cell is not None and len(cell["q_null"]) == B and cell.get("B") == B,
                     f"{k} w>={w}: null ensemble is not complete at B = {B}")
    return {"B": B, "thresholds": res.get("thresholds"), "hemispheres": KEYS,
            "seed0": res.get("seed0"), "sweeps": res.get("sweeps"),
            "registered_minimum_B": B_MIN, "p_floor_upper": 1 / (B + 1)}


def f1_recompute(res: dict, ig: Integrity) -> dict:
    """Point 3: statistics from the raw null Q values, cross-checked against the stored ones."""
    cells = {}
    for k in KEYS:
        for w in THRESHOLDS:
            st = res["hemispheres"][k]["thresholds"][str(w)]
            s = summarise_null(st["q_real"], st["q_null"])
            tag = f"{k} w>={w}"
            ig.check(close(s["mean"], st["q_null_mean"]), f"{tag}: null mean differs from stored")
            ig.check(close(s["sd"], st["q_null_sd"]), f"{tag}: null SD differs from stored")
            ig.check(close(s["z"], st["z"]), f"{tag}: z differs from stored")
            ig.check(close(s["relative_difference"], st["relative_difference"]),
                     f"{tag}: relative difference differs from stored")
            for p in ("p_upper", "p_lower", "p_two_sided", "rank_from_top"):
                ig.check(s[p] == st[p], f"{tag}: {p} differs from stored")
            cells[(k, w)] = {"q_real": st["q_real"], **s}
    for w in THRESHOLDS:
        adj = stats.holm([cells[(k, w)]["p_upper"] for k in KEYS])
        for k, a in zip(KEYS, adj):
            cells[(k, w)]["p_upper_holm"] = a
            ig.check(a == res["hemispheres"][k]["thresholds"][str(w)]["p_upper_holm"],
                     f"{k} w>={w}: Holm p differs from stored")
            t = res.get("table", {}).get(str(w), {}).get(k)
            ig.check(t is not None and t["p_upper"] == cells[(k, w)]["p_upper"]
                     and t["departs"] == (cells[(k, w)]["p_upper"] < ALPHA),
                     f"{k} w>={w}: stored table disagrees")
    return cells


def f1_rebuild(res: dict, cfg, ig: Integrity, spot: bool, log=print) -> dict:
    """Points 4 and 5: rebuild each thresholded matrix, recompute Q, redraw spot nulls."""
    B = res["B"]
    spot_b = sorted({0, B // 2, B - 1})
    out = {"spot_seeds_b": spot_b if spot else [], "cells": {}}
    t0 = time.time()
    for ds, side in C.HEMISPHERES:
        k = f"{ds}_{side}"
        h = res["hemispheres"][k]
        p = C.projection(cfg, ds, side)
        m = np.asarray(p.matrix, float)
        ig.check(list(p.glomeruli) == h["glomeruli"], f"{k}: glomerulus list differs from stored")
        ig.check(close(float(m.sum()), h["synapses"]), f"{k}: synapse total differs from stored")
        for w in THRESHOLDS:
            st = h["thresholds"][str(w)]
            b = m >= w
            keep = b.any(axis=0)
            a = b[:, keep]
            tag = f"{k} w>={w}"
            got = {"sha1": sha1_binary(a), "shape": list(a.shape), "nnz": int(a.sum()),
                   "n_cells_dropped": int((~keep).sum()), "q_real": q_stat(a),
                   "connections_retained": float(a.sum() / (m > 0).sum()),
                   "synapses_retained": float(m[m >= w].sum() / m.sum())}
            ig.check(got["sha1"] == st["matrix_sha1"], f"{tag}: rebuilt matrix SHA-1 differs")
            ig.check(got["shape"] == [st["n_glomeruli"], st["n_cells"]], f"{tag}: shape differs")
            ig.check(got["nnz"] == st["nnz"], f"{tag}: nnz differs")
            ig.check(got["n_cells_dropped"] == st["n_cells_dropped"], f"{tag}: dropped cells differ")
            ig.check(got["q_real"] == st["q_real"], f"{tag}: rebuilt Q_real differs")
            ig.check(close(got["connections_retained"], st["connections_retained"]),
                     f"{tag}: connections retained differs")
            ig.check(close(got["synapses_retained"], st["synapses_retained"]),
                     f"{tag}: synapses retained differs")
            spot_rows = []
            if spot:
                pa = fh.Projection("b", a.astype(np.float64), [str(i) for i in range(len(a))])
                for bb in spot_b:
                    q = fh.curveball(pa, seed=SEED0 + bb, sweeps=SWEEPS)
                    qn = q_stat(q.matrix > 0)
                    ok = qn == st["q_null"][bb]
                    ig.check(ok, f"{tag}: redrawn null b = {bb} has Q {qn}, stored {st['q_null'][bb]}")
                    spot_rows.append({"b": bb, "q": qn, "equal": bool(ok)})
            out["cells"][f"{k}|{w}"] = {**got, "spot_nulls": spot_rows}
        log(f"  rebuilt {k} ({time.time() - t0:.0f}s)")
    return out


def f1_paper_reproduction(res: dict, cells: dict, ig: Integrity) -> dict:
    """Point 6: w >= 1 against the paper's B = 100 structure test."""
    old = json.loads(PAPER_IN.read_text())
    n = int(old.get("B", 100))
    rows = {}
    for k in KEYS:
        o = old["hemispheres"][k]
        st = res["hemispheres"][k]["thresholds"]["1"]
        s = summarise_null(st["q_real"], st["q_null"][:n])
        r = {"q_real_equal": st["q_real"] == o["structure"]["q_real"],
             "null_mean_equal": close(s["mean"], o["structure"]["q_null_mean"]),
             "null_sd_equal": close(s["sd"], o["structure"]["q_null_sd"], 1e-9),
             "p_upper_equal": s["p_upper"] == o["structure"]["test"]["p_upper"],
             "nnz_equal": st["nnz"] == o["full"]["nnz"],
             "n_cells_equal": st["n_cells"] == o["full"]["n_cells"]}
        for name, ok in r.items():
            ig.check(bool(ok), f"{k}: B = {n} paper reproduction failed ({name})")
        rows[k] = {kk: bool(v) for kk, v in r.items()}
    stored = res.get("reproduction_check") or {}
    ig.check(stored.get("all_equal") is True, "runner's stored reproduction_check is not all_equal")
    return {"source": "results/connectomes.json", "B_compared": n, "hemispheres": rows,
            "all_equal": all(all(v.values()) for v in rows.values())}


def f1_summary(res: dict, cells: dict) -> dict:
    """Point 8: per-threshold and per-hemisphere descriptive tables."""
    B = res["B"]
    table, per_w = {}, {}
    for w in THRESHOLDS:
        rows = {}
        for k in KEYS:
            c = cells[(k, w)]
            st = res["hemispheres"][k]["thresholds"][str(w)]
            rows[k] = {"z": c["z"], "relative_difference": c["relative_difference"],
                       "q_real": c["q_real"], "q_null_mean": c["mean"], "q_null_sd": c["sd"],
                       "q_null_mean_se": c["se_mean"],
                       "p_upper": c["p_upper"], "p_lower": c["p_lower"],
                       "p_upper_holm": c["p_upper_holm"],
                       "departs_p_upper_lt_0.05": c["p_upper"] < ALPHA,
                       "holm_lt_0.05": c["p_upper_holm"] < ALPHA,
                       "at_p_floor": c["p_upper"] <= 1 / (B + 1) + 1e-15,
                       "n_cells": st["n_cells"], "nnz": st["nnz"],
                       "inputs_mean": st["inputs_mean"], "fan_out_cv": st["fan_out_cv"],
                       "connections_retained": st["connections_retained"],
                       "synapses_retained": st["synapses_retained"]}
        table[str(w)] = rows
        zs = [r["z"] for r in rows.values()]
        rel = [r["relative_difference"] for r in rows.values()]
        per_w[str(w)] = {"n_departing": sum(r["departs_p_upper_lt_0.05"] for r in rows.values()),
                         "n_holm_lt_0.05": sum(r["holm_lt_0.05"] for r in rows.values()),
                         "n_at_p_floor": sum(r["at_p_floor"] for r in rows.values()),
                         "not_departing": [k for k, r in rows.items() if not r["departs_p_upper_lt_0.05"]],
                         "z_min": min(zs), "z_max": max(zs),
                         "relative_difference_min": min(rel), "relative_difference_max": max(rel)}
    drop = {}
    for k in KEYS:
        a, b = res["hemispheres"][k]["thresholds"]["1"], res["hemispheres"][k]["thresholds"]["2"]
        drop[k] = {"connections_removed": a["nnz"] - b["nnz"],
                   "connections_removed_fraction": 1 - b["connections_retained"],
                   "synapses_removed_fraction": 1 - b["synapses_retained"],
                   "cells_dropped": b["n_cells_dropped"],
                   "z_w1": cells[(k, 1)]["z"], "z_w2": cells[(k, 2)]["z"],
                   "z_change": cells[(k, 2)]["z"] - cells[(k, 1)]["z"]}
    return {"per_threshold": per_w, "table": table, "single_synapse_removal_w1_to_w2": drop}


def f1_hemibrain(cells: dict, res: dict, ig: Integrity) -> dict:
    """Point 1: the registered rule."""
    k = "hemibrain_R"
    at = cells[(k, STATEMENT_W)]
    departs = at["p_upper"] < ALPHA
    stored = res.get("hemibrain_statement") or {}
    ig.check(stored.get("departs_at_w2") == departs and stored.get("revise_paper_statement") == departs,
             "runner's stored hemibrain statement disagrees with the recomputed rule")
    return {"rule": f"hemibrain Q departs from its null iff p_upper < {ALPHA} on the "
                    f"w >= {STATEMENT_W} matrix (flypath/f1_audit.py point 7)",
            "paper_sentence": PAPER_SENTENCE,
            "paper_location": "paper/flyhash.tex, Results, paragraph starting 'By $Q$ the pairing departs'",
            "departs_at_w2": bool(departs),
            "revise_paper_statement": bool(departs),
            "at_w2": {"z": at["z"], "relative_difference": at["relative_difference"],
                      "p_upper": at["p_upper"], "p_upper_holm": at["p_upper_holm"],
                      "rank_from_top": at["rank_from_top"], "B": at["B"]},
            "reported_not_rule": {str(w): {"z": cells[(k, w)]["z"],
                                           "relative_difference": cells[(k, w)]["relative_difference"],
                                           "p_upper": cells[(k, w)]["p_upper"],
                                           "p_lower": cells[(k, w)]["p_lower"],
                                           "p_upper_holm": cells[(k, w)]["p_upper_holm"]}
                                  for w in THRESHOLDS}}


def paper_q_sentence(cells: dict) -> dict:
    """Point 9: the paper's B = 100 Q macros (as report.py computes them) beside B = 1,000, w >= 1."""
    old = json.loads(PAPER_IN.read_text())["hemispheres"]
    p_old = [old[k]["structure"]["test"]["p_upper"] for k in KEYS]
    z_old = [old[k]["structure"]["z"] for k in KEYS]
    p_new = [cells[(k, 1)]["p_upper"] for k in KEYS]
    z_new = [cells[(k, 1)]["z"] for k in KEYS]

    def block(p, z, B):
        adj = stats.holm(p)
        dep = [i for i, v in enumerate(p) if v < ALPHA]
        return {"B": B, "n_departing": len(dep), "z_min": min(z), "z_max": max(z),
                "raw_p_min": min(p), "holm_min": min(adj),
                "raw_p_max_departing": max(p[i] for i in dep) if dep else None,
                "holm_max_departing": max(adj[i] for i in dep) if dep else None,
                "n_holm_lt_0.05": sum(a < ALPHA for a in adj)}

    return {"note": "descriptive; the paper's Q numbers use B = 100 at w >= 1. F1 registers only "
                    "the hemibrain rule; nothing here is a registered criterion.",
            "paper_B100_w1": block(p_old, z_old, 100),
            "audit_B1000_w1": block(p_new, z_new, cells[(KEYS[0], 1)]["B"])}


# ---------------------------------------------------------------- F2

def f2_evaluate(ig: Integrity) -> dict:
    if not F2_IN.exists():
        cdir = ROOT / "data" / "cache" / "f2_audit"
        done = sorted(p.name.split("-B")[0] for p in cdir.glob(f"*-B{F2_B}.json")) if cdir.exists() else []
        return {"status": "pending",
                "reason": "results/f2_nulls.json not written yet; rerun this script when "
                          "`python -m flypath.f2_audit` prints 'complete'",
                "runner_checkpoints_finished": done,
                "note": "no F2 number is taken from partial checkpoints (point 10)"}
    res = json.loads(F2_IN.read_text())
    recs = res.get("hemispheres", {})
    ok_design = (res.get("B") == F2_B and sorted(recs) == sorted(KEYS) and res.get("holm") is not None
                 and res["holm"].get("family") == KEYS)
    ig.check(ok_design, "f2_nulls.json: B, hemisphere set or Holm family is not as registered")
    if not ok_design:
        return {"status": "invalid", "reason": "design check failed; see integrity failures"}
    rows = {}
    for k in KEYS:
        r = recs[k]
        null = np.asarray(r["null_scores"], float)
        ig.check(len(null) == F2_B, f"F2 {k}: {len(null)} null scores, expected {F2_B}")
        s = summarise_null(r["real"], null)
        ig.check(close(s["mean"], r["null_mean"]), f"F2 {k}: null mean differs from stored")
        ig.check(close(s["relative_difference"], r["relative_difference"]),
                 f"F2 {k}: relative difference differs from stored")
        ig.check(s["p_two_sided"] == r["p_two_sided"], f"F2 {k}: two-sided p differs from stored")
        for name, v in (r.get("checks") or {}).items():
            if isinstance(v, bool):
                ig.check(v, f"F2 {k}: runner check {name} is false")
        ig.check(r["seeds"] == [SEED0, SEED0 + F2_B - 1] and r["sweeps"] == SWEEPS,
                 f"F2 {k}: seeds or sweeps not as registered")
        paper = r.get("b100_paper") or {}
        rows[k] = {"k": r["k"], "real": r["real"], "null_mean": s["mean"], "null_sd": s["sd"],
                   "null_mean_se": s["se_mean"], "z": s["z"],
                   "relative_difference": s["relative_difference"],
                   "p_two_sided": s["p_two_sided"], "p_lower": s["p_lower"], "p_upper": s["p_upper"],
                   "at_p_floor": s["p_two_sided"] <= 2 / (F2_B + 1) + 1e-15,
                   "paper_B100": {"relative_difference": paper.get("relative_difference"),
                                  "p_two_sided": paper.get("p_two_sided")}}
    adj = stats.holm([rows[k]["p_two_sided"] for k in KEYS])
    for k, a in zip(KEYS, adj):
        rows[k]["p_holm"] = a
        rows[k]["significant_holm_lt_0.05"] = a < ALPHA
        ig.check(a == res["holm"]["p_holm"][k], f"F2 {k}: Holm p differs from stored")
    rel = [rows[k]["relative_difference"] for k in KEYS]
    paper_holm = stats.holm([rows[k]["paper_B100"]["p_two_sided"] for k in KEYS]) \
        if all(rows[k]["paper_B100"]["p_two_sided"] is not None for k in KEYS) else None
    return {"status": "evaluated", "B": F2_B, "p_floor": 2 / (F2_B + 1),
            "rule": "two-sided randomization p on 2,000 curveball nulls; Holm across the 7 "
                    "hemispheres; significant iff Holm p < 0.05 (flypath/f2_audit.py points 5-6)",
            "hemispheres": rows,
            "n_significant_holm": sum(rows[k]["significant_holm_lt_0.05"] for k in KEYS),
            "n_raw_p_lt_0.05": sum(rows[k]["p_two_sided"] < ALPHA for k in KEYS),
            "n_negative": sum(v < 0 for v in rel),
            "relative_difference_min": min(rel), "relative_difference_max": max(rel),
            "min_p_holm": min(adj),
            "paper_B100_min_p_holm": min(paper_holm) if paper_holm else None,
            "replaces": "the paper's B = 100 odour results, whatever the outcome (pre-registration F2)"}


# ---------------------------------------------------------------- report

def report(v: dict, log=print) -> None:
    f1 = v["f1"]
    log("F1 weak-connection audit (descriptive), B = %d curveball nulls per cell" % f1["design"]["B"])
    log("  z by hemisphere and threshold (p_upper; * = p_upper < 0.05):")
    log("  %-12s" % "" + "".join(f"{'w>=' + str(w):>16s}" for w in THRESHOLDS))
    for k in KEYS:
        cells = [f1["summary"]["table"][str(w)][k] for w in THRESHOLDS]
        log("  %-12s" % k + "".join(
            f"{c['z']:+7.2f} ({c['p_upper']:.3f}){'*' if c['departs_p_upper_lt_0.05'] else ' '}"
            for c in cells))
    for w in THRESHOLDS:
        s = f1["summary"]["per_threshold"][str(w)]
        log(f"  w>={w}: {s['n_departing']}/7 depart, {s['n_holm_lt_0.05']}/7 Holm < 0.05, "
            f"z {s['z_min']:+.2f} to {s['z_max']:+.2f}, rel. diff "
            f"{100 * s['relative_difference_min']:+.2f}% to {100 * s['relative_difference_max']:+.2f}%")
    hb = f1["hemibrain_statement"]
    log(f"  hemibrain at w>=2: z {hb['at_w2']['z']:+.2f}, p_upper {hb['at_w2']['p_upper']:.4f} "
        f"-> departs = {hb['departs_at_w2']}; revise paper sentence = {hb['revise_paper_statement']}")
    f2 = v["f2"]
    if f2["status"] == "evaluated":
        log(f"F2 odour endpoint, B = {f2['B']}: {f2['n_significant_holm']}/7 Holm < 0.05")
        for k, r in f2["hemispheres"].items():
            old = r["paper_B100"]
            was = (f"(paper B=100: {100 * old['relative_difference']:+.2f}%, "
                   f"p {old['p_two_sided']:.4f})" if old["p_two_sided"] is not None else "")
            log(f"  {k:12s} k {r['k']:3d} rel {100 * r['relative_difference']:+6.2f}% "
                f"p {r['p_two_sided']:.4f} Holm {r['p_holm']:.4f} {was}")
    else:
        log(f"F2: {f2['status']} ({f2.get('reason', '')})")
    p = v["primary"]
    log(f"integrity: {p['integrity_checks']} checks, {len(p['integrity_failures'])} failures")
    for f in p["integrity_failures"]:
        log(f"  FAIL {f}")
    log(f"run completed as registered: {p['run_completed_as_registered']}")
    for f in f2.get("integrity_failures", []):
        log(f"  F2 FAIL {f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-spot", action="store_true", help="skip the null spot check (point 5)")
    a = ap.parse_args(argv)
    t0 = time.time()
    from flypath.config import load as load_cfg
    ig = Integrity()
    res = json.loads(F1_IN.read_text())
    design = f1_design(res, ig)
    if ig.failures:
        v = {"primary": {"run_completed_as_registered": False, "integrity_checks": ig.n_checks,
                         "integrity_failures": ig.failures}}
        Path(a.out).write_text(json.dumps(v, indent=1))
        print("\n".join(ig.failures))
        return 1
    cells = f1_recompute(res, ig)
    rebuild = f1_rebuild(res, load_cfg(), ig, spot=not a.no_spot)
    repro = f1_paper_reproduction(res, cells, ig)
    hemi = f1_hemibrain(cells, res, ig)
    summary = f1_summary(res, cells)
    q_sentence = paper_q_sentence(cells)
    ig2 = Integrity()                       # F2 checks stay out of the F1 flag (point 11)
    f2 = f2_evaluate(ig2)
    f2["integrity_checks"], f2["integrity_failures"] = ig2.n_checks, ig2.failures
    v = {"experiment": "F1 weak-connection audit (and F2 larger null ensembles when available)",
         "preregistration": PREREG,
         "evaluator": "experiments/eval_f1.py",
         "inputs": {"f1_audit.json_sha256": sha256(F1_IN),
                    "f1_audit.py_sha256": sha256(ROOT / "flypath" / "f1_audit.py"),
                    "connectomes.json_sha256": sha256(PAPER_IN),
                    "f2_nulls.json_sha256": sha256(F2_IN) if F2_IN.exists() else None},
         "primary": {"run_completed_as_registered": not ig.failures,
                     "definition": "point 11: registered design present in full (B >= 100, "
                                   "w in {1,2,3,5}, 7 hemispheres) and no integrity failure; "
                                   "F1 is descriptive and has no pass/fail outcome",
                     "integrity_checks": ig.n_checks, "integrity_failures": ig.failures},
         "f1": {"design": design,
                "hemibrain_statement": hemi,
                "summary": summary,
                "paper_q_sentence_B100_vs_B1000": q_sentence,
                "paper_reproduction_w1": repro,
                "rebuild": {"spot_seeds_b": rebuild["spot_seeds_b"],
                            "cells_checked": len(rebuild["cells"]),
                            "spot_nulls_checked": sum(len(c["spot_nulls"]) for c in rebuild["cells"].values()),
                            "spot_nulls_equal": all(r["equal"] for c in rebuild["cells"].values()
                                                    for r in c["spot_nulls"])}},
         "f2": f2,
         "seconds": time.time() - t0}
    Path(a.out).write_text(json.dumps(v, indent=1, default=float))
    report(v)
    print(f"wrote {a.out} ({v['seconds']:.0f}s)")
    return 0 if not (ig.failures or ig2.failures) else 1


if __name__ == "__main__":
    sys.exit(main())
