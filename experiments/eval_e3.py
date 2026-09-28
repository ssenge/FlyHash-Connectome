"""Standalone evaluation of E3 (collision law of the fly hash) against its
pre-registered pass criteria.

Pre-registration: experiments/PREREGISTRATION_2026-09-28.md, section E3 (also
UPDATES.md section 14), frozen 2026-09-28 before any run. Input:
results/e3_collision.json, written by `python -m flypath.collision`. Output:
results/e3_verdict.json and a printed report.

    PYTHONPATH=. /opt/homebrew/anaconda3/envs/flybrain/bin/python experiments/eval_e3.py

The script uses numpy and scipy only; it does not import flypath. Every verdict
is recomputed from the raw per-trial and per-bin numbers stored in the results
file. The runner's own stored `criteria` block is used only as a cross-check at
the end. Any disagreement is printed and stored, and it never overrides the
recomputed verdict.

Criteria, copied from the pre-registration
------------------------------------------
(a) Equivalence. SIFT, GloVe, MNIST raw; 10 trials x 1,000 queries; m = 10d,
    s = 0.1d, k in {4, 16}. AP@200 of the sparse binary fly hash vs
    Gaussian-WTA. Pass: |AP_fly - AP_GWTA| / AP_GWTA <= 5% in >= 5 of 6 cells.
(b) Collapse. theta_half = angle at which the mean winner-set overlap
    fraction equals 0.5, from >= 1e5 pairs binned by angle, linear
    interpolation between bins. Predictor theta_half = c / t,
    t = inverse upper-tail normal of k/m. One c fitted on MNIST random
    matrices (all k, m cells), then frozen. Held-out: k in {4, 16, 64, 256} x
    m in {20k, 10d, 40d} on SIFT, GloVe, odours (36 cells), plus the 7
    connectome matrices at their own m with k = 5%, on odour mixtures.
    Pass: |theta_half - c/t| < 3 deg in >= 90% of held-out random cells AND
    in >= 6 of 7 connectome matrices.
(c) Fan-in boundary. PCA-51 SIFT, MNIST, GloVe and odours (d = 35);
    m = 1,838; k in {16, 92}; s in {1, 2, 3, 5, 8, 13, 26, d}.
    deficit(s) = 1 - AP_fly(s) / AP_GWTA; d_eff = participation ratio of the
    input covariance. Pass on >= 3 of 4 inputs, both k:
    Spearman(s, deficit) <= -0.9; deficit > 10% for every s <= 5;
    deficit < 5% for every s >= 0.3 d_eff.

Primary criteria. The E3 section labels no criterion "primary" or
"secondary". That split exists only in E2. Parts (a), (b) and (c) are the
three registered pass criteria of E3, and each is evaluated and reported on
its own. The registration defines no rule for combining them into one E3
verdict, so this script adds none. It reports the three part verdicts, and
`primary.all_parts_pass` says only whether every part passes. It is not a
registered aggregate.

Unspecified details fixed before running
----------------------------------------
Points 1 to 4 below were fixed in the runner's docstring (flypath/collision.py,
"Unspecified details fixed before running") before the full run. This script
applies them as written there. Points 5 to 8 are fixed here. This script was
written after the full run had finished and after its stored `criteria` block
existed, so points 5 to 8 were chosen so that none of them can change a
verdict: they touch only descriptive intervals or reporting.

 1. (a) AP_fly and AP_GWTA per cell are means over the 10 trials. The 5%
    tolerance is applied to the ratio of those means (runner detail 4).
 2. (b) theta_half: fixed 2 deg bins from 0 deg. A bin's position is the mean
    angle of its pairs and its value is the mean overlap fraction. Bins with
    fewer than 100 pairs are dropped, and theta_half is the first downward
    crossing of 0.5 found by scanning upward in angle, with linear
    interpolation between adjacent retained bins. If theta_half is undefined
    (the first retained bin is already below 0.5, or the curve never falls
    below it), the cell fails. A held-out cell with t <= 0 (k/m >= 0.5) has
    no valid prediction and also fails (runner details 7 and 10). The
    stored curves are the overlap averaged over R = 5 matrix draws per
    random cell (runner detail 8).
 3. (b) c = least squares through the origin in degrees,
    c = sum(theta_i / t_i) / sum(1 / t_i^2), over the MNIST fit cells with a
    defined theta_half (runner detail 10). This script refits c from the
    stored fit cells and checks it against the frozen value. The frozen value
    is the one used for prediction.
 4. (c) deficit(s) uses trial-mean APs (ratio of means) over the 20 trials.
    d_eff is the stored participation ratio of the row-centred hash input
    (runner details 12 and 13). Spearman comes from scipy.stats.spearmanr
    over the 8 s values, and nan fails. "Pass on >= 3 of 4 inputs, both k"
    is read strictly: an input passes only if all three conditions hold at
    k = 16 and at k = 92 (runner detail 14). The per-k counts are reported
    too, labelled as the lenient reading, and they are not the verdict.
 5. Bootstrap. No E3 criterion is stated in terms of an interval. All three
    are point criteria on trial means or on the stored collision curves, so
    no verdict here uses a bootstrap. Trial-bootstrap intervals are reported
    as descriptive uncertainty, and only where independent trials exist:
    (a) with 10 trials and (c) with 20 trials. Method: 2,000 resamples of
    the trial indices with replacement, drawn once per trial count from
    numpy.random.default_rng(0) as an index matrix of shape (2000, n_trials)
    and shared by every cell with that trial count. Paired: fly and GWTA use
    the same resampled trials, because they share queries within a trial.
    Two-sided 95% percentile interval, np.quantile at 0.025 and 0.975, the
    convention of flypath/stats.py. (b) has no trials. Its 5 per-cell draws
    are matrix draws averaged inside one curve, so no bootstrap is made
    there. The min/max of the stored per-draw theta_half is shown for
    information only.
 6. (a) The interval is given for the signed relative difference
    (AP_fly - AP_GWTA) / AP_GWTA. The criterion uses its absolute value.
 7. (c) Intervals are given for deficit(s) at every s and for Spearman.
 8. Integrity checks, all failures listed in the verdict file: the results
    file is complete; the collision.py SHA-256 is identical across all
    invocations; the stored design matches the registered constants; every
    (b) cell has at least 1e5 pairs; the connectome k equals round(0.05 m);
    the recomputed theta_half equals the stored value to within 1e-9 deg;
    the refitted c equals the frozen c to within 1e-9 deg.

Deviations
----------
None.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "e3_collision.json"
VERDICT = ROOT / "results" / "e3_verdict.json"

N_BOOT = 2000
BOOT_SEED = 0
LEVEL = 0.95

# Registered constants (PREREGISTRATION_2026-09-28.md, E3). The stored design
# must match them; the thresholds below are used directly for the verdicts.
PREREG = {
    "a": {"datasets": ["sift", "glove", "mnist"], "trials": 10, "queries": 1000,
          "m_per_d": 10, "sampling": 0.1, "ks": [4, 16], "tolerance": 0.05,
          "cells_needed": 5, "n_cells": 6},
    "b": {"fit_dataset": "mnist", "heldout": ["sift", "glove", "odours"],
          "ks": [4, 16, 64, 256], "ms": ["20k", "10d", "40d"], "min_pairs": 100_000,
          "level": 0.5, "tolerance_deg": 3.0, "pass_fraction": 0.90,
          "n_connectomes": 7, "connectome_sparsity": 0.05, "connectomes_needed": 6},
    "c": {"inputs": ["sift", "mnist", "glove", "odours"], "pca_dims": 51, "odour_d": 35,
          "m": 1838, "ks": [16, 92], "s_grid": [1, 2, 3, 5, 8, 13, 26, "d"],
          "spearman_max": -0.9, "low_s_max": 5, "low_s_deficit": 0.10,
          "high_s_factor": 0.3, "high_s_deficit": 0.05, "inputs_needed": 3},
}


# ----------------------------------------------------------------- helpers

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(o):
    """JSON-safe: numpy scalars to Python, non-finite floats to None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    return o


def ci(samples: np.ndarray) -> list[float]:
    a = (1 - LEVEL) / 2
    lo, hi = np.quantile(np.asarray(samples, float), [a, 1 - a])
    return [float(lo), float(hi)]


def boot_index(n_trials: int) -> np.ndarray:
    """(N_BOOT, n_trials) resampled trial indices, seed 0 (detail 5)."""
    return np.random.default_rng(BOOT_SEED).integers(0, n_trials, size=(N_BOOT, n_trials))


def trial_array(trials: dict, *path) -> np.ndarray:
    """Values at `path` for trials "0".."n-1", in trial order."""
    out = []
    for t in range(len(trials)):
        v = trials[str(t)]
        for p in path:
            v = v[p]
        out.append(v)
    return np.asarray(out, float)


def spearman(x, y) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = spearmanr(x, y).statistic
    return float(r)


def theta_half(curve: dict, min_count: int, level: float) -> tuple[float, str]:
    """Independent re-implementation of the registered estimator (detail 2)."""
    ang = np.asarray(curve["angle"], float)
    val = np.asarray(curve["value"], float)
    keep = np.asarray(curve["count"]) >= min_count
    ang, val = ang[keep], val[keep]
    if len(ang) == 0:
        return float("nan"), "no_bins"
    if val[0] < level:
        return float("nan"), "below_range"
    for i in range(len(val) - 1):
        if val[i] >= level and val[i + 1] < level:
            th = ang[i] + (val[i] - level) * (ang[i + 1] - ang[i]) / (val[i] - val[i + 1])
            return float(th), "ok"
    return float("nan"), "above_range"


def first_retained(curve: dict, min_count: int) -> float | None:
    keep = [i for i, c in enumerate(curve["count"]) if c >= min_count]
    return float(curve["angle"][keep[0]]) if keep else None


# ----------------------------------------------------------------- checks

def integrity(res: dict) -> list[dict]:
    D = res["design"]
    chk = []

    def add(name, ok, detail=""):
        chk.append({"check": name, "ok": bool(ok), "detail": detail})

    add("results complete", res.get("complete") is True)
    shas = {inv["code_sha256"] for inv in res["invocations"]}
    add("collision.py SHA-256 identical across invocations",
        len(shas) == 1 and res["code_sha256_at_creation"] in shas, ", ".join(sorted(shas)))
    here = ROOT / "flypath" / "collision.py"
    if here.exists():
        add("collision.py on disk unchanged since the run", sha256(here) in shas, sha256(here))
    P, A = PREREG["a"], D["a"]
    add("(a) design", A["datasets"] == P["datasets"] and A["trials"] == P["trials"]
        and A["ks"] == P["ks"] and A["m_per_d"] == P["m_per_d"]
        and D["queries"] == P["queries"] and D["sampling"] == P["sampling"]
        and A["tolerance"] == P["tolerance"] and A["cells_needed"] == P["cells_needed"])
    for name in P["datasets"]:
        e = res["a"][name]
        add(f"(a) {name}: m = 10d, s = round(0.1d), {P['trials']} trials",
            e["m"] == 10 * e["d"] and e["s"] == max(1, round(0.1 * e["d"]))
            and len(e["trials"]) == P["trials"], f"d={e['d']} m={e['m']} s={e['s']}")
    P, B = PREREG["b"], D["b"]
    add("(b) design", B["fit_dataset"] == P["fit_dataset"] and B["heldout"] == P["heldout"]
        and B["ks"] == P["ks"] and B["ms"] == P["ms"] and B["min_pairs"] >= P["min_pairs"]
        and B["level"] == P["level"] and B["tolerance_deg"] == P["tolerance_deg"]
        and B["pass_fraction"] == P["pass_fraction"]
        and len(B["hemispheres"]) == P["n_connectomes"]
        and B["connectome_sparsity"] == P["connectome_sparsity"]
        and B["connectomes_needed"] == P["connectomes_needed"])
    cells = res["b"]["cells"]
    few = [k for k, c in cells.items() if c["n_pairs"] < P["min_pairs"]]
    add("(b) every cell has >= 1e5 pairs", not few, ", ".join(few))
    bad_k = [k for k, c in cells.items() if c["role"] == "connectome"
             and c["k"] != int(round(P["connectome_sparsity"] * c["m"]))]
    add("(b) connectome k = round(0.05 m)", not bad_k, ", ".join(bad_k))
    bad_m = []
    for c in cells.values():
        if c["role"] in ("fit", "heldout"):
            want = {"20k": 20 * c["k"], "10d": 10 * c["d"], "40d": 40 * c["d"]}[c["m_label"]]
            if c["m"] != want or abs(c["t"] - norm.isf(c["k"] / c["m"])) > 1e-12:
                bad_m.append(c["key"])
    add("(b) random-cell m and t = isf(k/m) as labelled", not bad_m, ", ".join(bad_m))
    P, C = PREREG["c"], D["c"]
    add("(c) design", C["inputs"] == P["inputs"] and C["pca_dims"] == P["pca_dims"]
        and C["m"] == P["m"] and C["ks"] == P["ks"] and C["s_grid"] == P["s_grid"]
        and C["spearman_max"] == P["spearman_max"] and C["low_s_max"] == P["low_s_max"]
        and C["low_s_deficit"] == P["low_s_deficit"] and C["high_s_factor"] == P["high_s_factor"]
        and C["high_s_deficit"] == P["high_s_deficit"] and C["inputs_needed"] == P["inputs_needed"])
    for name in P["inputs"]:
        e = res["c"][name]
        d_want = P["odour_d"] if name == "odours" else P["pca_dims"]
        s_want = sorted({d_want if s == "d" else min(int(s), d_want) for s in P["s_grid"]})
        add(f"(c) {name}: d, m, s grid, {C['trials']} trials",
            e["d"] == d_want and e["m"] == P["m"] and e["s_grid"] == s_want
            and len(e["trials"]) == C["trials"], f"d={e['d']} s={e['s_grid']}")
    return chk


# ----------------------------------------------------------------- (a)

def eval_a(res: dict) -> dict:
    P = PREREG["a"]
    rows = []
    for name in P["datasets"]:
        tr = res["a"][name]["trials"]
        idx = boot_index(len(tr))
        for k in P["ks"]:
            f = trial_array(tr, "fly", str(k), "ap")
            g = trial_array(tr, "gwta", str(k), "ap")
            fm, gm = f.mean(), g.mean()
            signed = (fm - gm) / gm
            rel = abs(fm - gm) / gm
            bs = (f[idx].mean(1) - g[idx].mean(1)) / g[idx].mean(1)
            rows.append({"dataset": name, "k": k, "trials": len(f),
                         "ap_fly": fm, "ap_gwta": gm,
                         "relative_difference": rel, "signed_relative_difference": signed,
                         "signed_relative_difference_ci95_descriptive": ci(bs),
                         "pass": bool(rel <= P["tolerance"])})
    n = sum(r["pass"] for r in rows)
    return {"criterion": "|AP_fly - AP_GWTA| / AP_GWTA <= 5% in >= 5 of 6 cells",
            "cells": rows, "n_pass": n, "n_cells": len(rows), "needed": P["cells_needed"],
            "pass": bool(len(rows) == P["n_cells"] and n >= P["cells_needed"])}


# ----------------------------------------------------------------- (b)

def eval_b(res: dict, checks: list) -> dict:
    P, B = PREREG["b"], res["design"]["b"]
    cells = res["b"]["cells"]
    min_bin, lvl = B["min_bin"], P["level"]

    # recompute theta_half for every cell from its stored curve
    th = {}
    mism = []
    for key, c in cells.items():
        v, status = theta_half(c["curve"], min_bin, lvl)
        th[key] = (v, status)
        stored = c["theta_half"]
        same = (stored is None and not math.isfinite(v)) or \
               (stored is not None and math.isfinite(v) and abs(stored - v) < 1e-9)
        if not same or status != c["status"]:
            mism.append(f"{key}: stored {stored} ({c['status']}), recomputed {v} ({status})")
    checks.append({"check": "(b) recomputed theta_half equals stored (all 55 cells)",
                   "ok": not mism, "detail": "; ".join(mism)})

    # c: refit from the MNIST fit cells and compare with the frozen value
    fit = [c for c in cells.values() if c["role"] == "fit"]
    use = [c for c in fit if math.isfinite(th[c["key"]][0]) and c["t"] > 0]
    theta = np.array([th[c["key"]][0] for c in use])
    t = np.array([c["t"] for c in use])
    c_refit = float(np.sum(theta / t) / np.sum(1 / t ** 2))
    c_frozen = float(res["b"]["fit"]["c_deg"])
    checks.append({"check": "(b) refitted c equals frozen c",
                   "ok": len(fit) == len(P["ks"]) * len(P["ms"]) and abs(c_refit - c_frozen) < 1e-9,
                   "detail": f"frozen {c_frozen:.9f} deg, refit {c_refit:.9f} deg, "
                             f"{len(use)} of {len(fit)} MNIST cells used"})
    in_sample = [{"key": c["key"], "t": c["t"], "theta_half": th[c["key"]][0],
                  "predicted": c_frozen / c["t"],
                  "error_deg": th[c["key"]][0] - c_frozen / c["t"]} for c in use]

    def row(c):
        v, status = th[c["key"]]
        pred = c_frozen / c["t"] if c["t"] > 0 else None
        err = (v - pred) if (pred is not None and math.isfinite(v)) else None
        fr = first_retained(c["curve"], min_bin)
        return {"key": c["key"], "dataset": c["dataset"], "k": c["k"], "m": c["m"],
                "k_over_m": c["k"] / c["m"], "t": c["t"], "theta_half": v, "status": status,
                "predicted": pred, "error_deg": err,
                "pass": bool(err is not None and abs(err) < P["tolerance_deg"]),
                "diagnostic_first_retained_bin_deg": fr,
                "diagnostic_pass_possible": bool(pred is not None
                                                 and (fr is None or pred + P["tolerance_deg"] > fr)),
                "descriptive_theta_half_draws_min_max": (
                    [min(x for x in c["theta_half_draws"] if x is not None),
                     max(x for x in c["theta_half_draws"] if x is not None)]
                    if any(x is not None for x in c["theta_half_draws"]) else None)}

    order = {n: i for i, n in enumerate(P["heldout"])}
    held = sorted((row(c) for c in cells.values() if c["role"] == "heldout"),
                  key=lambda r: (order[r["dataset"]], r["k"], r["m"]))
    conn = [row(c) | {"hemisphere": c["hemisphere"], "d": c["d"]}
            for c in cells.values() if c["role"] == "connectome"]
    n_held = len(P["heldout"]) * len(P["ks"]) * len(P["ms"])
    n_hp = sum(r["pass"] for r in held)
    n_cp = sum(r["pass"] for r in conn)
    frac = n_hp / n_held
    need_held = math.ceil(P["pass_fraction"] * n_held - 1e-12)
    pass_held = bool(len(held) == n_held and frac >= P["pass_fraction"])
    pass_conn = bool(len(conn) == P["n_connectomes"] and n_cp >= P["connectomes_needed"])
    per_ds = {n: {"n_pass": sum(r["pass"] for r in held if r["dataset"] == n),
                  "n_cells": sum(r["dataset"] == n for r in held),
                  "n_pass_possible": sum(r["diagnostic_pass_possible"] for r in held
                                         if r["dataset"] == n)} for n in P["heldout"]}
    return {"criterion": "|theta_half - c/t| < 3 deg in >= 90% of the 36 held-out random "
                         "cells AND in >= 6 of 7 connectome matrices",
            "c_deg_frozen": c_frozen, "c_deg_refit_check": c_refit, "c_rad": math.radians(c_frozen),
            "fit_cells_in_sample": in_sample,
            "heldout": held, "heldout_n_pass": n_hp, "heldout_n_cells": n_held,
            "heldout_fraction": frac, "heldout_needed": need_held,
            "heldout_per_dataset": per_ds,
            "diagnostic_heldout_n_pass_possible": sum(r["diagnostic_pass_possible"] for r in held),
            "connectomes": conn, "connectome_n_pass": n_cp, "connectome_n": len(conn),
            "connectomes_needed": P["connectomes_needed"],
            "pass_heldout": pass_held, "pass_connectomes": pass_conn,
            "pass": bool(pass_held and pass_conn)}


# ----------------------------------------------------------------- (c)

def conditions(s: np.ndarray, deficit: np.ndarray, d_eff: float) -> dict:
    P = PREREG["c"]
    rho = spearman(s, deficit)
    low = s <= P["low_s_max"]
    high = s >= P["high_s_factor"] * d_eff
    ok_rho = bool(math.isfinite(rho) and rho <= P["spearman_max"])
    ok_low = bool(low.any() and (deficit[low] > P["low_s_deficit"]).all())
    ok_high = bool(high.any() and (deficit[high] < P["high_s_deficit"]).all())
    return {"spearman": rho, "spearman_ok": ok_rho,
            "low_s": s[low].tolist(), "low_s_deficits": deficit[low].tolist(), "low_s_ok": ok_low,
            "high_s_threshold": P["high_s_factor"] * d_eff,
            "high_s": s[high].tolist(), "high_s_deficits": deficit[high].tolist(),
            "high_s_ok": ok_high,
            "diagnostic_s_in_both_sets": s[low & high].tolist(),
            "pass": bool(ok_rho and ok_low and ok_high)}


def eval_c(res: dict) -> dict:
    P = PREREG["c"]
    per_input = {}
    for name in P["inputs"]:
        e = res["c"][name]
        tr = e["trials"]
        s = np.asarray(e["s_grid"])
        idx = boot_index(len(tr))
        ks = {}
        for k in P["ks"]:
            g = trial_array(tr, "gwta", str(k), "ap")
            f = np.stack([trial_array(tr, "fly", str(si), str(k), "ap") for si in s])  # (S, T)
            deficit = 1 - f.mean(1) / g.mean()
            bdef = 1 - f[:, idx].mean(2) / g[idx].mean(1)[None, :]                    # (S, N_BOOT)
            brho = np.array([spearman(s, bdef[:, b]) for b in range(N_BOOT)])
            ks[str(k)] = {"ap_gwta": float(g.mean()), "ap_fly": f.mean(1).tolist(),
                          "deficit": deficit.tolist(),
                          "deficit_ci95_descriptive": [ci(bdef[i]) for i in range(len(s))],
                          "spearman_ci95_descriptive": ci(brho[np.isfinite(brho)]),
                          **conditions(s, deficit, e["d_eff"])}
        per_input[name] = {"d": e["d"], "d_eff": e["d_eff"], "s": s.tolist(), "trials": len(tr),
                           "k": ks, "pass": all(v["pass"] for v in ks.values())}
    n_in = sum(v["pass"] for v in per_input.values())
    per_k = {str(k): sum(v["k"][str(k)]["pass"] for v in per_input.values()) for k in P["ks"]}
    return {"criterion": "on >= 3 of 4 inputs, at both k: Spearman(s, deficit) <= -0.9; "
                         "deficit > 10% for every s <= 5; deficit < 5% for every s >= 0.3 d_eff",
            "inputs": per_input, "n_inputs_pass_both_k": n_in, "needed": P["inputs_needed"],
            "lenient_reading_n_inputs_pass_per_k_not_the_verdict": per_k,
            "diagnostic_n_inputs_satisfiable": sum(
                not any(v["k"][str(k)]["diagnostic_s_in_both_sets"] for k in P["ks"])
                for v in per_input.values()),
            "pass": bool(n_in >= P["inputs_needed"])}


# ----------------------------------------------------------------- cross-check

def cross_check(res: dict, a: dict, b: dict, c: dict) -> list[str]:
    """Differences between this evaluation and the runner's stored `criteria`."""
    rc = res.get("criteria")
    if not rc:
        return ["runner stored no criteria block"]
    diff = []
    for x, y, what in [
        (a["n_pass"], rc["a"]["n_pass"], "(a) n_pass"), (a["pass"], rc["a"]["pass"], "(a) pass"),
        (b["heldout_n_pass"], rc["b"]["heldout_n_pass"], "(b) heldout_n_pass"),
        (b["connectome_n_pass"], rc["b"]["connectome_n_pass"], "(b) connectome_n_pass"),
        (b["pass"], rc["b"]["pass"], "(b) pass"),
        (c["n_inputs_pass_both_k"], rc["c"]["n_inputs_pass_both_k"], "(c) n_inputs_pass_both_k"),
        (c["pass"], rc["c"]["pass"], "(c) pass")]:
        if x != y:
            diff.append(f"{what}: eval {x}, runner {y}")
    for r in a["cells"]:
        s = next(q for q in rc["a"]["cells"] if q["dataset"] == r["dataset"] and q["k"] == r["k"])
        if abs(s["relative_difference"] - r["relative_difference"]) > 1e-12 or s["pass"] != r["pass"]:
            diff.append(f"(a) {r['dataset']} k={r['k']}")
    stored_b = {q["key"]: q for q in rc["b"]["heldout"] + rc["b"]["connectomes"]}
    for r in b["heldout"] + b["connectomes"]:
        q = stored_b[r["key"]]
        if q["pass"] != r["pass"]:
            diff.append(f"(b) {r['key']} pass")
    for name, v in c["inputs"].items():
        for k, kv in v["k"].items():
            q = rc["c"]["inputs"][name]["k"][k]
            if q["pass"] != kv["pass"] or not np.allclose(q["deficit"], kv["deficit"], atol=1e-12):
                diff.append(f"(c) {name} k={k}")
    return diff


# ----------------------------------------------------------------- report

def yn(v: bool) -> str:
    return "PASS" if v else "FAIL"


def fmt(v, spec=".2f") -> str:
    return "  n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def report(out: dict) -> None:
    print("E3 collision law of the fly hash: evaluation against the pre-registered criteria")
    print(f"source {out['source']['path']} (sha256 {out['source']['sha256'][:12]}), "
          f"bootstrap {N_BOOT} resamples, seed {BOOT_SEED}, descriptive only\n")
    bad = [c for c in out["integrity_checks"] if not c["ok"]]
    print(f"integrity checks: {len(out['integrity_checks']) - len(bad)}/{len(out['integrity_checks'])} ok")
    for c in bad:
        print(f"  FAILED {c['check']}: {c['detail']}")

    a = out["a"]
    print(f"\n(a) Equivalence: {a['criterion']}")
    print(f"  {'dataset':8s} {'k':>3s} {'AP_fly':>8s} {'AP_GWTA':>8s} {'|rel|':>7s} "
          f"{'signed 95% CI (descr.)':>24s}  verdict")
    for r in a["cells"]:
        lo, hi = r["signed_relative_difference_ci95_descriptive"]
        print(f"  {r['dataset']:8s} {r['k']:3d} {r['ap_fly']:8.4f} {r['ap_gwta']:8.4f} "
              f"{100 * r['relative_difference']:6.2f}% [{100 * lo:+7.2f}%, {100 * hi:+7.2f}%]"
              f"      {yn(r['pass'])}")
    print(f"  {a['n_pass']} of {a['n_cells']} cells within 5% (need {a['needed']}) -> (a) {yn(a['pass'])}")

    b = out["b"]
    print(f"\n(b) Collapse: {b['criterion']}")
    print(f"  c fitted on MNIST (frozen) = {b['c_deg_frozen']:.4f} deg "
          f"({b['c_rad']:.4f} rad); refit check {b['c_deg_refit_check']:.4f} deg")
    print(f"  {'held-out cell':24s} {'m':>6s} {'t':>6s} {'theta':>7s} {'c/t':>7s} {'err':>7s} "
          f"{'1st bin':>7s}  verdict")
    for r in b["heldout"]:
        print(f"  {r['key']:24s} {r['m']:6d} {fmt(r['t'])!s:>6s} {fmt(r['theta_half']):>7s} "
              f"{fmt(r['predicted']):>7s} {fmt(r['error_deg'], '+.2f'):>7s} "
              f"{fmt(r['diagnostic_first_retained_bin_deg']):>7s}  {yn(r['pass'])}"
              + ("" if r["status"] == "ok" else f" ({r['status']})")
              + ("" if r["t"] > 0 else " (t <= 0: no prediction)"))
    for n, v in b["heldout_per_dataset"].items():
        print(f"  {n}: {v['n_pass']}/{v['n_cells']} pass "
              f"(diagnostic: {v['n_pass_possible']} reachable given the first retained bin)")
    print(f"  held-out: {b['heldout_n_pass']}/{b['heldout_n_cells']} = "
          f"{100 * b['heldout_fraction']:.1f}% (need >= 90%, i.e. >= {b['heldout_needed']}) "
          f"-> {yn(b['pass_heldout'])}")
    print(f"  {'connectome':24s} {'m':>6s} {'k':>4s} {'theta':>7s} {'c/t':>7s} {'err':>7s}  verdict")
    for r in b["connectomes"]:
        print(f"  {r['hemisphere']:24s} {r['m']:6d} {r['k']:4d} {fmt(r['theta_half']):>7s} "
              f"{fmt(r['predicted']):>7s} {fmt(r['error_deg'], '+.2f'):>7s}  {yn(r['pass'])}")
    print(f"  connectomes: {b['connectome_n_pass']}/{b['connectome_n']} (need >= "
          f"{b['connectomes_needed']}) -> {yn(b['pass_connectomes'])}")
    print(f"  (b) requires both -> (b) {yn(b['pass'])}")

    c = out["c"]
    print(f"\n(c) Fan-in boundary: {c['criterion']}")
    for name, v in c["inputs"].items():
        print(f"  {name}: d = {v['d']}, d_eff = {v['d_eff']:.2f}, 0.3 d_eff = {0.3 * v['d_eff']:.2f}, "
              f"s = {v['s']}")
        for k, kv in v["k"].items():
            defs = " ".join(f"{100 * x:6.1f}" for x in kv["deficit"])
            lo, hi = kv["spearman_ci95_descriptive"]
            print(f"    k={k:>2s}  deficit% [{defs}]")
            print(f"          Spearman {kv['spearman']:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                  f"{'ok' if kv['spearman_ok'] else 'fails'}; "
                  f"low s {kv['low_s']} >10%: {'ok' if kv['low_s_ok'] else 'fails'}; "
                  f"high s {kv['high_s']} <5%: {'ok' if kv['high_s_ok'] else 'fails'}"
                  + (f"; s {kv['diagnostic_s_in_both_sets']} in both sets (unsatisfiable)"
                     if kv["diagnostic_s_in_both_sets"] else "")
                  + f" -> {yn(kv['pass'])}")
        print(f"    input {name} (both k): {yn(v['pass'])}")
    print(f"  {c['n_inputs_pass_both_k']} of 4 inputs pass at both k (need {c['needed']}) "
          f"-> (c) {yn(c['pass'])}")
    print(f"  lenient per-k reading (not the verdict): "
          f"{c['lenient_reading_n_inputs_pass_per_k_not_the_verdict']}")

    p = out["primary"]
    print("\nVerdict per registered part (E3 labels no primary/secondary split; all three are "
          "registered pass criteria):")
    for part, v in p["parts"].items():
        print(f"  E3({part}) {yn(v)}")
    print(f"  all registered E3 parts pass: {p['all_parts_pass']}")
    if out["cross_check_vs_runner"]:
        print("\nDISAGREEMENT with the runner's stored criteria:")
        for d in out["cross_check_vs_runner"]:
            print(f"  {d}")
    else:
        print("\ncross-check: identical verdicts and numbers to the runner's stored criteria")
    print(f"\nwrote {VERDICT.relative_to(ROOT)}")


def main() -> int:
    res = json.loads(RESULTS.read_text())
    checks = integrity(res)
    a = eval_a(res)
    b = eval_b(res, checks)
    c = eval_c(res)
    out = {
        "experiment": "E3 collision law of the fly hash",
        "preregistration": "experiments/PREREGISTRATION_2026-09-28.md, E3 (UPDATES.md s. 14)",
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "evaluator": {"path": "experiments/eval_e3.py", "sha256": sha256(Path(__file__).resolve())},
        "source": {"path": str(RESULTS.relative_to(ROOT)), "sha256": sha256(RESULTS),
                   "created": res["created"], "complete": res["complete"],
                   "code_sha256": res["code_sha256_at_creation"]},
        "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "level": LEVEL,
                      "method": "percentile, paired resampling of trials (fly and GWTA share "
                                "trials); one (2000, n_trials) index matrix per trial count",
                      "role": "descriptive only: every E3 criterion is a point criterion, "
                              "none is stated as an interval bound"},
        "integrity_checks": checks,
        "a": a, "b": b, "c": c,
        "primary": {"note": "The E3 pre-registration labels no criterion primary or secondary; "
                            "(a), (b), (c) are its three registered pass criteria and are "
                            "judged separately. No aggregate rule is registered; "
                            "all_parts_pass is reported for convenience only.",
                    "parts": {"a": a["pass"], "b": b["pass"], "c": c["pass"]},
                    "all_parts_pass": bool(a["pass"] and b["pass"] and c["pass"])},
    }
    out["cross_check_vs_runner"] = cross_check(res, a, b, c)
    VERDICT.write_text(json.dumps(clean(out), indent=1, allow_nan=False))
    report(out)
    return 0 if all(ch["ok"] for ch in checks) and not out["cross_check_vs_runner"] else 1


if __name__ == "__main__":
    sys.exit(main())
