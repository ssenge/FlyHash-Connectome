"""Standalone evaluation of E4 (spectral fan-out allocation) against its
pre-registered pass criteria, stages 1 and 2.

Pre-registration: experiments/PREREGISTRATION_2026-09-28.md, section E4 (also
UPDATES.md section 14), frozen 2026-09-28 before any run. Inputs:
results/e4_spectral.json (per-trial AP@200 of every arm, written by
`python -m flypath.spectral_run`) and results/e4_frozen.json (the stage-0
freeze of f*, written by `python -m flypath.spectral`). Output:
results/e4_verdict.json and a printed report.

    PYTHONPATH=. /opt/homebrew/anaconda3/envs/flybrain/bin/python experiments/eval_e4.py

The script uses numpy and scipy only; it does not import flypath. Every number
behind a verdict is recomputed from the per-trial AP values and the stored
integer allocations. The runner stores no intervals and no verdicts, so there
is nothing of its own to defer to. Its stored trial means ("summary") are only
cross-checked.

Criteria, copied from the pre-registration
------------------------------------------
Stage 0. Derive f*(Lambda, k) under a Gaussian-drive, order-statistic top-k
    model at nnz = 9,967 with MaleCNS R inputs per cell. Freeze it (code plus
    SHA-256 in UPDATES.md) before any new AP run.
Stage 1 (synthetic). d = 51, lambda_j ~ j^-beta, beta in {0, 0.5, 1, 1.5, 2},
    k in {4, 16, 94}, 10 trials, AP@200. Pass: at beta = 0, f* is flat
    (CV < 0.05) with AP within +-1% of even fan-out; the best grid exponent
    alpha decreases in beta and in k (Spearman <= -0.8).
Stage 2 (real). PCA-51 SIFT, MNIST, GloVe; Lambda fitted on a disjoint
    training split; 20 trials. Pass at k in {4, 16}: f* lies inside the
    best-grid-alpha 95% interval on 3 of 3 datasets, AND f* beats both even
    fan-out and the best lambda^(alpha/2) input pre-scaling (paired 95%
    intervals excluding 0) on >= 2 of 3 datasets.
Primary (as specified for this evaluation): stage 1 AND stage 2. Stage 0 has
    no pass criterion; its freeze is checked for integrity only. A near miss
    is a fail.

This script was written after the runner had completed, but before its author
looked at any AP or recall value in results/e4_spectral.json. Only the file's
structure was inspected beforehand: keys, trial counts, the design block, the
f* exponents alpha* and CVs, the validation-chosen pre-scaling exponents and
the "coincident_arms" lists. The details below were fixed at that point.


Unspecified details fixed before running
----------------------------------------
General
 1. AP of an arm in a cell is the mean over the cell's test trials of the
    stored per-trial AP@200 (10 trials in stage 1, 20 in stage 2). Validation
    trials are used only to re-check the stored pre-scaling choice.
 2. Bootstrap. Percentile bootstrap over trials, 10,000 resamples of the trial
    indices with replacement, drawn once per trial count from
    numpy.random.default_rng(0) as an index matrix of shape (10000, n_trials)
    and shared by every arm, cell and dataset with that trial count. Arms are
    therefore always resampled in pairs, because within a trial they share the
    queries, the wiring seed and the tie priorities (runner detail 1). The 95%
    interval is np.quantile at 0.025 and 0.975 (linear interpolation). The
    house convention in the other evaluators is 2,000 resamples; 10,000 is
    used here to reduce Monte Carlo error at decision boundaries. Stability:
    every interval-based decision is recomputed with (10,000, seed 1) and
    (2,000, seed 0). Any flip is reported, but it does not change the
    verdict, which uses (10,000, seed 0).
 3. "Excluding 0" means strictly: the lower bound of the paired interval is
    > 0 for "beats". A bound equal to 0 does not exclude 0.

Stage 1
 4. CV is the population coefficient of variation std(f) / mean(f) (ddof 0)
    of the stored integer f* at beta = 0, the definition the runner and the
    frozen code use. "Flat" requires CV < 0.05 at every k in {4, 16, 94}.
 5. "AP within +-1% of even fan-out" is relative: |AP(f*) / AP(even) - 1| <=
    0.01, with trial-mean APs (ratio of means), at every k in {4, 16, 94}.
    The paired bootstrap interval of the ratio is reported, not used. At
    beta = 0 every arm uses the identical matrix and input as even fan-out
    (runner detail 1), so this comparison is exact by construction and not a
    noise-level test. The report says so.
 6. "Best grid exponent alpha" in a (beta, k) cell is the grid arm
    grid:<a>, a in {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2}, with the largest
    trial-mean test AP. Exact ties go to the smallest a. f* itself is not a
    grid arm, even when alpha* lies inside the grid's range.
 7. beta = 0 is excluded from the trend. At an isotropic spectrum every grid
    arm is the same integer allocation and the same matrix (the runner's
    "coincident_arms"), so the best alpha is not identified there: all eight
    are tied. The trend uses beta in {0.5, 1, 1.5, 2} and k in {4, 16, 94},
    12 cells. As a sensitivity check (not the verdict) the trend is recomputed
    with beta = 0 included, taking the tie rule's value alpha = 0.
 8. "Decreases in beta and in k (Spearman <= -0.8)" is read as monotone
    decrease in each argument with the other held fixed, which is what
    "decreases in beta and in k" means for a function of (beta, k):
      - for every k in {4, 16, 94}: Spearman over the 4 betas of
        (beta, best alpha) <= -0.8, and
      - for every beta in {0.5, 1, 1.5, 2}: Spearman over the 3 ks of
        (k, best alpha) <= -0.8.
    All 7 correlations must hold. Spearman is scipy.stats.spearmanr (average
    ranks for ties). A correlation that is undefined (best alpha constant
    within the stratum) fails. Reported but not the verdict: the lenient
    reading (the mean of the 3 per-k correlations <= -0.8 and the mean of the
    4 per-beta correlations <= -0.8) and the pooled Spearman over the 12
    cells. The pooled reading is not used because it fails even for an
    exactly additive monotone table: the other factor's spread alone pulls
    the correlation above -0.8 (about -0.5 for k).
 9. The trend's robustness is reported as the fraction of bootstrap
    resamples in which the strict trend criterion holds, with the best alpha
    re-taken in every resample. Descriptive only.

Stage 2
10. "Pass at k in {4, 16}" means pass at k = 4 AND at k = 16, each counted
    over the 3 datasets on its own. k = 94 is reported, not scored.
11. "f* lies inside the best-grid-alpha 95% interval" (the operationalisation
    in the task specification of this evaluation): the best grid arm is
    chosen as in detail 6 on the full set of 20 test trials and then held
    fixed. Its 95% trial-bootstrap interval of mean AP is computed. The
    condition holds if the trial-mean AP of f* lies in [lower, upper],
    endpoints included. It is two-sided as written: an AP above the upper
    bound is outside. Where that happens it is labelled "above". Reported,
    not used: the one-sided reading AP(f*) >= lower bound, and the paired
    interval of AP(f*) - AP(best grid arm).
12. "f* beats even fan-out" and "f* beats the best pre-scaling": the paired
    95% interval of mean_t [AP_f*(t) - AP_other(t)] has lower bound > 0. The
    best pre-scaling is the arm "prescale_best": alpha chosen on the
    validation split for each dataset and k before any test trial ran
    (runner detail 5). A dataset counts only if f* beats both there. (With
    paired resampling, the interval of the ratio AP(f*)/AP(other) - 1
    excludes 0 exactly when the difference interval does, so the scale does
    not matter.)
13. The alpha reading (reported, not the verdict; see Deviations). This one
    compares alpha* with a 95% bootstrap interval of the best grid alpha.
    The interval is the 2.5% and 97.5% quantiles (np.quantile, method
    "lower"/"higher" so that the ends are grid values) of the best grid alpha
    re-taken in every resample. Containment is alpha_lo <= alpha* <= alpha_hi,
    with alpha* = inf larger than every grid value.

Integrity checks (all listed in the verdict file; a failure sets
`integrity.all_ok` false and `primary.valid` false, but never changes a
recomputed number)
14. Both result files are complete. flypath/spectral.py now has the SHA-256
    recorded in the freeze, and so do the runner's check at creation and every
    invocation. The runner code hash is the same across invocations. The AP
    run was created after the freeze time. The stored design matches the
    registered constants. Every cell has the registered number of test trials
    and a finite AP in [0, 1] for every arm and k. The stage-1 f* equals the
    frozen f_star. The stored CVs equal the recomputed ones. Every allocation
    sums to nnz = 9,967. prescale_best equals prescale:<a_pre> in every trial,
    and a_pre, recomputed from the validation trials, equals the stored
    choice. The arms listed as coincident have identical AP in every trial.
    The time order in stage 2 is f* first, then the pre-scaling choice, then
    completion. The stored trial means equal the recomputed ones.
15. Stage-0 record placement. The script checks whether UPDATES.md contains
    the frozen SHA-256. This is informational: the pre-registration asks for
    it there, and flypath/spectral.py records its absence as its Deviation 1.
    It is not an integrity failure of the AP run.


Deviations
----------
 1. flypath/spectral.py ("Unspecified details" 7, frozen at stage 0) says
    that "f* inside the best-grid-alpha interval" compares alpha* with that
    interval, i.e. an interval on the exponent. The specification of this
    evaluation states the criterion on the AP scale instead: AP(f*) inside
    the 95% trial-bootstrap interval of the best grid alpha. The verdict
    follows that specification (detail 11), which is also the direct reading
    of the registered sentence. The frozen alpha reading is computed and
    reported (detail 13). If the two readings disagree for any stage-2
    decision, the verdict file says so under
    stage2.reading_disagreements.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "e4_spectral.json"
FROZEN = ROOT / "results" / "e4_frozen.json"
VERDICT = ROOT / "results" / "e4_verdict.json"
SPECTRAL = ROOT / "flypath" / "spectral.py"
RUNNER = ROOT / "flypath" / "spectral_run.py"
UPDATES = ROOT / "UPDATES.md"

N_BOOT = 10_000
BOOT_SEED = 0
LEVEL = 0.95
STABILITY = ((10_000, 1), (2_000, 0))          # (resamples, seed), detail 2

GRID = ["0", "0.25", "0.5", "0.75", "1", "1.25", "1.5", "2"]   # ascending (tie rule)

# Registered constants (PREREGISTRATION_2026-09-28.md, E4) and the protocol.
PREREG = {
    "nnz": 9967,
    "stage1": {"betas": [0.0, 0.5, 1.0, 1.5, 2.0], "ks": [4, 16, 94], "d": 51, "trials": 10,
               "cv_max": 0.05, "ap_tolerance": 0.01, "spearman_max": -0.8,
               "trend_betas": ["0.5", "1", "1.5", "2"]},
    "stage2": {"datasets": ["sift", "mnist", "glove"], "pca_dims": 51, "trials": 20,
               "ks": [4, 16], "containment_needed": 3, "beats_needed": 2},
    "protocol": {"n_items": 10_000, "n_queries": 1000, "top_fraction": 0.02},
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
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def alpha_value(label: str) -> float:
    return math.inf if label == "inf" else float(label)


def boot_index(n_trials: int, draws: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """(draws, n_trials) resampled trial indices (detail 2)."""
    return np.random.default_rng(seed).integers(0, n_trials, size=(draws, n_trials))


def ci(samples: np.ndarray) -> list[float]:
    a = (1 - LEVEL) / 2
    lo, hi = np.quantile(np.asarray(samples, float), [a, 1 - a])
    return [float(lo), float(hi)]


def trials_ap(e: dict, arm: str, k: int, split: str = "test") -> np.ndarray:
    """Per-trial AP of `arm` at `k`, trials "0".."n-1" in order."""
    tr = e[split]["trials"]
    return np.array([tr[str(t)]["ap"][arm][str(k)] for t in range(len(tr))], float)


def grid_matrix(e: dict, k: int) -> np.ndarray:
    """(n_grid, n_trials) AP of the grid arms, in ascending alpha."""
    return np.stack([trials_ap(e, f"grid:{a}", k) for a in GRID])


def best_grid(G: np.ndarray) -> int:
    """Index of the grid arm with the largest trial mean; ties to the smallest alpha
    (np.argmax returns the first maximum and GRID is ascending)."""
    return int(np.argmax(G.mean(1)))


def best_grid_boot(G: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Best grid index re-taken in every resample: (draws,)."""
    means = G[:, idx].mean(2)                      # (n_grid, draws)
    return np.argmax(means, axis=0)


def spearman(x, y) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = spearmanr(x, y).statistic
    return float(r)


def spearman_rows(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman of the fixed vector x with every row of Y (average ranks);
    nan where a row is constant."""
    rx = rankdata(x)
    rx = rx - rx.mean()
    rY = rankdata(Y, axis=1)
    rY = rY - rY.mean(1, keepdims=True)
    num = (rY * rx).sum(1)
    den = np.sqrt((rY ** 2).sum(1) * (rx ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def ok_le(r: float, thr: float) -> bool:
    return bool(np.isfinite(r) and r <= thr)


# ----------------------------------------------------------------- integrity

def integrity(res: dict, fro: dict) -> dict:
    checks = []

    def check(name: str, ok: bool, detail=None):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    P = PREREG
    check("e4_spectral.json complete", res.get("complete") is True)
    check("e4_frozen.json complete", fro.get("complete") is True)
    now = sha256(SPECTRAL)
    fsha = fro["code"]["sha256"]
    check("flypath/spectral.py SHA-256 equals the freeze", now == fsha, {"now": now, "frozen": fsha})
    check("AP run records the same freeze", res["frozen"]["spectral_sha256"] == fsha)
    inv = res["invocations"]
    check("every invocation verified the frozen spectral.py",
          all(i.get("spectral_sha256_verified") == fsha for i in inv), len(inv))
    rsh = {i.get("code_sha256") for i in inv}
    check("spectral_run.py SHA-256 identical across invocations",
          len(rsh) == 1 and res["code_sha256_at_creation"] in rsh, sorted(rsh))
    check("AP run created after the freeze",
          res["created_utc"] >= fro["frozen_utc"],
          {"frozen_utc": fro["frozen_utc"], "ap_created_utc": res["created_utc"]})

    D = res["design"]
    pr = P["protocol"]
    design_ok = (D["n_items"] == pr["n_items"] and D["n_queries"] == pr["n_queries"]
                 and D["top_fraction"] == pr["top_fraction"] and D["d"] == P["stage1"]["d"]
                 and D["nnz"] == P["nnz"] and set(P["stage1"]["ks"]) <= set(D["ks"])
                 and D["grid_alphas"] == GRID
                 and D["stage1"]["betas"] == P["stage1"]["betas"]
                 and D["stage1"]["trials"] == P["stage1"]["trials"]
                 and D["stage2"]["datasets"] == P["stage2"]["datasets"]
                 and D["stage2"]["trials"] == P["stage2"]["trials"]
                 and D["stage2"]["pca_dims"] == P["stage2"]["pca_dims"])
    check("stored design matches the registered constants", design_ok)
    check("inputs per cell: nnz = 9,967",
          res["inputs_per_cell"]["nnz"] == P["nnz"] and fro["inputs_per_cell"]["nnz"] == P["nnz"]
          and res["inputs_per_cell"]["sha1"] == fro["inputs_per_cell"]["sha1"])

    ks = D["ks"]
    arms_needed = (["even", "fstar", "prescale_best"] + [f"grid:{a}" for a in GRID]
                   + [f"prescale:{a}" for a in D["prescale_alphas"]])
    for stage, n_tr in (("stage1", P["stage1"]["trials"]), ("stage2", P["stage2"]["trials"])):
        for name, e in res[stage].items():
            tr = e["test"]["trials"]
            good = sorted(tr, key=int) == [str(t) for t in range(n_tr)]
            vals = [tr[t]["ap"][a][str(k)] for t in tr for a in arms_needed for k in ks]
            good &= all(v is not None and 0.0 <= v <= 1.0 for v in vals)
            check(f"{stage} {name}: {n_tr} test trials, finite AP in [0,1] for every arm and k",
                  good)
            # allocations
            alloc_ok = all(sum(v["fanout"]) == P["nnz"] and min(v["fanout"]) >= 0
                           and abs(v["cv"] - float(np.std(v["fanout"]) / np.mean(v["fanout"])))
                           < 1e-12 for v in e["allocations"].values())
            fs_ok = all(e["allocations"][f"fstar:{k}"]["fanout"] == e["fstar"][str(k)]["fanout"]
                        and abs(e["fstar"][str(k)]["cv"]
                                - float(np.std(e["fstar"][str(k)]["fanout"])
                                        / np.mean(e["fstar"][str(k)]["fanout"]))) < 1e-9
                        for k in ks)
            check(f"{stage} {name}: allocations sum to nnz, stored CVs recomputed, f* consistent",
                  alloc_ok and fs_ok)
            # pre-scaling choice
            ch = e["prescale_choice"]
            vt = e["validation"]["trials"]
            rec_ok = True
            for k in ks:
                means = [np.mean([vt[t]["ap"][f"prescale:{a}"][str(k)] for t in vt])
                         for a in D["prescale_alphas"]]
                a_pre = D["prescale_alphas"][int(np.argmax(means))]
                rec_ok &= a_pre == ch[str(k)]["alpha"]
                rec_ok &= np.array_equal(trials_ap(e, "prescale_best", k),
                                         trials_ap(e, f"prescale:{ch[str(k)]['alpha']}", k))
            check(f"{stage} {name}: pre-scaling choice recomputed from validation; "
                  "prescale_best equals it in every trial", rec_ok,
                  {k: ch[str(k)]["alpha"] for k in ks})
            # coincident arms
            co = e["coincident_arms"]
            co_ok = all(np.array_equal(trials_ap(e, "even", k), trials_ap(e, a, k))
                        for a in co["even"] for k in ks)
            for k in ks:
                co_ok &= all(np.array_equal(trials_ap(e, "fstar", k), trials_ap(e, a, k))
                             for a in co[f"fstar@{k}"])
            check(f"{stage} {name}: arms listed as coincident have identical AP in every trial",
                  co_ok)
            # summary
            sm = e["summary"]["mean_ap"]
            sm_ok = all(abs(sm[a][str(k)] - trials_ap(e, a, k).mean()) < 1e-12
                        for a in arms_needed for k in ks)
            check(f"{stage} {name}: stored trial means equal recomputed", sm_ok)
            order_ok = e["prescale_choice_utc"] <= e["completed_utc"]
            if stage == "stage2":
                order_ok &= e["fstar_utc"] <= e["prescale_choice_utc"]
            check(f"{stage} {name}: time order (f*, pre-scaling choice, completion)", order_ok)
    for b, e in res["stage1"].items():
        fr = fro["synthetic"][b]
        check(f"stage1 beta {b}: f* equals the frozen f_star",
              all(e["fstar"][str(k)]["fanout"] == fr["f_star"][str(k)] for k in ks)
              and np.allclose(e["lambda"], fr["lambda"], rtol=1e-12, atol=0))
    for name, e in res["stage2"].items():
        sp = e["splits"]
        check(f"stage2 {name}: training and validation splits of 10,000",
              sp["n_train"] == 10_000 and sp["n_validation"] == 10_000 and sp["n_items"] == 10_000)
    failures = [c["check"] for c in checks if not c["ok"]]
    upd = UPDATES.read_text() if UPDATES.exists() else ""
    notes = {"updates_md_contains_frozen_sha256": fsha in upd,
             "note": "stage 0 asks for the freeze record (code plus SHA-256) in UPDATES.md; it is "
                     "in results/e4_frozen.json (frozen_utc " + str(fro["frozen_utc"]) + "), "
                     "recorded as Deviation 1 in flypath/spectral.py; informational (detail 15)"}
    return {"checks": checks, "failures": failures, "all_ok": not failures, "stage0_record": notes}


# ----------------------------------------------------------------- stage 1

def stage1(res: dict, idx10: np.ndarray) -> dict:
    P = PREREG["stage1"]
    S = res["stage1"]
    ks = P["ks"]
    out = {"beta0": {}, "best_alpha": {}, "mean_ap": {}}

    # beta = 0: flatness and AP
    e0 = S["0"]
    for k in ks:
        f = np.asarray(e0["fstar"][str(k)]["fanout"], float)
        cv = float(f.std() / f.mean())
        af, ae = trials_ap(e0, "fstar", k), trials_ap(e0, "even", k)
        rel = float(af.mean() / ae.mean() - 1)
        boot = af[idx10].mean(1) / ae[idx10].mean(1) - 1
        out["beta0"][str(k)] = {
            "alpha_star": e0["fstar"][str(k)]["alpha_star"], "cv": cv,
            "cv_pass": cv < P["cv_max"],
            "ap_fstar": float(af.mean()), "ap_even": float(ae.mean()),
            "relative_difference": rel, "relative_difference_ci95": ci(boot),
            "ap_pass": abs(rel) <= P["ap_tolerance"],
            "identical_every_trial": bool(np.array_equal(af, ae)),
        }
    grid_same_at_0 = all(np.array_equal(trials_ap(e0, f"grid:{a}", k), trials_ap(e0, "even", k))
                         for a in GRID for k in ks)
    out["beta0_note"] = ("at beta = 0 every arm hashes with the identical matrix and input as even "
                         "fan-out (runner detail 1), so AP(f*) = AP(even) exactly by construction; "
                         "this is not a noise-level test. All grid arms tied: "
                         + str(grid_same_at_0))
    flat_pass = all(v["cv_pass"] for v in out["beta0"].values())
    ap_pass = all(v["ap_pass"] for v in out["beta0"].values())

    # best grid alpha per cell
    for b in P["trend_betas"] + ["0"]:
        e = S[b]
        out["best_alpha"][b] = {}
        out["mean_ap"][b] = {}
        for k in ks:
            G = grid_matrix(e, k)
            i = best_grid(G)
            bb = best_grid_boot(G, idx10)
            freq = {GRID[j]: float((bb == j).mean()) for j in range(len(GRID)) if (bb == j).any()}
            out["best_alpha"][b][str(k)] = {
                "alpha": GRID[i], "grid_mean_ap": dict(zip(GRID, G.mean(1).tolist())),
                "all_tied": bool(np.all(G.mean(1) == G.mean(1)[0])),
                "bootstrap_frequency": freq,
                "fstar_alpha_star": e["fstar"][str(k)]["alpha_star"]}
            out["mean_ap"][b][str(k)] = {a: float(trials_ap(e, a, k).mean())
                                         for a in ["even", "fstar", "prescale_best"]
                                         + [f"grid:{g}" for g in GRID]
                                         + [f"prescale:{g}" for g in GRID]}

    # trend (details 7-8)
    tb = P["trend_betas"]
    bnum = np.array([float(b) for b in tb])
    A = np.array([[float(out["best_alpha"][b][str(k)]["alpha"]) for k in ks] for b in tb])  # (4, 3)
    thr = P["spearman_max"]
    rho_beta = {str(k): spearman(bnum, A[:, j]) for j, k in enumerate(ks)}
    rho_k = {b: spearman(np.array(ks, float), A[i]) for i, b in enumerate(tb)}
    strict = all(ok_le(r, thr) for r in rho_beta.values()) and all(ok_le(r, thr) for r in rho_k.values())
    mean_rb = float(np.mean(list(rho_beta.values())))
    mean_rk = float(np.mean(list(rho_k.values())))
    lenient = ok_le(mean_rb, thr) and ok_le(mean_rk, thr)
    BB, KK = np.meshgrid(bnum, np.array(ks, float), indexing="ij")
    pooled = {"beta": spearman(BB.ravel(), A.ravel()), "k": spearman(KK.ravel(), A.ravel())}

    # sensitivity: beta = 0 included with the tie rule's alpha = 0
    b5 = np.array([0.0] + list(bnum))
    A5 = np.vstack([[float(out["best_alpha"]["0"][str(k)]["alpha"]) for k in ks], A])
    rho_beta5 = {str(k): spearman(b5, A5[:, j]) for j, k in enumerate(ks)}
    rho_k5 = {"0": spearman(np.array(ks, float), A5[0]), **rho_k}
    strict5 = (all(ok_le(r, thr) for r in rho_beta5.values())
               and all(ok_le(r, thr) for r in rho_k5.values()))

    # bootstrap robustness of the strict trend (detail 9)
    Ab = np.stack([np.stack([np.asarray(GRID, float)[best_grid_boot(grid_matrix(S[b], k), idx10)]
                             for k in ks], axis=1) for b in tb], axis=1)   # (draws, 4, 3)
    okb = np.ones(len(idx10), bool)
    for j in range(len(ks)):
        r = spearman_rows(bnum, Ab[:, :, j])
        okb &= np.isfinite(r) & (r <= thr)
    for i in range(len(tb)):
        r = spearman_rows(np.array(ks, float), Ab[:, i, :])
        okb &= np.isfinite(r) & (r <= thr)

    out["trend"] = {
        "betas": tb, "ks": ks, "best_alpha_table": {b: dict(zip(map(str, ks), A[i].tolist()))
                                                    for i, b in enumerate(tb)},
        "spearman_over_beta_per_k": rho_beta, "spearman_over_k_per_beta": rho_k,
        "threshold": thr, "strict_pass": strict,
        "lenient": {"mean_over_beta": mean_rb, "mean_over_k": mean_rk, "pass": lenient,
                    "used_for_verdict": False},
        "pooled": {**pooled, "used_for_verdict": False},
        "with_beta0_alpha0": {"spearman_over_beta_per_k": rho_beta5,
                              "spearman_over_k_per_beta": rho_k5, "strict_pass": strict5,
                              "used_for_verdict": False},
        "bootstrap_fraction_strict_pass": float(okb.mean()),
    }
    out["criteria"] = {"flat_cv_all_k": flat_pass, "ap_within_1pct_all_k": ap_pass,
                       "trend_strict": strict}
    out["pass"] = bool(flat_pass and ap_pass and strict)
    return out


# ----------------------------------------------------------------- stage 2

def stage2_cell(e: dict, k: int, idx: np.ndarray, full: bool = True) -> dict:
    G = grid_matrix(e, k)
    i = best_grid(G)
    gbest = G[i]
    lo, hi = ci(gbest[idx].mean(1))
    af = trials_ap(e, "fstar", k)
    m_f = float(af.mean())
    inside = lo <= m_f <= hi
    pos = "inside" if inside else ("above" if m_f > hi else "below")
    out = {"best_grid_alpha": GRID[i], "ap_best_grid": float(gbest.mean()),
           "best_grid_ci95": [lo, hi], "ap_fstar": m_f, "fstar_position": pos,
           "containment": bool(inside), "containment_one_sided": bool(m_f >= lo)}
    for other in ("even", "prescale_best"):
        ao = trials_ap(e, other, k)
        d = af - ao
        dci = ci(d[idx].mean(1))
        out[f"vs_{other}"] = {"ap_other": float(ao.mean()), "difference": float(d.mean()),
                              "difference_ci95": dci,
                              "relative": float(af.mean() / ao.mean() - 1),
                              "beats": bool(dci[0] > 0)}
    out["beats_both"] = bool(out["vs_even"]["beats"] and out["vs_prescale_best"]["beats"])
    if not full:
        return out
    d = af - gbest
    out["vs_best_grid_paired"] = {"difference": float(d.mean()), "difference_ci95": ci(d[idx].mean(1))}
    # alpha reading (detail 13)
    ab = np.asarray(GRID, float)[best_grid_boot(G, idx)]
    a_lo = float(np.quantile(ab, (1 - LEVEL) / 2, method="lower"))
    a_hi = float(np.quantile(ab, 1 - (1 - LEVEL) / 2, method="higher"))
    a_star = e["fstar"][str(k)]["alpha_star"]
    out["alpha_reading"] = {"alpha_star": a_star, "best_alpha_ci95": [a_lo, a_hi],
                            "containment": bool(a_lo <= alpha_value(a_star) <= a_hi),
                            "used_for_verdict": False}
    # descriptive: best pre-scaling chosen on test (oracle), not a criterion
    P = np.stack([trials_ap(e, f"prescale:{a}", k) for a in GRID])
    j = best_grid(P)
    out["test_oracle_prescale"] = {"alpha": GRID[j], "ap": float(P[j].mean()),
                                   "validation_choice": e["prescale_choice"][str(k)]["alpha"]}
    out["mean_ap"] = {a: float(trials_ap(e, a, k).mean())
                      for a in ["even", "fstar", "prescale_best"] + [f"grid:{g}" for g in GRID]
                      + [f"prescale:{g}" for g in GRID]}
    out["fstar_cv"] = e["fstar"][str(k)]["cv"]
    return out


def stage2(res: dict, idx20: np.ndarray) -> dict:
    P = PREREG["stage2"]
    S = res["stage2"]
    out = {"k": {}, "descriptive_k94": {}}
    for k in P["ks"]:
        cells = {name: stage2_cell(S[name], k, idx20) for name in P["datasets"]}
        n_in = sum(c["containment"] for c in cells.values())
        n_beat = sum(c["beats_both"] for c in cells.values())
        out["k"][str(k)] = {
            "datasets": cells, "containment_count": n_in, "beats_both_count": n_beat,
            "containment_pass": n_in >= P["containment_needed"],
            "beats_pass": n_beat >= P["beats_needed"],
            "pass": bool(n_in >= P["containment_needed"] and n_beat >= P["beats_needed"]),
            "alpha_reading_containment_count": sum(c["alpha_reading"]["containment"]
                                                   for c in cells.values()),
            "one_sided_containment_count": sum(c["containment_one_sided"] for c in cells.values()),
        }
    for name in P["datasets"]:
        out["descriptive_k94"][name] = stage2_cell(S[name], 94, idx20)
    dis = []
    for k, v in out["k"].items():
        for name, c in v["datasets"].items():
            if c["containment"] != c["alpha_reading"]["containment"]:
                dis.append(f"k={k} {name}: AP reading {'inside' if c['containment'] else 'outside'}, "
                           f"alpha reading {'inside' if c['alpha_reading']['containment'] else 'outside'}")
    out["reading_disagreements"] = dis
    out["pass"] = bool(all(v["pass"] for v in out["k"].values()))
    return out


def stability(res: dict, main: dict) -> dict:
    """Recompute every interval-based stage-2 decision with other bootstrap draws (detail 2)."""
    P = PREREG["stage2"]
    rows = {}
    flips = []
    for draws, seed in STABILITY:
        idx = boot_index(P["trials"], draws, seed)
        tag = f"{draws}@seed{seed}"
        rows[tag] = {}
        for k in P["ks"]:
            for name in P["datasets"]:
                c = stage2_cell(res["stage2"][name], k, idx, full=False)
                m = main["k"][str(k)]["datasets"][name]
                dec = {"containment": c["containment"], "beats_even": c["vs_even"]["beats"],
                       "beats_prescale_best": c["vs_prescale_best"]["beats"]}
                ref = {"containment": m["containment"], "beats_even": m["vs_even"]["beats"],
                       "beats_prescale_best": m["vs_prescale_best"]["beats"]}
                rows[tag][f"k={k} {name}"] = dec
                flips += [f"{tag} k={k} {name} {d}" for d in dec if dec[d] != ref[d]]
    return {"decisions": rows, "flips": flips, "stable": not flips}


# ----------------------------------------------------------------- report

def report(v: dict) -> None:
    p = print
    s1, s2 = v["stage1"], v["stage2"]
    p("E4 spectral fan-out allocation: evaluation against the pre-registered criteria")
    p(f"bootstrap {N_BOOT} resamples over trials, seed {BOOT_SEED}, paired, percentile 95%")
    p("")
    p("Integrity: " + ("all checks passed" if v["integrity"]["all_ok"]
                       else "FAILURES: " + "; ".join(v["integrity"]["failures"])))
    p(f"  stage-0 SHA-256 in UPDATES.md: {v['integrity']['stage0_record']['updates_md_contains_frozen_sha256']}"
      " (informational; see detail 15)")
    p("")
    p("Stage 1 (synthetic)")
    p("  beta = 0:   k   alpha*   CV       AP(f*)   AP(even)  rel.diff   identical")
    for k, c in s1["beta0"].items():
        p(f"            {k:>3}  {c['alpha_star']:>5}  {c['cv']:.4f}  {c['ap_fstar']:.4f}   "
          f"{c['ap_even']:.4f}   {100 * c['relative_difference']:+.3f}%   {c['identical_every_trial']}")
    p("  " + s1["beta0_note"])
    p("  best grid alpha (trial-mean test AP; beta = 0 unidentified, all tied):")
    p("     beta   " + "  ".join(f"k={k:<4}" for k in PREREG["stage1"]["ks"])
      + "   (f* alpha*)")
    for b in PREREG["stage1"]["trend_betas"]:
        row = s1["best_alpha"][b]
        p(f"     {b:>4}   " + "  ".join(f"{row[str(k)]['alpha']:<6}" for k in PREREG["stage1"]["ks"])
          + "   (" + ", ".join(row[str(k)]["fstar_alpha_star"] for k in PREREG["stage1"]["ks"]) + ")")
    t = s1["trend"]
    p("  Spearman over beta, per k: " + ", ".join(f"k={k} {r:+.3f}" if r is not None and np.isfinite(r)
                                                  else f"k={k} nan"
                                                  for k, r in t["spearman_over_beta_per_k"].items()))
    p("  Spearman over k, per beta: " + ", ".join(f"b={b} {r:+.3f}" if r is not None and np.isfinite(r)
                                                  else f"b={b} nan"
                                                  for b, r in t["spearman_over_k_per_beta"].items()))
    p(f"  strict trend (all <= -0.8): {t['strict_pass']}; lenient (means {t['lenient']['mean_over_beta']:+.3f}, "
      f"{t['lenient']['mean_over_k']:+.3f}): {t['lenient']['pass']}; pooled beta {t['pooled']['beta']:+.3f}, "
      f"k {t['pooled']['k']:+.3f}; with beta=0 (alpha 0): {t['with_beta0_alpha0']['strict_pass']}")
    p(f"  bootstrap fraction of resamples with strict trend: {t['bootstrap_fraction_strict_pass']:.3f}")
    p(f"  STAGE 1: {'PASS' if s1['pass'] else 'FAIL'}  {s1['criteria']}")
    p("")
    p("Stage 2 (PCA-51 real inputs)")
    for k, kv in s2["k"].items():
        p(f"  k = {k}")
        p("    dataset  best a  AP best [95% CI]            AP f* (a*)       pos     "
          "f*-even [95% CI]               f*-pre* (a_pre) [95% CI]")
        for name, c in kv["datasets"].items():
            ve, vp = c["vs_even"], c["vs_prescale_best"]
            p(f"    {name:<7}  {c['best_grid_alpha']:<6}  {c['ap_best_grid']:.4f} "
              f"[{c['best_grid_ci95'][0]:.4f}, {c['best_grid_ci95'][1]:.4f}]  "
              f"{c['ap_fstar']:.4f} ({c['alpha_reading']['alpha_star']:>4})  {c['fstar_position']:<6}  "
              f"{ve['difference']:+.4f} [{ve['difference_ci95'][0]:+.4f}, {ve['difference_ci95'][1]:+.4f}]  "
              f"{vp['difference']:+.4f} ({c['test_oracle_prescale']['validation_choice']}) "
              f"[{vp['difference_ci95'][0]:+.4f}, {vp['difference_ci95'][1]:+.4f}]")
        p(f"    containment {kv['containment_count']}/3 (need 3), beats both {kv['beats_both_count']}/3 "
          f"(need 2) -> {'PASS' if kv['pass'] else 'FAIL'}; alpha reading containment "
          f"{kv['alpha_reading_containment_count']}/3; one-sided containment "
          f"{kv['one_sided_containment_count']}/3")
    if s2["reading_disagreements"]:
        p("  AP vs alpha reading disagree: " + "; ".join(s2["reading_disagreements"]))
    p("  k = 94 (descriptive): " + "; ".join(
        f"{n} f* {c['ap_fstar']:.4f} even {c['vs_even']['ap_other']:.4f} "
        f"pre* {c['vs_prescale_best']['ap_other']:.4f} best grid {c['ap_best_grid']:.4f} (a={c['best_grid_alpha']})"
        for n, c in s2["descriptive_k94"].items()))
    p(f"  STAGE 2: {'PASS' if s2['pass'] else 'FAIL'}")
    p(f"  Monte Carlo stability: {'no decision flips' if v['mc_stability']['stable'] else v['mc_stability']['flips']}")
    p("")
    p(f"PRIMARY (stage 1 AND stage 2): {v['primary']['verdict']}"
      + ("" if v["primary"]["valid"] else "  [integrity failures: verdict not valid]"))


# ----------------------------------------------------------------- main

def main() -> int:
    t0 = time.time()
    res = json.loads(RESULTS.read_text())
    fro = json.loads(FROZEN.read_text())
    idx10 = boot_index(PREREG["stage1"]["trials"])
    idx20 = boot_index(PREREG["stage2"]["trials"])
    integ = integrity(res, fro)
    s1 = stage1(res, idx10)
    s2 = stage2(res, idx20)
    stab = stability(res, s2)
    passed = bool(s1["pass"] and s2["pass"])
    v = {
        "experiment": "E4 spectral fan-out allocation: evaluation of stages 1 and 2",
        "preregistration": "experiments/PREREGISTRATION_2026-09-28.md, E4",
        "evaluator": {"path": "experiments/eval_e4.py", "sha256": sha256(Path(__file__))},
        "inputs": {"e4_spectral": {"path": "results/e4_spectral.json", "sha256": sha256(RESULTS)},
                   "e4_frozen": {"path": "results/e4_frozen.json", "sha256": sha256(FROZEN)},
                   "spectral_py_sha256": sha256(SPECTRAL), "spectral_run_py_sha256": sha256(RUNNER)},
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "interval": "percentile 2.5/97.5",
                      "paired": True, "stability_checks": [list(s) for s in STABILITY]},
        "criteria_text": {
            "stage1": "at beta = 0, f* is flat (CV < 0.05) with AP within +-1% of even fan-out; the "
                      "best grid exponent alpha decreases in beta and in k (Spearman <= -0.8)",
            "stage2": "pass at k in {4, 16}: f* (AP) lies inside the best-grid-alpha 95% trial-bootstrap "
                      "interval on 3 of 3 datasets, AND f* beats both even fan-out and the best "
                      "(validation-chosen) lambda^(alpha/2) input pre-scaling (paired 95% intervals "
                      "excluding 0) on >= 2 of 3 datasets",
            "primary": "stage 1 AND stage 2; a near miss is a fail"},
        "integrity": integ,
        "stage1": s1,
        "stage2": s2,
        "mc_stability": stab,
        "primary": {"stage1_pass": s1["pass"], "stage2_pass": s2["pass"], "pass": passed,
                    "verdict": "PASS" if passed else "FAIL", "valid": integ["all_ok"]},
        "seconds": time.time() - t0,
    }
    v = clean(v)
    tmp = VERDICT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(v, indent=1, allow_nan=False))
    tmp.replace(VERDICT)
    report(v)
    print(f"\nwrote {VERDICT.relative_to(ROOT)} ({v['seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
