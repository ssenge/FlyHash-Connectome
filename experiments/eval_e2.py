"""Standalone evaluation of E2 (graded-query readout of stored winner sets)
against its pre-registered pass criteria.

Pre-registration: experiments/PREREGISTRATION_2026-09-28.md, section E2 (also
UPDATES.md section 14), frozen 2026-09-28 before any run. Input:
results/e2_readout.json, written by `python -m flypath.readout`. Output:
results/e2_verdict.json and a printed report.

    PYTHONPATH=. /opt/homebrew/anaconda3/envs/flybrain/bin/python experiments/eval_e2.py

The script uses numpy only and does not import flypath. Every verdict is
recomputed from the raw per-trial AP values in the results file. The runner's
stored `analysis` block is used only as a cross-check at the end. Any
disagreement is printed and stored, and it never overrides the recomputed
verdict.

Criteria, copied from the pre-registration
------------------------------------------
Protocol: 10,000 items, 1,000 test queries, truth = top 2% (200) by Euclidean
distance on raw features, AP@200 and recall@200, 50 trials. Datasets SIFT,
GloVe, MNIST, odour mixtures. k in {2, 4, 8, 16, 32}. Random fly matrices
m = 10d, s = round(0.1d), row-centred input. Storage, primary: B(k) =
ceil(log2 C(m, k)) bits. Training-free competitors at B(k) bits, each with
symmetric and asymmetric scoring: Gaussian sign code, DenseFly, FlyLSH value
tag (k' = floor(B / (ceil(log2 m) + 4)) winners, 4-bit values), symmetric fly
overlap. PQ-ADC at B bits is a trained reference, reported and not scored.

Pass (primary, both required):
  (a) GloVe: the ratio AP(fly asymmetric) / AP(best training-free
      competitor) has a trial-bootstrap 95% lower bound >= 1.0 for >= 4 of
      the 5 values of k.
  (b) MNIST: the same at k = 2 and k = 4.
Secondary (paper robustness):
  (c) MaleCNS R, k in {4, 16}: the connectome-null and even-fan-out-null gaps
      under the asymmetric readout lie inside the symmetric readout's 95%
      interval on >= 3 of 4 datasets.

Details fixed in the runner before the full run
-----------------------------------------------
The runner (flypath/readout.py, "Unspecified details fixed before running")
fixed these before any full-scale result existed. This script applies them as
written there:
  - "fly asymmetric" is fly_asym_raw or fly_asym_std, chosen per dataset on
    the validation split: "std" iff its validation AP, averaged over the 10
    validation trials and all five k, is strictly higher than "raw"
    (runner item 3). This script recomputes that choice from the stored
    validation trials and requires it to equal the stored choice.
  - Competitor set of the criterion ("primary", runner items 5 and 11), all at
    B(k) (info accounting): fly_sym; for gauss, densefly and vtag the
    symmetric scorer and the asymmetric scorer with its own validated query
    variant; vtag's ADC form.
  - Estimand (runner item 11): mean over trials of fly-asymmetric AP divided
    by the largest mean-over-trials AP among the competitors. 95% percentile
    bootstrap over trials, with the maximum over competitors re-taken in every
    resample. "Lower bound" = the 2.5th percentile.
  - (c) (runner item 12): gap = 100 * (mean over the 20 trials of control AP
    / mean over trials of null-mean AP - 1), for control = connectome and
    control = even fan-out, under each readout. The asymmetric readout is the
    dataset's validated fly choice. "Lies inside" = the asymmetric point
    estimate lies within the symmetric readout's 95% bootstrap interval
    (closed interval). A dataset passes when all four of its gaps (two
    controls x k = 4, 16) lie inside. (c) passes on >= 3 of 4 datasets.

Unspecified details fixed before running
----------------------------------------
Fixed here. The script was written after the full run had finished. By then
the runner's stored `criteria` summary had been seen (a: 0 of 5 k passing;
b: pass; c: 1 of 4 datasets), but not the per-row ratios or intervals. So the
choices below follow the evaluation instruction ("2,000 resamples, seed 0")
and the runner's pre-run choices. None is picked to move a verdict, and every
place where the stored analysis differs is reported.

 1. Bootstrap. 2,000 resamples of the trial indices with replacement. The
    indices are drawn once per trial count as the matrix
    numpy.random.default_rng(0).integers(0, n, size=(2000, n)): n = 50 for
    the E2 tables, n = 20 for (c). The matrix is shared by every row with
    that trial count. Resampling is paired: the fly and every competitor
    (for (c): control and null mean) use the same resampled trials, because
    they share the trial's queries and truth. Two-sided 95% percentile
    interval: np.percentile at 2.5 and 97.5, numpy's default linear
    interpolation, the runner's convention. This matches the runner in all
    respects except the seed. The runner seeded each E2 row from
    SeedSequence([50000, dataset index, accounting index, k index]). It
    seeded each (c) row with seed = index of k within (4, 16), so seeds 0
    and 1. Runner item 12 says this matches the paper's seeding, but it does
    not. The paper (`replication.controls`) used seed = index of k within
    the paper's sizes (2, 4, 8, 16, 32, 94), so seed 1 for k = 4 and seed 3
    for k = 16. For (c) this script therefore computes three versions of the
    symmetric interval: seed 0 (the verdict); the runner's seeds, which must
    reproduce the stored analysis exactly; and the paper's seeds, which must
    reproduce the intervals stored in results/controls.json and so are
    literally "the paper's interval". Any verdict that depends on the seed
    is flagged as seed-sensitive, and the seed-0 verdict stands.
 2. Thresholds are applied to the unrounded numbers. A lower bound of
    0.99999 fails (a) and (b). A near miss is a fail.
 3. Missing values. A criterion row whose fly or competitor AP is undefined
    or non-finite in any trial fails, and it is listed under integrity
    failures. The pre-registered competitors are always defined at the
    protocol's budgets (k' >= 1), and this is checked.
 4. Integrity checks. Every failure is listed in the verdict file, and the
    verdict is still computed. The checks: the run status is complete; the
    stored settings equal the registered design; the stored
    preregistration SHA-256 equals the SHA-256 of the current file; the
    stored runner SHA-256 equals flypath/readout.py as it is now; 50 test
    trials, 10 validation trials and 20 connectome trials per dataset with
    contiguous keys; m = 10d and s = round(0.1d); B(k) recomputed exactly
    with math.comb; fixed-width bits and k' recomputed; k' >= 1; the
    validation choice recomputed; each stored null mean equal to the mean of
    its 10 stored nulls; the symmetric readout's per-trial connectome, null
    mean and even-fan-out AP identical to the paper's values in
    results/controls.json (hemispheres.malecns_R), since runner item 12 says
    it reproduces them exactly.
 5. Reported only, entering no verdict: the same ratio tables under the
    fixed-width accounting; the "raw_asym" and "all_variants" competitor
    sets (runner item 5); the fly's other query variant; the ratio to each
    competitor alone; the PQ-ADC reference ratio; mean AP and recall@200
    per method; and the paper's stored (c) symmetric intervals.

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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "e2_readout.json"
VERDICT = ROOT / "results" / "e2_verdict.json"
CONTROLS = ROOT / "results" / "controls.json"
PREREG_FILE = ROOT / "experiments" / "PREREGISTRATION_2026-09-28.md"
RUNNER_FILE = ROOT / "flypath" / "readout.py"

N_BOOT = 2000
BOOT_SEED = 0
Q_LO, Q_HI = 2.5, 97.5

# Registered design (PREREGISTRATION_2026-09-28.md, E2) and the thresholds
# used for the verdicts.
PREREG = {
    "datasets": ["sift", "glove", "mnist", "odours"],
    "trials": 50, "n_items": 10_000, "n_queries": 1_000, "top_fraction": 0.02,
    "ks": [2, 4, 8, 16, 32], "expansion": 10, "sampling": 0.1, "value_bits": 4,
    "val_items": 2_000, "val_queries": 200,
    "a": {"dataset": "glove", "accounting": "info", "threshold": 1.0, "k_needed": 4, "n_k": 5},
    "b": {"dataset": "mnist", "accounting": "info", "threshold": 1.0, "ks": [2, 4]},
    "c": {"ks": [4, 16], "datasets_needed": 3, "n_datasets": 4},
}
# Runner constants fixed before the run (flypath/readout.py).
RUNNER = {"val_trials": 10, "conn_trials": 20, "conn_nulls": 10, "boot_seed_base": 50_000,
          "dataset_order": ["sift", "glove", "mnist", "odours"],
          "accounting_index": {"info": 1, "fixed": 2}}
VARIANTS = ("raw", "std")
FAMILIES = ("gauss", "densefly", "vtag")
TABLES = ("primary", "raw_asym", "all_variants")


# ---------------------------------------------------------------- helpers

def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ceil_log2(n: int) -> int:
    """ceil(log2 n) for an integer n >= 1, exact."""
    return (int(n) - 1).bit_length()


def boot_index(n: int, seed: int = BOOT_SEED, draws: int = N_BOOT) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


_IDX_CACHE: dict[int, np.ndarray] = {}


def shared_index(n: int) -> np.ndarray:
    """The seed-0 resampling matrix for n trials, drawn once and shared."""
    if n not in _IDX_CACHE:
        _IDX_CACHE[n] = boot_index(n)
    return _IDX_CACHE[n]


def ratio_boot(fly: np.ndarray, comps: np.ndarray, idx: np.ndarray) -> dict:
    """mean(fly) / max_c mean(comp_c) over trials, with a percentile
    bootstrap over trials in which the maximum is re-taken."""
    fly = np.asarray(fly, float)
    c = np.atleast_2d(np.asarray(comps, float))
    means = c.mean(1)
    boot = fly[idx].mean(1) / c[:, idx].mean(2).max(0)
    lo, hi = np.percentile(boot, [Q_LO, Q_HI])
    return {"ratio": float(fly.mean() / means.max()), "ci95": [float(lo), float(hi)],
            "best": int(np.argmax(means))}


def gap_boot(ctrl: np.ndarray, null: np.ndarray, idx: np.ndarray) -> dict:
    """100 * (mean ctrl / mean null - 1) with a paired percentile bootstrap."""
    ctrl, null = np.asarray(ctrl, float), np.asarray(null, float)
    boot = 100 * (ctrl[idx].mean(1) / null[idx].mean(1) - 1)
    lo, hi = np.percentile(boot, [Q_LO, Q_HI])
    return {"estimate": float(100 * (ctrl.mean() / null.mean() - 1)), "ci95": [float(lo), float(hi)]}


def ap_matrix(trials: dict, method: str, what: str = "ap") -> np.ndarray:
    keys = sorted(trials, key=int)
    return np.array([[np.nan if v is None else v for v in trials[t][what][method]] for t in keys], float)


def competitor_set(acc: str, table: str, choice: dict) -> list[str]:
    """Competitor sets as fixed in the runner (items 5 and 11)."""
    out = ["fly_sym"]
    for f in FAMILIES:
        out.append(f"{f}_{acc}_sym")
        if table == "primary":
            out.append(f"{f}_{acc}_asym_{choice[f'{f}_{acc}']}")
        elif table == "raw_asym":
            out.append(f"{f}_{acc}_asym_raw")
        elif table == "all_variants":
            out += [f"{f}_{acc}_asym_{v}" for v in VARIANTS]
    if table in ("primary", "all_variants"):
        out.append(f"vtag_{acc}_adc")
    return out


def runner_row_seed(ds_name: str, acc: str, i: int) -> int:
    j = RUNNER["dataset_order"].index(ds_name)
    return int(np.random.SeedSequence([RUNNER["boot_seed_base"], j, RUNNER["accounting_index"][acc], i])
               .generate_state(1)[0])


# ---------------------------------------------------------------- integrity

def integrity(res: dict) -> tuple[list[str], dict]:
    fails: list[str] = []
    info: dict = {}

    def check(ok: bool, msg: str):
        if not ok:
            fails.append(msg)

    check(res.get("status") == "complete", f"status is {res.get('status')!r}, not 'complete'")
    st = res["settings"]
    for key in ("datasets", "trials", "n_items", "n_queries", "top_fraction", "ks", "expansion",
                "sampling", "value_bits", "val_items", "val_queries"):
        check(st.get(key) == PREREG[key], f"settings.{key} = {st.get(key)!r}, registered {PREREG[key]!r}")
    check(st.get("conn_ks") == PREREG["c"]["ks"], f"settings.conn_ks = {st.get('conn_ks')}")
    check(st.get("conn_datasets") == PREREG["datasets"], f"settings.conn_datasets = {st.get('conn_datasets')}")
    for key in ("val_trials", "conn_trials", "conn_nulls"):
        check(st.get(key) == RUNNER[key], f"settings.{key} = {st.get(key)!r}, runner fixed {RUNNER[key]!r}")
    pre_now, run_now = sha256(PREREG_FILE), sha256(RUNNER_FILE)
    info["preregistration_sha256"] = {"stored": res.get("preregistration_sha256"), "current": pre_now}
    info["runner_sha256"] = {"stored": res.get("code_sha256"), "current": run_now}
    check(res.get("preregistration_sha256") == pre_now, "pre-registration file changed since the run")
    check(res.get("code_sha256") == run_now, "flypath/readout.py changed since the run")

    ks = PREREG["ks"]
    recomputed_choice = {}
    for name in PREREG["datasets"]:
        ds = res["datasets"].get(name)
        if ds is None:
            fails.append(f"{name}: dataset missing")
            continue
        d, m = ds["d"], ds["m"]
        check(m == PREREG["expansion"] * d, f"{name}: m = {m}, expected 10d = {10 * d}")
        check(ds["s"] == max(1, int(round(PREREG["sampling"] * d))), f"{name}: s = {ds['s']}")
        check(ds.get("n_items") == PREREG["n_items"], f"{name}: n_items = {ds.get('n_items')}")
        check(ds.get("top") == int(round(PREREG["top_fraction"] * PREREG["n_items"])), f"{name}: top = {ds.get('top')}")
        info_bits = [ceil_log2(math.comb(m, k)) for k in ks]
        fixed_bits = [k * ceil_log2(m) for k in ks]
        check(ds["bits"]["info"] == info_bits, f"{name}: info bits {ds['bits']['info']} != {info_bits}")
        check(ds["bits"]["fixed"] == fixed_bits, f"{name}: fixed bits {ds['bits']['fixed']} != {fixed_bits}")
        for acc, bits in (("info", info_bits), ("fixed", fixed_bits)):
            kp = [b // (ceil_log2(m) + PREREG["value_bits"]) for b in bits]
            check(ds["value_tag_winners"][acc] == kp, f"{name}: value-tag k' ({acc}) {ds['value_tag_winners'][acc]} != {kp}")
            check(min(kp) >= 1, f"{name}: value tag undefined (k' = 0) under {acc}")
        keys = sorted(ds["trials"], key=int)
        check(keys == [str(t) for t in range(PREREG["trials"])], f"{name}: test trial keys {keys[:3]}... n={len(keys)}")
        for t in keys:
            for what in ("ap", "recall"):
                for meth, vals in ds["trials"][t][what].items():
                    if len(vals) != len(ks):
                        fails.append(f"{name} trial {t}: {what}.{meth} has {len(vals)} values")
        val = ds["validation"]
        check(val.get("n_items") == PREREG["val_items"], f"{name}: validation n_items = {val.get('n_items')}")
        check(val.get("n_queries") == PREREG["val_queries"], f"{name}: validation n_queries = {val.get('n_queries')}")
        check(val.get("test_subset_reproduced") is True, f"{name}: validation test subset not reproduced")
        vkeys = sorted(val["trials"], key=int)
        check(vkeys == [str(v) for v in range(RUNNER["val_trials"])], f"{name}: validation trial keys {vkeys}")
        ch = {}
        for fam in ["fly", *(f"{f}_{a}" for f in FAMILIES for a in ("info", "fixed"))]:
            pre = "fly_asym" if fam == "fly" else f"{fam}_asym"
            mm = {v: float(np.nanmean(ap_matrix(val["trials"], f"{pre}_{v}"))) for v in VARIANTS}
            ch[fam] = {"choice": "std" if mm["std"] > mm["raw"] else "raw", "mean_ap": mm}
            check(ch[fam]["choice"] == val["choice"][fam],
                  f"{name}: validation choice for {fam} recomputes as {ch[fam]['choice']}, stored {val['choice'][fam]}")
        recomputed_choice[name] = ch

    info["validation_choice_recomputed"] = recomputed_choice
    paper = json.loads(CONTROLS.read_text())["hemispheres"]["malecns_R"] if CONTROLS.exists() else {}
    pchk = {}
    for name in PREREG["datasets"]:
        cn = res["connectome"].get(name)
        if cn is None:
            fails.append(f"connectome {name}: missing")
            continue
        keys = sorted(cn["trials"], key=int)
        check(keys == [str(t) for t in range(RUNNER["conn_trials"])], f"connectome {name}: trial keys n={len(keys)}")
        check(cn["ks"] == PREREG["c"]["ks"], f"connectome {name}: ks {cn['ks']}")
        check(cn.get("n_nulls") == RUNNER["conn_nulls"], f"connectome {name}: n_nulls {cn.get('n_nulls')}")
        worst = 0.0
        for t in keys:
            tr = cn["trials"][t]
            for r in ("sym", "asym_raw", "asym_std"):
                nulls = np.array(tr["null"][r], float)
                if nulls.shape != (RUNNER["conn_nulls"], len(cn["ks"])):
                    fails.append(f"connectome {name} trial {t}: null.{r} shape {nulls.shape}")
                    continue
                worst = max(worst, float(np.abs(nulls.mean(0) - np.array(tr["null_mean"][r])).max()))
        check(worst < 1e-12, f"connectome {name}: stored null mean differs from the mean of its nulls by {worst}")
        pp = paper.get(name)
        if not pp:
            pchk[name] = {"available": False}
            fails.append(f"connectome {name}: paper values not found in results/controls.json")
            continue
        cols = [pp["sizes"].index(k) for k in cn["ks"]]
        diff = {}
        for kind, pk in (("real", "real"), ("null_mean", "null"), ("out_equal", "out_equal")):
            ours = np.array([cn["trials"][t][kind]["sym"] for t in keys], float)
            theirs = np.array(pp[pk], float)[:len(keys), cols]
            diff[kind] = float(np.abs(ours - theirs).max())
        same = (pp["trials"] == len(keys) and pp["B"] == cn.get("n_nulls") and max(diff.values()) < 1e-12)
        pchk[name] = {"available": True, "max_abs_diff": diff, "identical": bool(same)}
        check(same, f"connectome {name}: symmetric readout does not reproduce the paper (max diff {diff})")
    info["paper_check"] = pchk
    return fails, info


# ---------------------------------------------------------------- E2 tables

def ratio_table(res: dict, name: str, acc: str, table: str, fly_variant: str | None = None) -> list[dict]:
    ds = res["datasets"][name]
    ks = res["settings"]["ks"]
    choice = ds["validation"]["choice"]
    fly_name = f"fly_asym_{fly_variant or choice['fly']}"
    comps = competitor_set(acc, table, choice)
    fly = ap_matrix(ds["trials"], fly_name)
    mats = {c: ap_matrix(ds["trials"], c) for c in comps}
    idx = shared_index(fly.shape[0])
    rows = []
    for i, k in enumerate(ks):
        cols = np.array([mats[c][:, i] for c in comps])
        defined = bool(np.isfinite(fly[:, i]).all() and np.isfinite(cols).all())
        row = {"k": int(k), "bits": ds["bits"][acc][i], "fly": fly_name, "competitors": comps,
               "all_defined": defined, "fly_mean_ap": float(np.mean(fly[:, i])),
               "competitor_mean_ap": {c: float(np.mean(mats[c][:, i])) for c in comps}}
        if defined:
            r = ratio_boot(fly[:, i], cols, idx)
            rr = ratio_boot(fly[:, i], cols, boot_index(fly.shape[0], runner_row_seed(name, acc, i)))
            row.update({"ratio": r["ratio"], "ci95": r["ci95"], "lower": r["ci95"][0],
                        "lower_ge_1": bool(r["ci95"][0] >= PREREG["a"]["threshold"]),
                        "best_competitor": comps[r["best"]],
                        "runner_seed_ci95": rr["ci95"],
                        "runner_seed_lower_ge_1": bool(rr["ci95"][0] >= PREREG["a"]["threshold"]),
                        "per_competitor": {c: {k2: v for k2, v in ratio_boot(fly[:, i], mats[c][:, i], idx).items()
                                               if k2 != "best"} for c in comps}})
        else:
            row.update({"ratio": None, "ci95": None, "lower": None, "lower_ge_1": False,
                        "best_competitor": None})
        rows.append(row)
    return rows


def pq_table(res: dict, name: str, acc: str) -> list[dict]:
    ds = res["datasets"][name]
    ks = res["settings"]["ks"]
    fly_name = f"fly_asym_{ds['validation']['choice']['fly']}"
    fly = ap_matrix(ds["trials"], fly_name)
    pq = ap_matrix(ds["trials"], f"pq_{acc}_adc")
    idx = shared_index(fly.shape[0])
    out = []
    for i, k in enumerate(ks):
        if np.isfinite(pq[:, i]).all():
            r = ratio_boot(fly[:, i], pq[:, i], idx)
            out.append({"k": int(k), "pq_mean_ap": float(pq[:, i].mean()), "fly_over_pq": r["ratio"],
                        "ci95": r["ci95"]})
        else:
            out.append({"k": int(k), "pq_mean_ap": None, "fly_over_pq": None, "ci95": None})
    return out


# ---------------------------------------------------------------- criterion (c)

def connectome_eval(res: dict, name: str, paper: dict | None) -> dict:
    cn = res["connectome"][name]
    choice = res["datasets"][name]["validation"]["choice"]["fly"]
    other = "std" if choice == "raw" else "raw"
    keys = sorted(cn["trials"], key=int)
    idx = shared_index(len(keys))

    def arr(kind, readout):
        return np.array([cn["trials"][t][kind][readout] for t in keys], float)

    rows = []
    for i, k in enumerate(cn["ks"]):
        idx_runner = boot_index(len(keys), seed=i)                      # readout.analyse_connectome
        paper_seed = paper["sizes"].index(k) if paper and k in paper.get("sizes", []) else None
        idx_paper = boot_index(len(keys), seed=paper_seed) if paper_seed is not None else None
        for gap, label in (("real", "connectome-null"), ("out_equal", "even_fanout-null")):
            null_sym = arr("null_mean", "sym")[:, i]
            sym = gap_boot(arr(gap, "sym")[:, i], null_sym, idx)
            asy = gap_boot(arr(gap, f"asym_{choice}")[:, i], arr("null_mean", f"asym_{choice}")[:, i], idx)
            asy_o = gap_boot(arr(gap, f"asym_{other}")[:, i], arr("null_mean", f"asym_{other}")[:, i], idx)
            sym_r = gap_boot(arr(gap, "sym")[:, i], null_sym, idx_runner)
            inside = bool(sym["ci95"][0] <= asy["estimate"] <= sym["ci95"][1])
            inside_r = bool(sym_r["ci95"][0] <= asy["estimate"] <= sym_r["ci95"][1])
            row = {"k": int(k), "gap": label, "symmetric": sym, "asymmetric_readout": f"asym_{choice}",
                   "asymmetric": asy, "inside": inside,
                   "asymmetric_other_readout": f"asym_{other}", "asymmetric_other": asy_o,
                   "other_inside": bool(sym["ci95"][0] <= asy_o["estimate"] <= sym["ci95"][1]),
                   "runner_seed": i, "runner_seed_symmetric_ci95": sym_r["ci95"], "runner_seed_inside": inside_r}
            if idx_paper is not None:
                sym_p = gap_boot(arr(gap, "sym")[:, i], null_sym, idx_paper)
                stored = paper["contrasts"][gap][paper_seed]
                row.update({"paper_seed": paper_seed, "paper_seed_symmetric_ci95": sym_p["ci95"],
                            "paper_seed_inside": bool(sym_p["ci95"][0] <= asy["estimate"] <= sym_p["ci95"][1]),
                            "paper_stored_symmetric": {"estimate": stored["estimate"], "ci95": stored["ci95"]},
                            "paper_stored_reproduced": bool(
                                abs(stored["estimate"] - sym_p["estimate"]) < 1e-9
                                and max(abs(a - b) for a, b in zip(stored["ci95"], sym_p["ci95"])) < 1e-9)})
            rows.append(row)
    return {"n_trials": len(keys), "validated_fly_readout": f"asym_{choice}", "rows": rows,
            "dataset_passes": bool(all(r["inside"] for r in rows)),
            "runner_seed_dataset_passes": bool(all(r["runner_seed_inside"] for r in rows)),
            "paper_seed_dataset_passes": (bool(all(r["paper_seed_inside"] for r in rows))
                                          if all("paper_seed_inside" in r for r in rows) else None),
            "mean_ap": {kind: {r: arr(kind, r).mean(0).tolist() for r in ("sym", "asym_raw", "asym_std")}
                        for kind in ("real", "null_mean", "out_equal")}}


# ---------------------------------------------------------------- main

def fmt_ci(ci) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def main() -> int:
    res = json.loads(RESULTS.read_text())
    fails, info = integrity(res)
    ks = res["settings"]["ks"]
    out: dict = {
        "experiment": "E2: graded-query readout of stored winner sets",
        "evaluated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": str(RESULTS.relative_to(ROOT)),
        "evaluator_sha256": sha256(Path(__file__)),
        "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "interval": "percentile 2.5/97.5",
                      "unit": "trial", "paired": True,
                      "index": "numpy.random.default_rng(0).integers(0, n, (2000, n)), one per trial count"},
        "integrity": {"failures": fails, "ok": not fails, **info},
    }

    # ---- reported tables (every dataset, both accountings)
    tables: dict = {}
    for name in res["settings"]["datasets"]:
        ds = res["datasets"][name]
        choice = ds["validation"]["choice"]
        t = {"fly_readout": f"fly_asym_{choice['fly']}", "validation_choice": choice}
        for acc in ("info", "fixed"):
            t[acc] = {tb: ratio_table(res, name, acc, tb) for tb in TABLES}
            other = "std" if choice["fly"] == "raw" else "raw"
            t[acc]["primary_other_fly_variant"] = ratio_table(res, name, acc, "primary", fly_variant=other)
            t[acc]["pq_reference"] = pq_table(res, name, acc)
        t["mean_ap"] = {m: np.nanmean(ap_matrix(ds["trials"], m), 0).tolist() for m in ds["trials"]["0"]["ap"]}
        t["mean_recall"] = {m: np.nanmean(ap_matrix(ds["trials"], m, "recall"), 0).tolist()
                            for m in ds["trials"]["0"]["recall"]}
        tables[name] = t
    out["tables"] = tables

    # ---- primary criteria
    ra = tables[PREREG["a"]["dataset"]][PREREG["a"]["accounting"]]["primary"]
    n_a = sum(r["lower_ge_1"] for r in ra)
    crit_a = {"dataset": "glove", "accounting": "info (B(k) = ceil(log2 C(m,k)))",
              "rule": "trial-bootstrap 95% lower bound of AP(fly asym)/AP(best competitor) >= 1.0 for >= 4 of 5 k",
              "rows": [{"k": r["k"], "bits": r["bits"], "ratio": r["ratio"], "ci95": r["ci95"],
                        "lower_ge_1": r["lower_ge_1"], "best_competitor": r["best_competitor"],
                        "all_defined": r["all_defined"]} for r in ra],
              "k_passing": [r["k"] for r in ra if r["lower_ge_1"]], "n_passing": int(n_a),
              "required": PREREG["a"]["k_needed"], "pass": bool(n_a >= PREREG["a"]["k_needed"]),
              "runner_seed_n_passing": int(sum(r.get("runner_seed_lower_ge_1", False) for r in ra))}
    rb = {r["k"]: r for r in tables[PREREG["b"]["dataset"]][PREREG["b"]["accounting"]]["primary"]}
    crit_b = {"dataset": "mnist", "accounting": "info (B(k) = ceil(log2 C(m,k)))",
              "rule": "trial-bootstrap 95% lower bound >= 1.0 at k = 2 and at k = 4",
              "rows": [{"k": k, "bits": rb[k]["bits"], "ratio": rb[k]["ratio"], "ci95": rb[k]["ci95"],
                        "lower_ge_1": rb[k]["lower_ge_1"], "best_competitor": rb[k]["best_competitor"],
                        "all_defined": rb[k]["all_defined"]} for k in PREREG["b"]["ks"]],
              "pass": bool(all(rb[k]["lower_ge_1"] for k in PREREG["b"]["ks"])),
              "runner_seed_pass": bool(all(rb[k].get("runner_seed_lower_ge_1", False) for k in PREREG["b"]["ks"]))}
    primary_pass = bool(crit_a["pass"] and crit_b["pass"])

    # ---- secondary criterion (c)
    paper = json.loads(CONTROLS.read_text())["hemispheres"]["malecns_R"] if CONTROLS.exists() else {}
    conn = {name: connectome_eval(res, name, paper.get(name)) for name in PREREG["datasets"]}
    n_c = sum(c["dataset_passes"] for c in conn.values())
    crit_c = {"rule": "all 4 gaps (connectome-null, even_fanout-null; k = 4, 16) of a dataset: asymmetric "
                      "point estimate inside the symmetric 95% interval; pass on >= 3 of 4 datasets",
              "datasets_passing": [n for n, c in conn.items() if c["dataset_passes"]],
              "n_passing": int(n_c), "required": PREREG["c"]["datasets_needed"],
              "pass": bool(n_c >= PREREG["c"]["datasets_needed"]),
              "runner_seed_n_passing": int(sum(c["runner_seed_dataset_passes"] for c in conn.values())),
              "per_dataset": conn}

    seed_sensitive = []
    for r in ra:
        if r.get("runner_seed_lower_ge_1") is not None and r["runner_seed_lower_ge_1"] != r["lower_ge_1"]:
            seed_sensitive.append(f"(a) glove k={r['k']}")
    for k in PREREG["b"]["ks"]:
        if rb[k].get("runner_seed_lower_ge_1") is not None and rb[k]["runner_seed_lower_ge_1"] != rb[k]["lower_ge_1"]:
            seed_sensitive.append(f"(b) mnist k={k}")
    for n, c in conn.items():
        for r in c["rows"]:
            if r["runner_seed_inside"] != r["inside"]:
                seed_sensitive.append(f"(c) {n} k={r['k']} {r['gap']} (runner seed {r['runner_seed']})")
            if "paper_seed_inside" in r and r["paper_seed_inside"] != r["inside"]:
                seed_sensitive.append(f"(c) {n} k={r['k']} {r['gap']} (paper seed {r['paper_seed']})")
    crit_c["paper_seed_n_passing"] = (int(sum(c["paper_seed_dataset_passes"] for c in conn.values()))
                                      if all(c["paper_seed_dataset_passes"] is not None for c in conn.values())
                                      else None)
    crit_c["paper_stored_intervals_reproduced"] = bool(all(r.get("paper_stored_reproduced", False)
                                                           for c in conn.values() for r in c["rows"]))

    out["criteria"] = {"a_glove": crit_a, "b_mnist": crit_b, "primary_pass": primary_pass,
                       "c_connectome": crit_c,
                       "all_competitors_present": bool(all(r["all_defined"] for r in ra)
                                                       and all(rb[k]["all_defined"] for k in PREREG["b"]["ks"])),
                       "seed_sensitive_verdicts": seed_sensitive}

    # ---- cross-check against the runner's stored analysis
    xc: dict = {"differences": []}
    st = res.get("analysis", {})
    if st:
        worst_ratio, worst_ci = 0.0, 0.0
        for name, t in tables.items():
            sd = st["datasets"][name]["comparisons"]
            for acc in ("info", "fixed"):
                for tb in TABLES:
                    for mine, theirs in zip(t[acc][tb], sd[acc][tb]):
                        if mine["ratio"] is None:
                            continue
                        worst_ratio = max(worst_ratio, abs(mine["ratio"] - theirs["ratio"]))
                        worst_ci = max(worst_ci, max(abs(a - b) for a, b in zip(mine["runner_seed_ci95"], theirs["ci95"])))
                        if mine["best_competitor"] != theirs["best_competitor"]:
                            xc["differences"].append(f"{name} {acc} {tb} k={mine['k']}: best competitor "
                                                     f"{mine['best_competitor']} vs stored {theirs['best_competitor']}")
                        if mine["lower_ge_1"] != theirs["lower_ge_1"]:
                            xc["differences"].append(f"{name} {acc} {tb} k={mine['k']}: lower>=1 {mine['lower_ge_1']} "
                                                     f"(seed 0) vs stored {theirs['lower_ge_1']}")
        xc["max_abs_ratio_diff"] = worst_ratio
        xc["max_abs_ci_diff_runner_seed"] = worst_ci
        worst_c = 0.0
        for name, c in conn.items():
            for mine, theirs in zip(c["rows"], st["connectome"][name]["rows"]):
                worst_c = max(worst_c, abs(mine["symmetric"]["estimate"] - theirs["symmetric"]["estimate"]),
                              abs(mine["asymmetric"]["estimate"] - theirs["asymmetric"]["estimate"]),
                              *(abs(a - b) for a, b in zip(mine["runner_seed_symmetric_ci95"], theirs["symmetric"]["ci95"])))
                if mine["inside"] != theirs["inside"]:
                    xc["differences"].append(f"(c) {name} k={mine['k']} {mine['gap']}: inside {mine['inside']} "
                                             f"(seed 0) vs stored {theirs['inside']}")
        xc["max_abs_c_diff_runner_seed"] = worst_c
        sc = st.get("criteria", {})
        for key, mine in (("a_glove", crit_a["pass"]), ("b_mnist", crit_b["pass"]),
                          ("primary_pass", primary_pass), ("c_connectome", crit_c["pass"])):
            theirs = sc.get(key)
            theirs = theirs.get("pass") if isinstance(theirs, dict) else theirs
            if theirs != mine:
                xc["differences"].append(f"criterion {key}: recomputed {mine} vs stored {theirs}")
        xc["stored_criteria"] = sc
        xc["implementation_reproduces_stored_intervals"] = bool(worst_ratio < 1e-12 and worst_ci < 1e-12
                                                                and worst_c < 1e-9)
    out["cross_check"] = xc

    VERDICT.write_text(json.dumps(out, indent=1))

    # ---- report
    p = print
    p("E2 evaluation (graded-query readout), pre-registration section E2")
    p(f"input {RESULTS.relative_to(ROOT)}; bootstrap {N_BOOT} resamples over trials, seed {BOOT_SEED}, paired")
    p(f"integrity: {'OK' if not fails else f'{len(fails)} FAILURE(S)'}")
    for f in fails:
        p(f"  - {f}")
    p("")
    for key, crit in (("(a) GloVe", crit_a), ("(b) MNIST", crit_b)):
        p(f"{key}, info accounting, fly readout {tables[crit['dataset']]['fly_readout']}")
        p(f"  {'k':>3} {'B':>4} {'ratio':>8} {'95% CI':>20}  lower>=1  best competitor")
        for r in crit["rows"]:
            p(f"  {r['k']:>3} {r['bits']:>4} {r['ratio']:>8.4f} {fmt_ci(r['ci95']):>20}  "
              f"{'yes' if r['lower_ge_1'] else 'no':>8}  {r['best_competitor']}")
    p(f"  (a) {crit_a['n_passing']} of 5 k pass (need >= 4): {'PASS' if crit_a['pass'] else 'FAIL'}")
    p(f"  (b) k=2 and k=4 both required: {'PASS' if crit_b['pass'] else 'FAIL'}")
    p(f"PRIMARY (a AND b): {'PASS' if primary_pass else 'FAIL'}")
    p("")
    p("(c) MaleCNS R, gaps in % (asymmetric point estimate vs symmetric 95% CI)")
    for name, c in conn.items():
        for r in c["rows"]:
            p(f"  {name:>6} k={r['k']:>2} {r['gap']:<17} sym {r['symmetric']['estimate']:+7.2f} "
              f"{fmt_ci(r['symmetric']['ci95']):>20}  {r['asymmetric_readout']} {r['asymmetric']['estimate']:+7.2f}  "
              f"{'inside' if r['inside'] else 'OUTSIDE'}")
        p(f"  {name:>6}: {'passes' if c['dataset_passes'] else 'fails'}")
    p(f"  (c) {crit_c['n_passing']} of 4 datasets pass (need >= 3): {'PASS' if crit_c['pass'] else 'FAIL'}")
    p("")
    p("Reported, not scored: fly asym / best competitor under the fixed-width accounting, and fly / PQ-ADC (info)")
    for name, t in tables.items():
        fx = t["fixed"]["primary"]
        pq = t["info"]["pq_reference"]
        p(f"  {name:>6} fixed: " + "  ".join(f"k={r['k']}: {r['ratio']:.3f} {fmt_ci(r['ci95'])}" for r in fx))
        p(f"  {name:>6} PQ   : " + "  ".join(f"k={r['k']}: {r['fly_over_pq']:.3f}" for r in pq))
    p("")
    p(f"seed-sensitive verdicts (seed 0 vs runner seeds): {seed_sensitive or 'none'}")
    p(f"cross-check vs stored analysis: max |ratio diff| {xc.get('max_abs_ratio_diff')}, "
      f"max |CI diff| (runner seeds) {xc.get('max_abs_ci_diff_runner_seed')}, "
      f"(c) max diff {xc.get('max_abs_c_diff_runner_seed')}")
    for dff in xc["differences"]:
        p(f"  - {dff}")
    p(f"verdict written to {VERDICT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
