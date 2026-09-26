"""Glomerulus -> Kenyon cell projections from four connectomes.

  malecns    MaleCNS v1.0 (Berg et al. 2026), male, brain and nerve cord, FIB-SEM
  hemibrain  hemibrain v1.2 (Scheffer et al. 2020), female, right central brain, FIB-SEM
  flywire    FlyWire FAFB v783 (Dorkenwald et al. 2024; annotations Schlegel et al.
             2024), female, whole brain, serial-section TEM
  banc       BANC v888 (Bates et al. 2026), female, brain and nerve cord, serial-section TEM

Every projection is built by the same rule: uniglomerular olfactory projection
neurons are assigned to glomeruli by cell type (the part before the first
underscore, e.g. DA1_lPN -> DA1; multiglomerular and thermo-/hygrosensory types
excluded, see `flyhash.glomerulus_of`), Kenyon cells are the neurons annotated as
such on the same side, every connection counts (no synapse threshold), synapses
are summed per glomerulus, and cells without olfactory input are dropped. The
four datasets differ in imaging, synapse detection and proofreading, so part of
any difference between them is technical.
"""

from __future__ import annotations

import re
import tarfile

import numpy as np
import pandas as pd

from . import flyhash as fh
from .config import Config

SOURCES = {
    "hemibrain/exported-traced-adjacencies-v1.2.tar.gz":
        "https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz",
    "flywire/Supplemental_file1_neuron_annotations.tsv":
        "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
        "8587524c1748ce5ef2080822a2fc890fc03bf597/supplemental_files/"
        "Supplemental_file1_neuron_annotations.tsv",
    "flywire/proofread_connections_783.feather":
        "https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content",
    "banc/banc_888_meta.feather": "https://dataverse.harvard.edu/api/access/datafile/14033740",
    "banc/banc_888_edgelist_simple_v3.feather":
        "https://dataverse.harvard.edu/api/access/datafile/13918810",
}

# (dataset, side) pairs with a complete mushroom body calyx
HEMISPHERES = [("malecns", "R"), ("malecns", "L"), ("hemibrain", "R"),
               ("flywire", "R"), ("flywire", "L"), ("banc", "R"), ("banc", "L")]
LABELS = {"malecns": "MaleCNS", "hemibrain": "Hemibrain", "flywire": "FlyWire",
          "banc": "BANC"}
SEX = {"malecns": "male", "hemibrain": "female", "flywire": "female", "banc": "female"}

_PN_TYPE = re.compile(r"^[A-Z][A-Za-z0-9+]*_[a-z0-9]*PN")

# hemibrain v1.2 glomerulus names -> current nomenclature, from the
# hemibrain_type matches in the FlyWire annotations (Schlegel et al. 2024)
HEMIBRAIN_RENAME = {"VC3l": "VC3", "VC3m": "VC5", "VC5": "VM6"}
# "Z" projection neurons are not assigned to a canonical glomerulus
EXCLUDE = {"Z"}


def _fetch(cfg: Config, name: str) -> None:
    d = cfg.raw_dir / "connectomes"
    for rel, url in SOURCES.items():
        if rel.startswith(name + "/"):
            fh._fetch(url, d / rel)


def build(pn: pd.DataFrame, kc_ids, edges: pd.DataFrame, name: str, side: str) -> fh.Projection:
    """pn: columns id, glom; edges: columns pre, post, weight (any multiplicity)."""
    pn = pn[pn["glom"].notna() & ~pn["glom"].str.match(fh._NON_OLFACTORY) & ~pn["glom"].isin(EXCLUDE)]
    gloms = sorted(pn["glom"].unique())
    gi = {g: i for i, g in enumerate(gloms)}
    kc_ids = np.asarray(sorted(set(kc_ids)))
    ki = pd.Series(np.arange(len(kc_ids)), index=kc_ids)
    pg = pn.set_index("id")["glom"]
    e = edges[edges["pre"].isin(pg.index) & edges["post"].isin(ki.index)]
    e = e.groupby(["pre", "post"], as_index=False)["weight"].sum()
    rows = pg.loc[e["pre"]].map(gi).to_numpy()
    cols = ki.loc[e["post"]].to_numpy()
    m = np.zeros((len(gloms), len(kc_ids)))
    np.add.at(m, (rows, cols), e["weight"].to_numpy())
    counts = np.zeros(m.shape, np.int64)                 # distinct PNs per glomerulus and cell
    np.add.at(counts, (rows, cols), 1)
    keep = (m > 0).sum(0) > 0
    return fh.Projection("real", m[:, keep], gloms,
                         meta={"pn_counts": counts[:, keep], "pn_partners": counts[:, keep].sum(0),
                               "side": side, "dataset": name, "n_pn": int(len(pn))})


def hemibrain(cfg: Config, side: str = "R") -> fh.Projection:
    if side != "R":
        raise ValueError("the hemibrain contains only the right mushroom body")
    _fetch(cfg, "hemibrain")
    d = cfg.raw_dir / "connectomes" / "hemibrain"
    src = d / "exported-traced-adjacencies-v1.2"
    if not src.exists():
        with tarfile.open(d / "exported-traced-adjacencies-v1.2.tar.gz") as t:
            t.extractall(d, filter="data")
    n = pd.read_csv(src / "traced-neurons.csv")
    t = n["type"].astype(str)
    pn = n[t.str.match(_PN_TYPE) & ~n["instance"].astype(str).str.endswith("_L")]
    glom = pn["type"].map(fh.glomerulus_of)
    pn = pd.DataFrame({"id": pn["bodyId"], "glom": glom.replace(HEMIBRAIN_RENAME)})
    kc = n.loc[t.str.startswith("KC"), "bodyId"]
    e = pd.read_csv(src / "traced-total-connections.csv")
    e.columns = ["pre", "post", "weight"]
    return build(pn, kc, e, "hemibrain", side)


def flywire(cfg: Config, side: str = "R") -> fh.Projection:
    _fetch(cfg, "flywire")
    d = cfg.raw_dir / "connectomes" / "flywire"
    a = pd.read_csv(d / "Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False,
                    usecols=["root_id", "cell_class", "cell_type", "side"])
    s = {"R": "right", "L": "left"}[side]
    a = a[a["side"] == s]
    p = a[a["cell_class"] == "ALPN"]
    pn = pd.DataFrame({"id": p["root_id"], "glom": p["cell_type"].map(fh.glomerulus_of)})
    kc = a.loc[a["cell_class"] == "Kenyon_Cell", "root_id"]
    e = pd.read_feather(d / "proofread_connections_783.feather",
                        columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    e.columns = ["pre", "post", "weight"]
    return build(pn, kc, e, "flywire", side)


def banc(cfg: Config, side: str = "R") -> fh.Projection:
    _fetch(cfg, "banc")
    d = cfg.raw_dir / "connectomes" / "banc"
    meta = pd.read_feather(d / "banc_888_meta.feather",
                           columns=["banc_888_id", "cell_class", "cell_type", "side"])
    for c in ("cell_class", "cell_type", "side"):
        meta[c] = meta[c].astype(object).fillna("").astype(str)
    meta = meta[meta["side"] == {"R": "right", "L": "left"}[side]]
    p = meta[meta["cell_class"] == "antennal_lobe_projection_neuron"]
    pn = pd.DataFrame({"id": p["banc_888_id"], "glom": p["cell_type"].map(fh.glomerulus_of)})
    kc = meta.loc[meta["cell_class"] == "kenyon_cell", "banc_888_id"]
    e = pd.read_feather(d / "banc_888_edgelist_simple_v3.feather", columns=["pre", "post", "count"])
    e.columns = ["pre", "post", "weight"]
    return build(pn, kc, e, "banc", side)


def projection(cfg: Config, dataset: str, side: str) -> fh.Projection:
    if dataset == "malecns":
        from .data import load_graph
        return fh.mushroom_body(load_graph(cfg), side=side)
    return {"hemibrain": hemibrain, "flywire": flywire, "banc": banc}[dataset](cfg, side)


# ---------------------------------------------------------------- comparison

def summary(p: fh.Projection) -> dict:
    """Degree statistics of the binary matrix; `synapses` needs the weighted one."""
    c, f = p.inputs(), p.fan_out()
    return {"n_glomeruli": int(p.matrix.shape[0]), "n_cells": int(p.matrix.shape[1]),
            "n_pn": p.meta.get("n_pn"), "nnz": p.nnz, "synapses": float(p.matrix.sum()),
            "inputs_mean": float(c.mean()), "inputs_median": float(np.median(c)),
            "inputs_hist": np.bincount(c, minlength=16)[:16].tolist(),
            "fan_out_cv": float(f.std() / f.mean()),
            "pn_partners_mean": float(np.mean(p.meta["pn_partners"]))}


def _odour_test(ex, stats, po: fh.Projection, x: np.ndarray, B: int) -> list[dict]:
    """The odour benchmark of the confirmatory analysis (4000 mixtures of the
    rows of `x`, kappa = 10, unit-mean input) against B nulls, plus the
    six-input construction and the equal-connection controls (10 draws each)."""
    pool = ex.null_pool(po, B)
    sizes = ex.sizes_for(po)
    b0 = ex.bench(fh.mixtures(x, ex.N_ITEMS, seed=0))
    real = ex.scores(b0, po, sizes)
    null = np.array([ex.scores(b0, q, sizes) for q in pool])
    alt = {"uniform6": [fh.uniform_inputs(po, seed=20_000 + i, inputs=6) for i in range(10)]}
    from .replication import CONTROLS
    for c, kw in CONTROLS.items():
        alt[c] = [fh.margin_control(po, seed=30_000 + i, **kw) for i in range(10)]
    alt = {c: np.array([ex.scores(b0, q, sizes) for q in qs]).mean(0) for c, qs in alt.items()}
    rows = []
    for i, k in enumerate(sizes):
        rt = stats.randomization_test(real[i], null[:, i])
        mu = null[:, i].mean()
        rows.append({"k": k, "primary": k == ex.primary_k(po), "real": float(real[i]),
                     "null_mean": float(mu), "null_sd": float(null[:, i].std(ddof=1)),
                     "relative_difference": float(real[i] / mu - 1),
                     "p_two_sided": rt["p_two_sided"],
                     **{f"{c}_relative": float(v[i] / mu - 1) for c, v in alt.items()}})
    return rows


def compare(cfg: Config, B: int = 100, B_bench: int = 50, trials: int = 5,
            only: list[str] | None = None, log=print) -> dict:
    """The same analyses on every hemisphere of every connectome, at reduced
    size so that all seven run with identical settings:

      structure  glomerulus co-occurrence Q against B curveball nulls
      odours     the odour benchmark (`_odour_test`) against B nulls, on the
                 DoOR glomeruli this connectome has, and again on the DoOR
                 glomeruli that every hemisphere has ("odours_matched")
      protocol   the 2017 protocol on SIFT, GloVe, MNIST and odours
                 (`replication.score_wiring`, with the equal-connection
                 controls) against the first B_bench nulls over `trials` trials.
                 With B_bench = 50 the smallest two-sided p is 0.039; these
                 runs are effect estimates, the odour tests are the inference.

    Seven hemispheres come from four animals; hemispheres of one animal are
    not independent. Results are merged by hemisphere into
    results/connectomes.json.
    """
    import json
    import time
    from . import experiments as ex
    from . import replication as rp
    from . import stats

    path = ex.ROOT / "results" / "connectomes.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out.update({"B": B, "B_bench": B_bench, "trials": trials,
                "hemispheres": out.get("hemispheres", {})})
    projections = {f"{ds}_{sd}": projection(cfg, ds, sd) for ds, sd in HEMISPHERES}
    common = sorted(set.intersection(*(set(p.glomeruli) for p in projections.values())))
    od_common = fh.load_odours(cfg, common)
    out["common_glomeruli"] = common
    out["common_odour_glomeruli"] = od_common.glomeruli
    data = {n: rp.load_benchmark(cfg, n) for n in ("sift", "glove", "mnist")}
    for ds, side in HEMISPHERES:
        key = f"{ds}_{side}"
        if only and key not in only:
            continue
        t0 = time.time()
        p = projections[key]
        pb = fh.Projection("b", p.binary, p.glomeruli, meta=p.meta)
        pool = ex.null_pool(pb, B)
        q_null = np.array([ex._cooccurrence_q(q) for q in pool])
        q_real = ex._cooccurrence_q(pb)

        od = fh.load_odours(cfg, p.glomeruli)
        po_w = fh.align(p, od.glomeruli)
        po = fh.Projection("b", po_w.binary, po_w.glomeruli, meta=po_w.meta)
        pm = fh.align(pb, od_common.glomeruli)
        odour_rows = _odour_test(ex, stats, po, od.x, B)
        matched_rows = _odour_test(ex, stats, pm, od_common.x, B)
        protocol = {}
        for name in rp.DATASETS:
            log(f"  {key}: 2017 protocol, {name}")
            if name == "odours":
                x = rp.load_benchmark(cfg, "odours", odours=od.x)
                protocol[name] = rp.score_wiring(name, x, po, ex.null_pool(po, B)[:B_bench], trials, log)
            else:
                protocol[name] = rp.score_wiring(name, data[name], pb, pool[:B_bench], trials, log)
        out["hemispheres"][key] = {
            "dataset": ds, "side": side, "label": f"{LABELS[ds]} {side}", "sex": SEX[ds],
            "full": summary(p), "odour_projection": summary(po_w),
            "odour_glomeruli": od.glomeruli, "n_odorants": int(len(od.x)),
            "structure": {"q_real": q_real, "q_null_mean": float(q_null.mean()),
                          "q_null_sd": float(q_null.std(ddof=1)),
                          "z": float((q_real - q_null.mean()) / q_null.std(ddof=1)),
                          "test": stats.randomization_test(q_real, q_null)},
            "odours": odour_rows, "odours_matched": matched_rows,
            "protocol": protocol, "seconds": time.time() - t0}
        prim = next(r for r in odour_rows if r["primary"])
        pm_ = next(r for r in matched_rows if r["primary"])
        log(f"  {key}: Q z {out['hemispheres'][key]['structure']['z']:+.1f}; odours "
            f"{100 * prim['relative_difference']:+.2f}% (p {prim['p_two_sided']:.3f}); matched "
            f"{100 * pm_['relative_difference']:+.2f}% (p {pm_['p_two_sided']:.3f}); "
            f"{time.time() - t0:.0f}s")
        ex.save(out, "connectomes.json")
    add_degrees(cfg)
    return json.loads(path.read_text())


def add_degrees(cfg: Config) -> None:
    """Store every hemisphere's per-glomerulus fan-out and per-cell input
    counts in results/connectomes.json (read by the figures; no analysis is
    rerun)."""
    import json
    from . import experiments as ex
    path = ex.ROOT / "results" / "connectomes.json"
    out = json.loads(path.read_text())
    for ds, side in HEMISPHERES:
        h = out["hemispheres"].get(f"{ds}_{side}")
        if h is None:
            continue
        p = projection(cfg, ds, side)
        h["fan_out"] = dict(zip(p.glomeruli, p.fan_out().tolist()))
        h["inputs"] = np.bincount(p.inputs()).tolist()
    ex.save(out, "connectomes.json")


def fanout_analysis(cfg: Config) -> dict:
    """Is the skew in glomerular fan-out a conserved trait, and what does it
    track? Writes results/fanout.json:

      conservation  Spearman correlation of relative fan-out (fan-out / mean)
                    over the glomeruli all hemispheres share, for every pair
                    of hemispheres, within and between animals
      weights       per hemisphere, Spearman correlation between fan-out and
                    mean synapses per connection (positive: weights reinforce
                    the skew; negative: they compensate)
      odours        per hemisphere, Spearman correlation between fan-out and
                    DoOR response breadth (fraction of measured odorants with
                    response > 0.2) and response SD, over the DoOR glomeruli.
                    All hemispheres use the same DoOR data and conserved
                    fan-out, so these are not independent tests.
    """
    import itertools
    import json
    from scipy.stats import spearmanr
    from . import experiments as ex
    proj = {f"{ds}_{sd}": projection(cfg, ds, sd) for ds, sd in HEMISPHERES}
    common = sorted(set.intersection(*(set(p.glomeruli) for p in proj.values())))
    rel = {}
    out = {"common": common, "hemispheres": list(proj), "weights": {}, "odours": {}}
    for k, p in proj.items():
        b = p.matrix > 0
        fan = b.sum(1)
        rel[k] = {g: float(f / fan.mean()) for g, f in zip(p.glomeruli, fan)}
        r = spearmanr(fan, p.matrix.sum(1) / np.maximum(fan, 1))
        out["weights"][k] = {"rho": float(r.correlation), "p": float(r.pvalue)}
        od = fh.load_odours(cfg, p.glomeruli)
        x = np.where(od.observed, od.x, np.nan)
        f = np.array([fan[p.glomeruli.index(g)] for g in od.glomeruli])
        res = {}
        for name, v in (("breadth", np.nanmean(x > 0.2, 0)), ("sd", np.nanstd(x, 0))):
            r = spearmanr(f, v)
            res[name] = {"rho": float(r.correlation), "p": float(r.pvalue)}
        out["odours"][k] = res
    M = np.array([[rel[k][g] for g in common] for k in proj])
    pairs = []
    for a, b in itertools.combinations(range(len(proj)), 2):
        ka, kb = list(proj)[a], list(proj)[b]
        pairs.append({"a": ka, "b": kb, "same_animal": ka.split("_")[0] == kb.split("_")[0],
                      "rho": float(spearmanr(M[a], M[b]).correlation)})
    out["conservation"] = pairs
    out["fanout_rel"] = M.tolist()
    ex.save(out, "fanout.json")
    return out
