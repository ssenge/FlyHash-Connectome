"""Is the fly's mushroom body a better hash function than chance?

The olfactory circuit is a known algorithm. Odour identity arrives on ~51
glomeruli, expands onto ~2000 Kenyon cells that each sample about six
glomeruli, and a global inhibitory neuron keeps only the most active few
percent. Dasgupta, Stevens and Navlakha (Science, 2017) showed that this
shape -- sparse expansion followed by winner-take-all -- is a locality
sensitive hash that beats conventional LSH at similarity search.

They had to use *random* wiring, because the real connectivity was unknown.
It is known now. This module runs the same algorithm with the measured
glomerulus-to-Kenyon-cell wiring and asks whether evolution's version beats
a random one with identical statistics.

The controls are the point:

  real          the measured connectome
  shuffle_kc    every Kenyon cell keeps its number of claws, but samples
                glomeruli uniformly at random
  shuffle_both  a configuration model: claws per cell AND the number of cells
                each glomerulus feeds are both preserved, only the pairing is
                destroyed. This is the strict null.
  random_claws  the 2017 paper's own construction, six random glomeruli per cell
  gaussian_lsh  classical dense random projection, for reference

If `real` does not beat `shuffle_both`, then the wiring carries no structure
beyond its degree sequence, and that is a clean negative answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Glomeruli that are olfactory. VP* are thermo/hygro and MZ is not a glomerulus.
_NON_OLFACTORY = re.compile(r"^(VP|MZ)")


@dataclass
class Projection:
    """A glomerulus -> Kenyon cell matrix, however it was produced."""
    name: str
    matrix: np.ndarray            # (n_glomeruli, n_cells), synapse counts
    glomeruli: list[str]

    @property
    def binary(self) -> np.ndarray:
        return (self.matrix > 0).astype(np.float64)

    def claws(self) -> np.ndarray:
        return (self.matrix > 0).sum(axis=0)

    def fan_out(self) -> np.ndarray:
        return (self.matrix > 0).sum(axis=1)


# --------------------------------------------------------------------------- wiring

def glomerulus_of(cell_type: str | float) -> str | None:
    """DA1_lPN -> DA1. Multiglomerular (M_*) and unnamed (CB*) return None."""
    if not isinstance(cell_type, str) or cell_type.startswith(("M_", "CB")):
        return None
    return cell_type.split("_")[0]


def mushroom_body(graph, side: str = "R", olfactory_only: bool = True) -> Projection:
    """The measured glomerulus -> Kenyon cell projection of one hemisphere."""
    import scipy.sparse as sp

    nodes = graph.nodes
    pn = nodes[(nodes["class"] == "ALPN") & (nodes["somaSide"] == side)].copy()
    kc = nodes[(nodes["class"] == "Kenyon_Cell") & (nodes["somaSide"] == side)]
    pn["glom"] = pn["type"].map(glomerulus_of)
    pn = pn[pn["glom"].notna()]
    if olfactory_only:
        pn = pn[~pn["glom"].str.match(_NON_OLFACTORY)]
    if pn.empty or kc.empty:
        raise ValueError(f"no ALPN/Kenyon cells found for side {side!r}")

    a = sp.csr_matrix((graph.weight, graph.indices, graph.indptr),
                      shape=(graph.n, graph.n))
    sub = a[pn["idx"].to_numpy()][:, kc["idx"].to_numpy()].toarray()

    gloms = sorted(pn["glom"].unique())
    index = {g: i for i, g in enumerate(gloms)}
    m = np.zeros((len(gloms), sub.shape[1]))
    for row, g in enumerate(pn["glom"]):
        m[index[g]] += sub[row]

    keep = (m > 0).sum(axis=0) > 0        # cells with no olfactory input carry nothing
    return Projection("real", m[:, keep], gloms)


# --------------------------------------------------------------------------- controls

def shuffle_kc(p: Projection, seed: int = 0) -> Projection:
    """Each cell keeps its claw count; the glomeruli it samples are random."""
    rng = np.random.default_rng(seed)
    n_g, n_c = p.matrix.shape
    out = np.zeros_like(p.matrix)
    weights = p.matrix
    for c in range(n_c):
        src = np.flatnonzero(weights[:, c] > 0)
        pick = rng.choice(n_g, len(src), replace=False)
        out[pick, c] = weights[src, c]
    return Projection("shuffle_kc", out, p.glomeruli)


def shuffle_both(p: Projection, seed: int = 0, sweeps: int = 30) -> Projection:
    """Configuration model by curveball swaps, preserving both margins exactly.

    A naive approach -- permuting edge endpoints and discarding collisions --
    loses a few percent of the connections and is therefore not a sample from
    the fixed-margin space at all. The curveball algorithm of Strona et al.
    instead repeatedly picks two Kenyon cells, takes the glomeruli unique to
    each, and redeals them at random between the two. Every cell keeps its claw
    count and every glomerulus its fan-out by construction, while the pairing
    is randomised. `sweeps` swap attempts per cell is well past the mixing time
    for a matrix this sparse.
    """
    rng = np.random.default_rng(seed)
    n_g, n_c = p.matrix.shape
    cells = [set(np.flatnonzero(p.matrix[:, c] > 0).tolist()) for c in range(n_c)]

    for _ in range(sweeps * n_c):
        i, j = rng.integers(0, n_c, 2)
        if i == j:
            continue
        a, b = cells[i], cells[j]
        shared = a & b
        free = list((a | b) - shared)
        if len(free) < 2:
            continue
        rng.shuffle(free)
        k = len(a) - len(shared)
        cells[i] = shared | set(free[:k])
        cells[j] = shared | set(free[k:])

    out = np.zeros_like(p.matrix)
    weights = p.matrix[p.matrix > 0]
    weights = rng.permutation(weights)            # keep the weight distribution
    at = 0
    for c, gl in enumerate(cells):
        idx = sorted(gl)
        out[idx, c] = weights[at:at + len(idx)]
        at += len(idx)
    return Projection("shuffle_both", out, p.glomeruli)


def random_claws(p: Projection, seed: int = 0, claws: int | None = None) -> Projection:
    """The 2017 paper's construction: a fixed number of random glomeruli per cell."""
    rng = np.random.default_rng(seed)
    n_g, n_c = p.matrix.shape
    k = claws or int(round(p.claws().mean()))
    out = np.zeros_like(p.matrix)
    for c in range(n_c):
        out[rng.choice(n_g, k, replace=False), c] = 1.0
    return Projection("random_claws", out, p.glomeruli)


def balanced(p: Projection, seed: int = 0) -> Projection:
    """Positive control: every glomerulus feeds equally many cells.

    The real fan-out is lopsided, so this is the obvious way a wiring could be
    *better* for general-purpose hashing. If the test cannot see this as an
    improvement, it has no power to detect one anywhere.
    """
    rng = np.random.default_rng(seed)
    n_g, n_c = p.matrix.shape
    claws = (p.matrix > 0).sum(axis=0)
    stubs = np.repeat(np.arange(n_g), int(np.ceil(claws.sum() / n_g)) + 1)
    rng.shuffle(stubs)
    out = np.zeros_like(p.matrix)
    pos = 0
    for c in range(n_c):
        chosen, guard = set(), 0
        while len(chosen) < int(claws[c]) and guard < 500:
            chosen.add(int(stubs[pos % len(stubs)]))
            pos += 1
            guard += 1
        if chosen:
            out[list(chosen), c] = 1.0
    return Projection("balanced", out, p.glomeruli)


def degenerate(p: Projection, seed: int = 0, n_used: int = 4) -> Projection:
    """Negative control: every cell samples the same handful of glomeruli."""
    rng = np.random.default_rng(seed)
    n_g, n_c = p.matrix.shape
    pool = rng.choice(n_g, n_used, replace=False)
    claws = (p.matrix > 0).sum(axis=0)
    out = np.zeros_like(p.matrix)
    for c in range(n_c):
        k = min(max(int(claws[c]), 1), n_used)
        out[rng.choice(pool, k, replace=False), c] = 1.0
    return Projection("degenerate", out, p.glomeruli)


CONTROLS = {"shuffle_kc": shuffle_kc, "shuffle_both": shuffle_both,
            "random_claws": random_claws}
# Not nulls: these bracket the test's sensitivity in both directions.
SENSITIVITY = {"balanced": balanced, "degenerate": degenerate}


def mixtures(x: np.ndarray, n: int, seed: int = 0,
             lo: int = 2, hi: int = 5, with_clusters: bool = False):
    """Blends of measured odorants, which is what a fly meets in the world.

    DoOR tops out near 250 usable odorants, too few to resolve a small effect.
    Mixing real responses keeps the natural correlation structure while giving
    enough items for the null spread to tighten.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n, x.shape[1]))
    cluster = np.zeros(n, np.int32)
    for i in range(n):
        k = int(rng.integers(lo, hi + 1))
        pick = rng.choice(len(x), k, replace=False)
        w = rng.dirichlet(np.ones(k))
        out[i] = (x[pick] * w[:, None]).sum(axis=0)
        cluster[i] = int(pick[np.argmax(w)])     # dominant component
    return (out, cluster) if with_clusters else out


def equivalence(real_ap: np.ndarray, null_aps: list[np.ndarray],
                cluster: np.ndarray | None, seed: int = 0,
                draws: int = 2000) -> dict:
    """A confidence interval on (real - null), resampling whole odorants.

    Queries built from the same measured odorant are not independent, so the
    bootstrap resamples odorants rather than queries. The interval carries both
    sources of variation, across items and across null matrices. Equivalence at
    a margin delta is declared when the whole interval lies inside
    (-delta, +delta), which is the two-one-sided-tests criterion; a
    non-significant difference on its own would not license the claim.
    """
    rng = np.random.default_rng(seed)
    n = len(real_ap)
    groups = (np.arange(n) if cluster is None else cluster)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    null_mat = np.vstack(null_aps)

    deltas = np.empty(draws)
    for b in range(draws):
        pick = rng.choice(uniq, len(uniq), replace=True)
        rows = np.concatenate([index[g] for g in pick])
        deltas[b] = real_ap[rows].mean() - null_mat[:, rows].mean()
    base = float(null_mat.mean())
    lo, hi = np.percentile(deltas, [5, 95])       # 90% CI, the TOST convention
    return {"delta": float(real_ap.mean() - base),
            "ci90": [float(lo), float(hi)],
            "relative_ci90": [float(lo / base), float(hi / base)],
            "equivalence_margin": float(max(abs(lo), abs(hi)) / base),
            "n_clusters": int(len(uniq))}


# --------------------------------------------------------------------------- the hash

def normalise(x: np.ndarray) -> np.ndarray:
    """Divisive normalisation: the antennal lobe removes concentration."""
    m = x.mean(axis=1, keepdims=True)
    return np.divide(x, m, out=np.zeros_like(x), where=m > 0)


def tags(x: np.ndarray, p: Projection, k: int, weighted: bool = False) -> np.ndarray:
    """Odours -> sparse binary tags: normalise, expand, keep the top k cells."""
    m = p.matrix if weighted else p.binary
    y = normalise(x) @ m
    out = np.zeros(y.shape, bool)
    if k >= y.shape[1]:
        return np.ones_like(out)
    win = np.argpartition(-y, k, axis=1)[:, :k]
    np.put_along_axis(out, win, True, axis=1)
    return out


def gaussian_lsh(x: np.ndarray, bits: int, seed: int = 0) -> np.ndarray:
    """Classical LSH: dense random projection, sign as the bit."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(x.shape[1], bits))
    return (normalise(x) @ w) > 0


# --------------------------------------------------------------------------- scoring

def true_neighbours(x: np.ndarray, k: int) -> np.ndarray:
    """The k nearest items by Euclidean distance: the ground truth to recover.

    This does not depend on the wiring, so callers comparing many matrices
    should compute it once and pass it in; it is by far the costliest step.
    """
    sq = (x ** 2).sum(1)
    d = sq[:, None] + sq[None, :] - 2 * (x @ x.T)
    np.fill_diagonal(d, np.inf)
    return np.argpartition(d, k, axis=1)[:, :k]


def mean_average_precision(x: np.ndarray, tag: np.ndarray, k: int = 10,
                           truth: np.ndarray | None = None,
                           per_item: bool = False):
    """How well does the hash recover each item's true nearest neighbours?

    Kept dense on purpose. A sparse overlap product looks tempting because fly
    tags are sparse, but ranking by overlap only equals ranking by Hamming
    distance when every tag has the same number of ones. The Gaussian LSH
    baseline does not, so the shortcut silently mis-scores it.
    """
    if truth is None:
        truth = true_neighbours(x, k)
    t = tag.astype(np.float32)
    ones = t.sum(1)
    ham = ones[:, None] + ones[None, :] - 2 * (t @ t.T)
    np.fill_diagonal(ham, np.inf)
    top = np.argpartition(ham, k, axis=1)[:, :k]
    top = np.take_along_axis(top, np.argsort(np.take_along_axis(ham, top, 1), axis=1), 1)

    n = len(x)
    want = np.zeros((n, n), bool)
    np.put_along_axis(want, truth, True, axis=1)
    hit = np.take_along_axis(want, top, axis=1)            # (n, k) ranked hits
    cum = np.cumsum(hit, axis=1)
    ranks = np.arange(1, k + 1)
    prec = np.where(hit, cum / ranks, 0.0)
    per_query = prec.sum(axis=1) / truth.shape[1]
    return per_query if per_item else float(per_query.mean())


# --------------------------------------------------------------------------- odours

DOOR_BASE = "https://raw.githubusercontent.com/ropensci/DoOR.data/master/data"


def _fetch(url: str, dest) -> None:
    import urllib.request
    from pathlib import Path
    dest = Path(dest)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def load_odours(cfg, glomeruli: list[str], min_receptors: int = 12,
                min_odours: int = 40):
    """Real odour responses (DoOR 2.0) as an odorant x glomerulus matrix.

    DoOR's consensus matrix is only ~14% filled, so this keeps the receptors
    and odorants that are best measured and treats the remaining gaps as
    no response, which is what an unmeasured weak pairing usually is.
    Returns (matrix, odorant names, glomerulus names actually used).
    """
    import pandas as pd

    raw = cfg.raw_dir
    _fetch(f"{DOOR_BASE}/door_response_matrix.csv", raw / "door_response_matrix.csv")
    _fetch(f"{DOOR_BASE}/door_mappings.csv", raw / "door_mappings.csv")
    resp = pd.read_csv(raw / "door_response_matrix.csv", sep=";")
    maps = pd.read_csv(raw / "door_mappings.csv", sep=";")

    # The response matrix is keyed by receptor (Or10a, ...), so join on that
    # and drop ambiguous assignments like "DL2d/v" or "DM5+DM3".
    pairs = maps[["receptor", "glomerulus"]].dropna()
    to_glom = {str(r.receptor): str(r.glomerulus) for r in pairs.itertuples()
               if not set("?/+") & set(str(r.glomerulus))}
    wanted = set(glomeruli)

    cols, names = [], []
    for c in resp.select_dtypes("number").columns:
        g = to_glom.get(str(c))
        if g in wanted:
            cols.append(c)
            names.append(g)
    if not cols:
        raise ValueError("no DoOR receptor maps onto a connectome glomerulus")

    m = resp[cols].copy()
    m.columns = names
    m = m.T.groupby(level=0).mean().T          # several receptors per glomerulus

    good_cols = m.columns[m.notna().sum(axis=0) >= min_odours]
    m = m[good_cols]
    m = m[m.notna().sum(axis=1) >= min_receptors]
    m = m.fillna(0.0)
    x = m.to_numpy(float)
    x = np.clip(x, 0.0, None)
    keep = x.sum(axis=1) > 0
    return x[keep], list(m.index[keep]), list(m.columns)


def align(p: Projection, used: list[str]) -> Projection:
    """Restrict a projection to the glomeruli a dataset actually covers."""
    idx = [p.glomeruli.index(g) for g in used]
    sub = p.matrix[idx]
    keep = (sub > 0).sum(axis=0) > 0
    return Projection(p.name, sub[:, keep], list(used))


# --------------------------------------------------------------------------- experiment

def experiment(graph, cfg, *, side: str = "R", sizes=(8, 16, 32, 64),
               seeds: int = 20, neighbours: int = 10, weighted: bool = False,
               dataset: str = "mixtures", n_items: int = 4000) -> dict:
    """Real wiring against its degree-matched nulls, plus classical LSH.

    `dataset="odours"` uses the ~172 measured odorants directly; that is real
    but underpowered, and at that size nothing short of a wrecked matrix is
    detectable. `dataset="mixtures"` blends them up to `n_items`, which tightens
    the null enough for a few-percent advantage to show. The `sensitivity`
    block reports whether the test can in fact see one.
    """
    full = mushroom_body(graph, side=side)
    cluster = None
    if dataset in ("odours", "mixtures"):
        x, names, used = load_odours(cfg, full.glomeruli)
        proj = align(full, used)
        if dataset == "mixtures":
            x, cluster = mixtures(x, n_items, seed=int(cfg["seed"]),
                                  with_clusters=True)
    else:
        rng = np.random.default_rng(int(cfg["seed"]))
        d = len(full.glomeruli)
        x = np.abs(rng.normal(size=(n_items, 6)) @ rng.normal(size=(6, d)))
        names, used, proj = [], full.glomeruli, full

    truth = true_neighbours(x, neighbours)      # matrix-independent; compute once
    rows = []
    for k in sizes:
        real_ap = mean_average_precision(
            x, tags(x, proj, k, weighted=weighted), neighbours, truth,
            per_item=True)
        entry = {"hash_size": int(k), "real": float(real_ap.mean())}
        conf_aps = []
        for cname, fn in CONTROLS.items():
            aps = [mean_average_precision(
                x, tags(x, fn(proj, seed=s), k, weighted=weighted), neighbours,
                truth, per_item=True) for s in range(seeds)]
            if cname == "shuffle_both":
                conf_aps = aps
            v = np.array([a.mean() for a in aps])
            entry[cname] = {"mean": float(v.mean()), "std": float(v.std()),
                            "nulls_beating_real": int((v >= entry["real"]).sum())}
        entry["equivalence"] = equivalence(real_ap, conf_aps, cluster,
                                           seed=int(cfg["seed"]))
        lsh = np.array([mean_average_precision(x, gaussian_lsh(x, k, seed=s),
                                               neighbours, truth)
                        for s in range(seeds)])
        entry["gaussian_lsh"] = {"mean": float(lsh.mean()), "std": float(lsh.std())}
        null = entry["shuffle_both"]
        entry["z_vs_strict_null"] = float(
            (entry["real"] - null["mean"]) / (null["std"] + 1e-12))
        # The smallest advantage this run could have seen at 2 sigma, relative
        # to the null. Any true advantage must be smaller than this.
        entry["min_detectable_effect"] = float(
            2.0 * null["std"] / max(1e-12, null["mean"]))
        # Can this test see an advantage at all? Bracket it in both directions.
        entry["sensitivity"] = {}
        for sname, fn in SENSITIVITY.items():
            score = mean_average_precision(
                x, tags(x, fn(proj, seed=0), k, weighted=weighted), neighbours, truth)
            entry["sensitivity"][sname] = {
                "map": float(score),
                "z": float((score - null["mean"]) / (null["std"] + 1e-12))}
        rows.append(entry)

    return {"dataset": dataset, "side": side, "weighted": weighted,
            "n_items": int(x.shape[0]), "n_glomeruli": int(x.shape[1]),
            "n_cells": int(proj.matrix.shape[1]),
            "claws_per_cell": float(proj.claws().mean()),
            "cells_per_glomerulus": [int(proj.fan_out().min()),
                                     int(proj.fan_out().max())],
            "seeds": seeds, "neighbours": neighbours, "rows": rows}


def print_experiment(res: dict) -> None:
    print(f"mushroom body as a hash function  ({res['dataset']}, "
          f"{res['side']} hemisphere{', synapse-weighted' if res['weighted'] else ''})")
    print(f"  {res['n_items']} items x {res['n_glomeruli']} glomeruli "
          f"-> {res['n_cells']} Kenyon cells, {res['claws_per_cell']:.2f} claws each, "
          f"{res['cells_per_glomerulus'][0]}-{res['cells_per_glomerulus'][1]} cells per glomerulus")
    print(f"  mean average precision at {res['neighbours']} neighbours, "
          f"{res['seeds']} seeds per control\n")
    head = ["hash", "real", "shuffle_kc", "shuffle_both", "random_claws", "gaussian_lsh"]
    print("  " + " ".join(f"{h:>16s}" for h in head))
    for r in res["rows"]:
        cells = [f"{r['hash_size']:>16d}", f"{r['real']:>16.4f}"]
        for c in ("shuffle_kc", "shuffle_both", "random_claws", "gaussian_lsh"):
            cells.append(f"{r[c]['mean']:.4f}+-{r[c]['std']:.3f}".rjust(16))
        print("  " + " ".join(cells))
    print("\n  real vs the strict null (shuffle_both):")
    for r in res["rows"]:
        n = r["shuffle_both"]
        print(f"    hash {r['hash_size']:3d}   z = {r['z_vs_strict_null']:+5.2f}   "
              f"nulls beating real: {n['nulls_beating_real']}/{res['seeds']}")
    print("\n  equivalence: 90% CI on (real - null), odorant-level bootstrap")
    for r in res["rows"]:
        e = r["equivalence"]
        lo, hi = e["relative_ci90"]
        print(f"    hash {r['hash_size']:3d}   {e['delta']:+.4f} absolute   "
              f"[{lo:+.1%}, {hi:+.1%}] relative   "
              f"equivalent within {e['equivalence_margin']:.1%}")
    print("\n  sensitivity: does the protocol see a difference that is there?")
    for r in res["rows"]:
        se = r["sensitivity"]
        print(f"    hash {r['hash_size']:3d}   balanced fan-out z = {se['balanced']['z']:+5.2f}"
              f"   wrecked z = {se['degenerate']['z']:+8.2f}"
              f"   (2-sigma resolution {r['min_detectable_effect']:.1%})")
    zs = [r["z_vs_strict_null"] for r in res["rows"]]
    arch = np.mean([r["real"] / max(1e-9, r["gaussian_lsh"]["mean"]) for r in res["rows"]])
    marg = min(r["equivalence"]["equivalence_margin"] for r in res["rows"])
    print(f"\n  architecture beats classical LSH by {arch:.1f}x.")
    print(f"  measured wiring vs degree-matched random: z in "
          f"[{min(zs):+.2f}, {max(zs):+.2f}]; equivalent to the null "
          f"within {marg:.1%} at the tightest hash size.")


# --------------------------------------------------------------------------- figure

# Categorical slots from the validated palette (light surface #fcfcfb).
_BLUE, _ORANGE, _AQUA, _YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
_INK, _MUTED, _SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def figure(res: dict, path) -> None:
    """Two panels: the architecture wins, the wiring does not."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 220, "font.size": 10, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c9c8c3", "axes.labelcolor": _MUTED,
        "xtick.color": _MUTED, "ytick.color": _MUTED,
        "figure.facecolor": _SURFACE, "axes.facecolor": _SURFACE,
    })
    rows = res["rows"]
    sizes = [r["hash_size"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.3))

    # The three fly-architecture traces overlap almost exactly - that is the
    # finding - so they get a legend rather than colliding direct labels.
    series = [("fly wiring", [r["real"] for r in rows], _BLUE, "-"),
              ("degree-matched null", [r["shuffle_both"]["mean"] for r in rows], _ORANGE, "-"),
              ("random claws (2017 model)", [r["random_claws"]["mean"] for r in rows], _AQUA, "--"),
              ("classical LSH", [r["gaussian_lsh"]["mean"] for r in rows], _YELLOW, "-")]
    for label, y, colour, style in series:
        ax1.plot(sizes, y, style, color=colour, lw=2, marker="o", ms=5,
                 markeredgecolor=_SURFACE, markeredgewidth=1.2, label=label)
    leg = ax1.legend(loc="lower right", frameon=False, fontsize=7.5,
                     handlelength=1.6, borderpad=0.2, labelspacing=0.35)
    for text in leg.get_texts():
        text.set_color(_MUTED)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sizes); ax1.set_xticklabels(sizes)
    ax1.set_xlabel("hash size (Kenyon cells kept)")
    ax1.set_ylabel("mean average precision")
    ax1.set_title("The architecture wins", color=_INK, fontsize=11, loc="left")
    ax1.set_xlim(sizes[0] * 0.92, sizes[-1] * 1.08)
    ax1.grid(axis="y", color="#e8e7e2", lw=0.8)
    ax1.set_axisbelow(True)

    # Panel B: the effect size with its bootstrap interval, which is what the
    # equivalence claim rests on. A z-score alone would not show the interval.
    rel = [100 * r["equivalence"]["delta"] / r["shuffle_both"]["mean"] for r in rows]
    lo = [100 * r["equivalence"]["relative_ci90"][0] for r in rows]
    hi = [100 * r["equivalence"]["relative_ci90"][1] for r in rows]
    err = np.vstack([np.array(rel) - np.array(lo), np.array(hi) - np.array(rel)])

    ax2.axhspan(-3, 3, color="#eeeeea", zorder=0)
    ax2.annotate("within 3% of the null", (sizes[0], 2.4), xytext=(2, 0),
                 textcoords="offset points", color=_MUTED, fontsize=8)
    ax2.axhline(0, color="#9a9992", lw=1.2, zorder=1)
    ax2.errorbar(sizes, rel, yerr=err, fmt="o-", color=_BLUE, lw=2, ms=5,
                 capsize=3, markeredgecolor=_SURFACE, markeredgewidth=1.2,
                 zorder=3, label="fly wiring, 90% CI")
    ax2.annotate("fly wiring", (sizes[-1], rel[-1]), xytext=(6, -2),
                 textcoords="offset points", color=_BLUE, fontsize=9,
                 va="center", fontweight="medium")
    worst = min(r["sensitivity"]["degenerate"]["z"] for r in rows)
    ax2.annotate(f"wrecked wiring lies at z = {worst:.0f}, far off scale",
                 (sizes[0], -7.2), xytext=(2, 0), textcoords="offset points",
                 color=_MUTED, fontsize=8)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(sizes); ax2.set_xticklabels(sizes)
    ax2.set_xlabel("hash size (Kenyon cells kept)")
    ax2.set_ylabel("retrieval vs degree-matched null (%)")
    ax2.set_title("The wiring does not", color=_INK, fontsize=11, loc="left")
    ax2.set_xlim(sizes[0] * 0.88, sizes[-1] * 1.35)
    ax2.set_ylim(-8, 5)
    ax2.grid(axis="y", color="#e8e7e2", lw=0.8)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
