"""The fly hashing algorithm, run on measured mushroom body wiring.

Dasgupta, Stevens and Navlakha (Science, 2017) described the fly olfactory
circuit as a locality-sensitive hash: glomerular input is normalised, expanded
through a sparse binary projection onto Kenyon cells, and sparsified by
winner-take-all. They drew the projection at random because the connectivity
was unknown. This module builds the projection from the MaleCNS connectome and
provides the null and comparison matrices, the hash, the retrieval benchmark
and the resource accounting used to compare it with Gaussian sign hashing.

Terminology. A column of the projection is a Kenyon cell; its number of
non-zero entries is the number of *distinct glomerular inputs* of that cell.
This is not an anatomical claw count: projection neurons are aggregated by
glomerulus, so several contacts, or several projection neurons of one
glomerulus, collapse into one input. A row's number of non-zero entries is the
glomerulus's *fan-out*.

Matrices compared in the experiments (see `flypath.experiments`):

  real              the measured projection
  curveball         uniform over binary matrices with the same row and column
                    sums (Strona et al. 2014; uniformity: Carstens 2015). This
                    is the null of record: only the pairing is randomised.
  shuffle_inputs    each cell keeps its number of inputs; the glomeruli it
                    samples are uniform, so fan-out is flattened on average
  uniform_inputs    every cell samples the same number of uniformly chosen
                    glomeruli; six by default, the construction of the 2017 paper
  balanced_fanout   per-cell input counts kept, fan-out made as even as the
                    counts allow, pairing randomised. An alternative wiring, not
                    a control of known effect size.
  degenerate        every cell restricted to the same four glomeruli; a
                    deliberately broken matrix used as a sanity probe
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

import numpy as np

# Glomeruli that are olfactory. VP* are thermo/hygro and MZ is not a glomerulus.
_NON_OLFACTORY = re.compile(r"^(VP|MZ)")


@dataclass
class Projection:
    """A glomerulus -> Kenyon cell matrix, however it was produced."""
    name: str
    matrix: np.ndarray            # (n_glomeruli, n_cells); synapse counts or 0/1
    glomeruli: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def binary(self) -> np.ndarray:
        return (self.matrix > 0).astype(np.float64)

    def inputs(self) -> np.ndarray:
        """Distinct glomerular inputs per Kenyon cell (column sums of the binary matrix)."""
        return (self.matrix > 0).sum(axis=0)

    def fan_out(self) -> np.ndarray:
        """Kenyon cells reached by each glomerulus (row sums of the binary matrix)."""
        return (self.matrix > 0).sum(axis=1)

    @property
    def nnz(self) -> int:
        return int((self.matrix > 0).sum())


# --------------------------------------------------------------------------- wiring

def glomerulus_of(cell_type: str | float) -> str | None:
    """DA1_lPN -> DA1. Multiglomerular (M_*) and unnamed (CB*) return None."""
    if not isinstance(cell_type, str) or cell_type.startswith(("M_", "CB")):
        return None
    return cell_type.split("_")[0]


def mushroom_body(graph, side: str = "R", olfactory_only: bool = True) -> Projection:
    """The measured glomerulus -> Kenyon cell projection of one hemisphere.

    `meta["pn_counts"]` records, per glomerulus and retained cell, the number
    of distinct projection neurons of that glomerulus contacting the cell;
    `meta["pn_partners"]` is its column sum (distinct PN partners per cell).
    Both follow the cells and glomeruli through `align`.
    """
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
    counts = np.zeros((len(gloms), sub.shape[1]), np.int64)
    for row, g in enumerate(pn["glom"]):
        m[index[g]] += sub[row]
        counts[index[g]] += sub[row] > 0

    keep = (m > 0).sum(axis=0) > 0        # cells with no olfactory input carry nothing
    return Projection("real", m[:, keep], gloms,
                      meta={"pn_counts": counts[:, keep], "pn_partners": counts[:, keep].sum(0),
                            "side": side, "n_pn": int(len(pn))})


def align(p: Projection, used: list[str]) -> Projection:
    """Restrict a projection to the glomeruli a dataset covers, dropping cells
    left without any input."""
    idx = [p.glomeruli.index(g) for g in used]
    sub = p.matrix[idx]
    keep = (sub > 0).sum(axis=0) > 0
    meta = dict(p.meta)
    if "pn_counts" in meta:
        meta["pn_counts"] = np.asarray(meta["pn_counts"])[idx][:, keep]
        meta["pn_partners"] = meta["pn_counts"].sum(0)
    elif "pn_partners" in meta:
        meta["pn_partners"] = np.asarray(meta["pn_partners"])[keep]
    return Projection(p.name, sub[:, keep], list(used), meta=meta)


# --------------------------------------------------------------------------- nulls

def curveball(p: Projection, seed: int = 0, sweeps: int = 30,
              checkpoints: tuple[float, ...] = ()) -> Projection | list[Projection]:
    """Uniform sample of binary matrices with the same row and column sums.

    Curveball trades (Strona et al. 2014): draw two Kenyon cells, pool the
    glomeruli each has and the other lacks, and redeal the pool between them at
    the original counts. Both margins are invariant by construction. Carstens
    (2015) proved the chain converges to the uniform distribution provided
    failed trades are kept as steps that leave the matrix unchanged, which is
    what the loop below does (`continue` still consumes an attempt).

    One sweep is `n_cells` trade attempts. The default of 30 sweeps is well
    past the point where the diagnostics in `experiments.convergence` stop
    moving; see results/convergence.json.

    With synapse weights, each cell keeps its own multiset of weights and
    assigns them to its new glomeruli in random order. That preserves each
    cell's total input strength and weight distribution; per-glomerulus total
    strength is *not* preserved.

    If `checkpoints` is given, returns the matrix after each listed number of
    sweeps instead of only the final one (used for mixing diagnostics).
    """
    rng = np.random.default_rng(seed)
    n_g, n_c = p.matrix.shape
    cells = [set(np.flatnonzero(p.matrix[:, c] > 0).tolist()) for c in range(n_c)]
    weights = [p.matrix[np.flatnonzero(p.matrix[:, c] > 0), c] for c in range(n_c)]

    def materialise(done_sweeps: float) -> Projection:
        out = np.zeros_like(p.matrix)
        for c, gl in enumerate(cells):
            idx = sorted(gl)
            if idx:
                out[idx, c] = rng.permutation(weights[c])
        return Projection("curveball", out, p.glomeruli,
                          meta={"sweeps": float(done_sweeps)})

    targets = sorted(set(int(round(s * n_c)) for s in checkpoints)) if checkpoints else []
    total = max([int(sweeps * n_c)] + targets)
    snaps = []
    if 0 in targets:
        snaps.append(materialise(0 / n_c))
    for step in range(1, total + 1):
        i, j = rng.integers(0, n_c, 2)
        if i != j:
            a, b = cells[i], cells[j]
            shared = a & b
            free = list((a | b) - shared)
            if free:
                rng.shuffle(free)
                k = len(a) - len(shared)
                cells[i] = shared | set(free[:k])
                cells[j] = shared | set(free[k:])
        if targets and step in targets:
            snaps.append(materialise(step / n_c))
    if checkpoints:
        return snaps
    return materialise(total / n_c)


def shuffle_inputs(p: Projection, seed: int = 0) -> Projection:
    """Each cell keeps its number of inputs (and weights); the glomeruli are uniform."""
    rng = np.random.default_rng(seed)
    n_g, n_c = p.matrix.shape
    out = np.zeros_like(p.matrix)
    for c in range(n_c):
        src = np.flatnonzero(p.matrix[:, c] > 0)
        pick = rng.choice(n_g, len(src), replace=False)
        out[pick, c] = p.matrix[src, c]
    return Projection("shuffle_inputs", out, p.glomeruli)


def uniform_inputs(p: Projection, seed: int = 0, inputs: int = 6) -> Projection:
    """Every cell samples `inputs` uniformly chosen glomeruli (2017 construction: 6)."""
    rng = np.random.default_rng(seed)
    n_g, n_c = p.matrix.shape
    if not 0 < inputs <= n_g:
        raise ValueError(f"inputs must be in 1..{n_g}, got {inputs}")
    out = np.zeros_like(p.matrix)
    for c in range(n_c):
        out[rng.choice(n_g, inputs, replace=False), c] = 1.0
    return Projection(f"uniform_inputs_{inputs}", out, p.glomeruli)


def balanced_fanout(p: Projection, seed: int = 0, sweeps: int = 30) -> Projection:
    """Per-cell input counts kept, fan-out as even as they allow, pairing random.

    Builds one matrix with the target margins by giving each cell (largest
    first) the glomeruli with the most remaining capacity, then randomises the
    pairing with curveball, which leaves both margins untouched.
    """
    rng = np.random.default_rng(seed)
    n_g, n_c = p.matrix.shape
    d = p.inputs()
    total = int(d.sum())
    cap = np.full(n_g, total // n_g)
    cap[rng.choice(n_g, total - cap.sum(), replace=False)] += 1
    out = np.zeros((n_g, n_c))
    for c in np.argsort(-d, kind="stable"):
        order = np.lexsort((rng.random(n_g), -cap))     # most capacity first
        chosen = order[:d[c]]
        if (cap[chosen] <= 0).any():
            raise RuntimeError("balanced margins not realisable")
        out[chosen, c] = 1.0
        cap[chosen] -= 1
    q = curveball(Projection("balanced_fanout", out, p.glomeruli), seed=seed + 1,
                  sweeps=sweeps)
    return Projection("balanced_fanout", q.matrix, p.glomeruli)


def _even(total: int, n: int, rng) -> np.ndarray:
    a = np.full(n, total // n)
    a[rng.choice(n, total - a.sum(), replace=False)] += 1
    return a


def margin_control(p: Projection, seed: int = 0, equal_in: bool = False,
                   equal_out: bool = False, sweeps: int = 30) -> Projection:
    """A binary matrix with exactly nnz(p) connections whose margins are
    either kept or made as even as the total allows: per-cell input counts
    (`equal_in`) and per-glomerulus fan-out (`equal_out`). The pairing is then
    randomised by curveball, which preserves both margins. With neither flag
    this is a draw from the null; with both it is the most regular matrix at
    the same operation count. Unlike the six-input construction, every
    variant has the same number of connections as the connectome.

    The margins are realised greedily (cells with most inputs first, each
    taking the glomeruli with most remaining capacity), which succeeds for any
    realisable pair of margins (Gale-Ryser).
    """
    rng = np.random.default_rng(seed)
    n_g, n_c = p.matrix.shape
    total = p.nnz
    cin = _even(total, n_c, rng) if equal_in else p.inputs()
    cap = _even(total, n_g, rng) if equal_out else p.fan_out().copy()
    out = np.zeros((n_g, n_c))
    for c in np.argsort(-cin, kind="stable"):
        chosen = np.lexsort((rng.random(n_g), -cap))[:cin[c]]
        if (cap[chosen] <= 0).any():
            raise RuntimeError("margins not realisable")
        out[chosen, c] = 1.0
        cap[chosen] -= 1
    name = f"margins_in-{'equal' if equal_in else 'kept'}_out-{'equal' if equal_out else 'kept'}"
    q = curveball(Projection(name, out, p.glomeruli), seed=seed + 1, sweeps=sweeps)
    return Projection(name, q.matrix, p.glomeruli)


def degenerate(p: Projection, seed: int = 0, n_used: int = 4) -> Projection:
    """Every cell samples from the same `n_used` glomeruli: a broken matrix."""
    rng = np.random.default_rng(seed)
    n_g, n_c = p.matrix.shape
    pool = rng.choice(n_g, n_used, replace=False)
    out = np.zeros_like(p.matrix)
    for c, k in enumerate(p.inputs()):
        k = min(max(int(k), 1), n_used)
        out[rng.choice(pool, k, replace=False), c] = 1.0
    return Projection("degenerate", out, p.glomeruli)


# --------------------------------------------------------------------------- odours

# DoOR 2.0 is fetched from a pinned commit so the input cannot drift.
DOOR_COMMIT = "db323a496577c4b4a72b5c2fcd1859e07521ffb5"
DOOR_BASE = f"https://raw.githubusercontent.com/ropensci/DoOR.data/{DOOR_COMMIT}/data"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest) -> None:
    import urllib.request
    from pathlib import Path
    dest = Path(dest)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


@dataclass
class Odours:
    """Retained DoOR responses: values, which entries were measured, labels."""
    x: np.ndarray                 # (odorants, glomeruli) after missing-data treatment
    observed: np.ndarray          # bool, True where DoOR has a measurement
    names: list[str]
    glomeruli: list[str]
    missing: str

    @property
    def missing_fraction(self) -> float:
        return float(1.0 - self.observed.mean())


def _impute(raw: np.ndarray, observed: np.ndarray, how: str, rank: int = 5,
            iters: int = 200) -> np.ndarray:
    x = np.where(observed, raw, 0.0)
    if how == "zero":
        return x
    if how == "glomerulus_mean":
        mu = np.nanmean(np.where(observed, raw, np.nan), axis=0)
        return np.where(observed, raw, np.nan_to_num(mu)[None, :])
    if how == "odour_mean":
        mu = np.nanmean(np.where(observed, raw, np.nan), axis=1)
        return np.where(observed, raw, np.nan_to_num(mu)[:, None])
    if how == "lowrank":
        # iterative rank-`rank` SVD imputation, observed entries held fixed
        z = np.where(observed, raw, np.nanmean(np.where(observed, raw, np.nan), axis=0))
        for _ in range(iters):
            u, s, vt = np.linalg.svd(z, full_matrices=False)
            approx = (u[:, :rank] * s[:rank]) @ vt[:rank]
            new = np.where(observed, raw, approx)
            if np.abs(new - z).max() < 1e-7:
                z = new
                break
            z = new
        return z
    raise ValueError(f"unknown missing-data treatment {how!r}")


def load_odours(cfg, glomeruli: list[str], min_glomeruli: int = 12,
                min_odours: int = 40, missing: str = "zero") -> Odours:
    """DoOR 2.0 consensus responses as an odorant x glomerulus matrix.

    DoOR integrates heterogeneous measurements and its consensus matrix is
    sparse. We keep glomeruli measured for at least `min_odours` odorants and
    odorants measured on at least `min_glomeruli` of those glomeruli. The
    remaining gaps are *unmeasured*, not known non-responses; `missing`
    chooses how they are filled ("zero" is the baseline, "glomerulus_mean",
    "odour_mean" and "lowrank" are sensitivity analyses). Negative values are
    clipped at zero and all-zero odorants dropped.
    """
    import pandas as pd

    raw_dir = cfg.raw_dir
    _fetch(f"{DOOR_BASE}/door_response_matrix.csv", raw_dir / "door_response_matrix.csv")
    _fetch(f"{DOOR_BASE}/door_mappings.csv", raw_dir / "door_mappings.csv")
    resp = pd.read_csv(raw_dir / "door_response_matrix.csv", sep=";")
    maps = pd.read_csv(raw_dir / "door_mappings.csv", sep=";")

    # The response matrix is keyed by receptor (Or10a, ...): join on receptor
    # and drop ambiguous glomerulus assignments such as "DL2d/v" or "DM5+DM3".
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
    m = m[m.columns[m.notna().sum(axis=0) >= min_odours]]
    m = m[m.notna().sum(axis=1) >= min_glomeruli]

    raw = m.to_numpy(float)
    observed = ~np.isnan(raw)
    x = np.clip(_impute(np.nan_to_num(raw), observed, missing), 0.0, None)
    keep = x.sum(axis=1) > 0
    return Odours(x[keep], observed[keep], list(m.index[keep]), list(m.columns), missing)


def mixtures(x: np.ndarray, n: int, seed: int = 0, lo: int = 2, hi: int = 5) -> np.ndarray:
    """Synthetic linear mixtures: each item is a Dirichlet(1)-weighted convex
    combination of `lo`..`hi` distinct rows of `x`.

    These are not measured mixture responses; olfactory mixture coding is not
    linear in general. They enlarge the benchmark while keeping the
    between-glomerulus correlations of the source profiles.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n, x.shape[1]))
    for i in range(n):
        k = int(rng.integers(lo, hi + 1))
        pick = rng.choice(len(x), k, replace=False)
        out[i] = rng.dirichlet(np.ones(k)) @ x[pick]
    return out


# --------------------------------------------------------------------------- the hash

def normalise(x: np.ndarray) -> np.ndarray:
    """Divisive normalisation to unit mean: removes concentration."""
    m = x.mean(axis=1, keepdims=True)
    return np.divide(x, m, out=np.zeros_like(x), where=m > 0)


def drive(x: np.ndarray, p: Projection, weighted: bool = False) -> np.ndarray:
    return normalise(x) @ (p.matrix if weighted else p.binary)


def tags(x: np.ndarray, p: Projection, k: int, weighted: bool = False,
         tie: str = "random", tie_seed: int = 0, y: np.ndarray | None = None) -> np.ndarray:
    """Winner-take-all tags with an explicit tie rule.

    Let y_(k) be the k-th largest drive of an item. Cells with y > y_(k) always
    win. Cells tied at y_(k) (cells with identical inputs have identical drive)
    are resolved by `tie`:

      "random"   exactly k winners; ties broken by a fixed random priority over
                 cells drawn from `tie_seed` (the baseline)
      "index"    exactly k winners; the lowest cell index wins
      "include"  every tied cell wins, so a tag may carry more than k ones
    """
    if y is None:
        y = drive(x, p, weighted)
    n, m = y.shape
    if k >= m:
        return np.ones((n, m), bool)
    kth = -np.partition(-y, k - 1, axis=1)[:, k - 1:k]      # y_(k) per row
    above = y > kth
    tied = y == kth
    if tie == "include":
        return above | tied
    need = k - above.sum(axis=1, keepdims=True)
    if tie == "index":
        prio = np.arange(m)
    elif tie == "random":
        prio = np.random.default_rng(tie_seed).permutation(m)
    else:
        raise ValueError(f"unknown tie rule {tie!r}")
    order = np.argsort(prio)                                  # cells by priority
    tied_sorted = tied[:, order]
    rank = np.cumsum(tied_sorted, axis=1)
    take_sorted = tied_sorted & (rank <= need)
    take = np.zeros_like(tied)
    take[:, order] = take_sorted
    return above | take


def gaussian_sign(x: np.ndarray, bits: int, seed: int = 0) -> np.ndarray:
    """Classical LSH baseline: signs of `bits` Gaussian projections of the
    normalised input (a `bits`-bit code)."""
    w = np.random.default_rng(seed).normal(size=(x.shape[1], bits))
    return (normalise(x) @ w) > 0


def storage_bits(m: int, k: int) -> int:
    """Bits of an optimal fixed-length code for any k-of-m subset:
    ceil(log2 C(m, k)), computed exactly."""
    c = math.comb(m, k)
    return (c - 1).bit_length() if c > 1 else 0


def computation_bits(p: Projection) -> int:
    """Gaussian code length with the same operation count per item as the fly
    expansion, counted as Dasgupta et al. (2017, Fig. 1C) count it: nnz(M)
    additions for the fly, d multiplications plus d additions per Gaussian
    projection. Normalisation, winner selection and thresholding are excluded
    from both sides."""
    return max(1, int(round(p.nnz / (2 * p.matrix.shape[0]))))


# --------------------------------------------------------------------------- scoring

def true_neighbours(x: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    """The k nearest items, excluding the item itself: the ground truth.

    "euclidean"   raw responses (concentration-sensitive; the baseline)
    "normalised"  Euclidean on divisively normalised responses, i.e. on what the
                  hash sees
    "angular"     cosine distance (scale-invariant)
    Exact distance ties are broken by item index.
    """
    if metric == "euclidean":
        z = x
    elif metric == "normalised":
        z = normalise(x)
    elif metric == "angular":
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        z = np.divide(x, nrm, out=np.zeros_like(x), where=nrm > 0)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    sq = (z ** 2).sum(1)
    d = sq[:, None] + sq[None, :] - 2 * (z @ z.T)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def retrieval_priority(n: int, seed: int = 0, rule: str = "random") -> np.ndarray:
    """Per-query priority over database items used to break Hamming ties.

    "random" gives every query its own seeded random order of the database;
    "index" orders by item index. The same matrix is used for every wiring on a
    benchmark, so ties are broken identically across the comparisons.
    """
    if rule == "index":
        return np.tile(np.arange(n, dtype=np.int32), (n, 1))
    if rule == "random":
        rng = np.random.default_rng(seed)
        return rng.permuted(np.tile(np.arange(n, dtype=np.int32), (n, 1)), axis=1)
    raise ValueError(f"unknown retrieval tie rule {rule!r}")


def mean_average_precision(tag: np.ndarray, truth: np.ndarray,
                           priority: np.ndarray | None = None,
                           per_item: bool = False):
    """Retrieval quality of a hash: mean over queries of average precision at
    kappa = truth.shape[1].

    Database items are ranked by Hamming distance to the query's tag, ties
    broken by `priority` (see `retrieval_priority`), which makes every ranking
    total and reproducible. Dense on purpose: Hamming distance is computed from
    popcounts, so codes with varying numbers of ones (Gaussian sign codes,
    "include" ties) are scored correctly.
    """
    n = tag.shape[0]
    kappa = truth.shape[1]
    if priority is None:
        priority = retrieval_priority(n, 0, "random")
    t = tag.astype(np.float32)
    ones = t.sum(1)
    ham = np.rint(ones[:, None] + ones[None, :] - 2 * (t @ t.T)).astype(np.int64)
    key = ham * n + priority                                  # unique per row
    np.fill_diagonal(key, np.iinfo(np.int64).max)
    top = np.argpartition(key, kappa - 1, axis=1)[:, :kappa]
    top = np.take_along_axis(top, np.argsort(np.take_along_axis(key, top, 1), axis=1), 1)

    want = np.zeros((n, n), bool)
    np.put_along_axis(want, truth, True, axis=1)
    hit = np.take_along_axis(want, top, axis=1)
    prec = np.where(hit, np.cumsum(hit, axis=1) / np.arange(1, kappa + 1), 0.0)
    per_query = prec.sum(axis=1) / kappa
    return per_query if per_item else float(per_query.mean())
