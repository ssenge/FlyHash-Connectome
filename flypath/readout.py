"""E2: graded-query readout of stored winner sets (pre-registered).

Pre-registration: experiments/PREREGISTRATION_2026-09-28.md, section "E2"
(identical to UPDATES.md section 14), frozen before this module was written.
This module implements that section and nothing else; the criteria are
copied, not changed. Results go to results/e2_readout.json.

Design
------
Protocol (the 2017 protocol of `flypath.replication`). Per dataset the fixed
10,000-item subset of `replication.load_benchmark`; 50 trials, each drawing
1,000 queries from the items; truth = the 200 (2%) nearest items by Euclidean
distance on the raw features, query excluded (`replication.truth`); AP@200
with all 200 true neighbours in the denominator and recall@200
(`replication.average_precision`, `replication.recall`). Datasets sift,
glove, mnist, odours; k in {2, 4, 8, 16, 32}. Every code sees the row-centred
input (`replication.centre`). Each trial draws one random fly matrix with
m = 10d Kenyon cells, each sampling s = round(0.1d) distinct inputs
(`replication.fly_matrix`).

The fly code. Store the k winner indices of the raw drive y = x_c M
(`replication.winners`), nothing else.
  fly_sym         the paper's scorer: overlap of the two tags (= Hamming,
                  `replication.rank_binary`).
  fly_asym_raw    score(q, j) = sum over j's k winners c of y_q[c]: the
                  query's graded drive gathered at the item's winners
                  (k lookups per comparison).
  fly_asym_std    the same with the per-cell-standardised query drive
                  (y_q[c] - mean_c) / sd_c. The stored codes are unchanged.
Which of raw / std is "fly asymmetric" is chosen per dataset on the
validation split (below), never on test.

Storage accounting. Primary ("info"): B(k) = ceil(log2 C(m, k)) bits, exact
(`flyhash.storage_bits`, math.comb). Secondary, reported ("fixed"):
k * ceil(log2 m) bits.

Training-free competitors at B bits, each with a symmetric and an asymmetric
(graded query . stored code) scorer; the asymmetric query is raw or per-cell
standardised, chosen on the validation split as the fly's is (item 5):
  gauss      B Gaussian projections g = x_c R, R ~ N(0, 1)^(d x B); bit g > 0.
             sym: Hamming. asym: g_q . (2b_j - 1) (Gordo et al. 2014,
             Dong et al. 2008), which ranks as g_q . b_j does.
  densefly   B cells, each summing s = round(0.1d) distinct inputs
             (`replication.fly_matrix(d, B)`); bit = drive >= 0 (Sharma and
             Navlakha 2018). sym: Hamming. asym: drive_q . (2b_j - 1).
  vtag       FlyLSH value tag: the k' = floor(B / (ceil(log2 m) + 4)) winners
             of the trial's fly drive, each with its drive value quantised to
             4 bits. sym: Euclidean distance between the two dequantised
             sparse vectors (the 2017 rule, "Euclidean distance between
             hashes"). asym: y_q . v_j with v_j the dequantised stored vector
             (the pre-registered "graded query . stored code" form), and the
             Euclidean ADC form -||y_q - v_j||^2 (item 5).
  fly_sym    the symmetric fly overlap, which stores exactly the fly's code.
Trained upper reference, reported and not scored:
  pq         PQ-ADC at B bits: product quantiser fitted by k-means per
             subspace (scipy.cluster.vq.kmeans2) on the stored items only;
             asymmetric distance from the uncompressed query to each item's
             reconstruction.

Validation split (chooses one thing only: raw vs std query drive, per
dataset). 2,000 items disjoint from the test items of every trial (the test
items are the same fixed subset in every trial), 200 queries drawn from them
per validation trial, truth and scoring as on test. The same raw-vs-std rule
is applied to every asymmetric scorer (item 5 below), so that every
competitor gets the same asymmetric readout as the fly.

Pass criteria (primary, both required), unchanged from the pre-registration:
  (a) GloVe: the ratio AP(fly asymmetric) / AP(best training-free
      competitor), under the info accounting, has a trial-bootstrap 95% lower
      bound >= 1.0 for >= 4 of the 5 values of k.
  (b) MNIST: the same at k = 2 and k = 4.
Secondary (paper robustness):
  (c) MaleCNS R, k in {4, 16}: the connectome-null and even-fan-out-null gaps
      under the asymmetric readout lie inside the symmetric readout's 95%
      interval on >= 3 of 4 datasets. MaleCNS R = `experiments.context(cfg)`:
      `.full` (51 glomeruli) for sift, glove, mnist, reduced by PCA to one
      component per glomerulus with a fresh random component-glomerulus
      permutation per trial (as `replication.score_wiring`); `.proj` for
      odours. The design is the paper's own MaleCNS R control analysis
      (`replication.trial_controls`, results/controls.json): 20 trials, the
      first 10 curveball nulls of `experiments.null_pool`, one even-fan-out
      control `flyhash.margin_control(equal_out=True)` per trial. The
      symmetric readout therefore reproduces the paper's per-trial values and
      intervals exactly (checked in the analysis, "paper_check").

Unspecified details fixed before running
----------------------------------------
Fixed on 2026-09-28, before any full-scale result existed; only a smoke run
on a reduced configuration (separate output file) was executed.

 1. Validation items. mnist: 2,000 of the 50,000 training images not in the
    test subset (the subset is re-derived exactly as `load_benchmark` draws
    it and checked against it). glove: 2,000 of the 390,000 words not in the
    test subset (same re-derivation and check). sift: 2,000 vectors of
    siftsmall_learn.fvecs, a separate set of SIFT descriptors (the test
    subset is all of siftsmall_base). odours: 2,000 new Dirichlet mixtures of
    the same DoOR profiles, `flyhash.mixtures(seed=40000)` (test: seed 0).
    Rows identical to any test row are excluded (none were found for sift;
    the count is recorded per dataset). Drawn with seed 40000.
 2. Validation protocol: 10 validation trials, each with fresh matrices for
    every code (all codes of the test protocol except PQ, both accountings)
    and 200 fresh queries from the 2,000 items; truth = top 2% = 40
    neighbours (the protocol's 2% of the database); AP@40. Nothing else is
    taken from the validation split.
 3. Selection rule: "std" iff its AP averaged over the validation trials and
    all five k is strictly higher than that of "raw"; otherwise "raw". One
    choice per dataset for the fly, recorded before the dataset's first test
    trial. The same rule, on the same validation trials, gives one choice per
    dataset for each competitor's asymmetric scorer (gauss, densefly, vtag;
    per accounting, because the budget changes the code); see item 5.
 4. Per-cell standardisation: mean and population SD of each cell's drive
    over the stored items of the database being searched (test: the n items
    of the trial; validation: the 2,000 validation items), queries included
    because they are database items in this protocol. A cell with SD 0
    contributes 0. The 2m statistics are per database, like the PQ
    codebooks, and are not counted in the per-item budget.
 5. Competitor asymmetric scorers. Every competitor gets the same asymmetric
    readout as the fly: graded query . stored code, with the query either raw
    or per-cell standardised (statistics over the database, as in item 4),
    the variant chosen on the validation split by the rule of item 3 (its own
    choice per competitor, dataset and accounting). The forms are
    Gordo-style g_q . (2b_j - 1) for the sign codes and y_q . v_j for the
    value tag. For the value tag the primary set also contains the Euclidean
    ADC form -||y_q - v_j||^2 (Jegou et al. 2011, to the zero-filled sparse
    reconstruction): for the fly and the +-1 sign codes the stored norm is
    constant, so the inner product and the Euclidean ADC rank identically
    and are the same trick; the value tag's stored norm varies, so both
    readings of "the same trick" are included. Reported, entering no
    criterion: "raw_asym", the competitors' raw-query forms only (the
    pilot's comparison), and "all_variants", every training-free competitor
    scorer computed (both query variants of every asymmetric competitor plus
    the ADC form). In every table the maximum over competitors is re-taken
    in every bootstrap resample.
 6. Ranking: by descending score (ascending distance). Exact ties are broken
    uniformly at random: Hamming ties as in `replication.rank_binary`; for
    real-valued scores every distinct stored code is scored once, so items
    with identical codes tie exactly, and tied items are ordered by an
    independent uniform priority. The query itself is excluded. The drive is
    computed in float32 on the row-centred input, as in `replication`, so
    scores of different codes that are equal in exact arithmetic (possible
    for integer-valued SIFT and MNIST features) can differ in the last bits;
    such near-ties are ordered by those rounding differences, which do not
    depend on the true neighbours.
 7. Winner ties in drive are resolved as in the paper's code
    (`replication.winners`, argpartition order).
 8. Value tag: winners of the trial's own m = 10d fly drive (the matrix the
    fly scorers use), values = raw drive; one uniform 16-level quantiser on
    [min, max] of all stored winner values of the database, reconstruction
    at the bin centres (2 floats per database, not counted per item). k' is
    at least 1 at every budget of the protocol (checked in the tests).
 9. Gaussian sign code threshold g > 0 (as `replication`'s sign LSH);
    DenseFly threshold >= 0 (as in the DenseFly paper).
10. PQ: M = min(ceil(B / 8), d) subquantisers over contiguous, near-equal
    chunks of the row-centred input dimensions (`numpy.array_split`), the B
    bits spread as evenly as possible (floor(B/M) or ceil(B/M) per
    subquantiser, total exactly B); kmeans2 with minit="points" (random data
    points, the FAISS default; k-means++ in scipy costs ~20x more) and 25
    iterations (the FAISS default), refitted in every trial on the n stored
    items; ADC = squared Euclidean distance between the row-centred query and
    the reconstruction. PQ is computed under both accountings.
11. Estimand of the ratio: mean over trials of fly-asymmetric AP divided by
    the largest mean-over-trials AP among the competitors (ratio of
    aggregate means, as `replication.contrast`); 95% percentile bootstrap
    over trials (2,000 draws), the maximum over competitors re-taken in
    every resample; "lower bound" = the 2.5th percentile. Competitor set of
    the criterion ("primary"): fly_sym; the sym scorer and the validated
    asym scorer (item 5) of gauss, densefly and vtag; and vtag's ADC form;
    all at B(k). A pre-registered competitor that is undefined at some k
    (k' = 0; impossible at the protocol's budgets) is listed in the row and
    makes the criteria report all_competitors_present = false.
12. Criterion (c) repeats the paper's MaleCNS R control analysis
    (`replication.trial_controls`, results/controls.json) exactly: 20
    trials; trial seeds, component permutations, queries, Hamming tie draws
    (the draws of the sizes the paper also scored, k = 2 and 8, are drawn and
    discarded) and the even-fan-out draws (the second draw of seed 80000 + t)
    as there; the first 10 nulls of the cached pools (null_pool(proj, 200)
    for odours, null_pool(full binary, 50) otherwise; the first 10 draws of
    any pool are the paper's 10 matrices, checked). The symmetric readout
    thus reproduces the paper's per-trial AP, and the analysis records the
    largest difference from results/controls.json ("paper_check"). The
    asymmetric readout is the dataset's validated fly choice from item 3
    (the other variant is reported). Standardisation statistics per matrix
    over the 10,000 stored items. Gap = 100 * (mean over trials of control
    AP / mean over trials of null-mean AP - 1), 95% trial bootstrap
    (`replication.contrast`, 2,000 draws, seed = index of k as in the
    paper). "Lies inside" = the asymmetric point estimate is within the
    symmetric 95% interval. A dataset passes when all four of its gaps
    (connectome and even fan-out, k = 4 and 16) lie inside; (c) passes on
    >= 3 of 4 datasets.
13. Seeds: test trial t uses SeedSequence entropy (20000, t, ...);
    validation trial v uses (30000, v, ...); every method and k has its own
    stream, so results do not depend on which other methods run.
14. recall@200 is recorded for every method; it enters no criterion.

Code review before the full run (2026-09-28; no full-scale result existed)
changed three of the details above:
  - item 5: the competitors' asymmetric scorers get the validated raw/std
    query variant, as the fly does, and the value tag's ADC form is in the
    primary set. Before, the primary set held only the competitors'
    raw-query forms, while the fly's query variant was tuned on validation.
    That comparison is still reported ("raw_asym").
  - item 12: 20 trials and the paper's first 10 nulls, with the paper's tie
    draws, replace 10 trials and 20 nulls, so that "the symmetric readout's
    95% interval" is the interval the paper reports.
  - item 6: corrected the statement on exact ties of integer-valued drives.
Validation now covers both accountings, because the fixed-width competitors
need their own choice.

Deviations
----------
None.

Results layout (results/e2_readout.json)
----------------------------------------
settings, code_sha256, preregistration_sha256;
datasets/<name>: d, m, s, bits, value_tag_winners, pq_subquantisers,
  validation {source, trials/<v>/{ap, recall}, mean_ap, choice, chosen_at},
  trials/<t>/{ap/<method>: [AP per k], recall/<method>: [...], seconds};
connectome/<name>: trials/<t>/{real, null_mean, out_equal: {readout: [per k]},
  null: {readout: [[per k] per null]}};
analysis (written once everything is complete): per dataset and accounting
  the "primary" (criterion), "raw_asym" and "all_variants" ratio tables and
  the PQ reference; criterion (c) tables with the paper_check; criteria
  {a_glove, b_mnist, primary_pass, c_connectome, all_competitors_present,
  protocol_as_preregistered}.
Method names: fly_sym, fly_asym_{raw,std}; {gauss,densefly,vtag}_{info,fixed}_
  {sym,asym_raw,asym_std}; vtag_{info,fixed}_adc; pq_{info,fixed}_adc.
Validation choice keys: fly, {gauss,densefly,vtag}_{info,fixed}.

Running
-------
  VECLIB_MAXIMUM_THREADS=5 PYTHONPATH=. python -m flypath.readout --max-minutes 8
resumes from results/e2_readout.json and exits with code 75 while work is
left (0 when complete); repeat until it exits 0. The result file records the
settings and this module's SHA-256 and refuses to resume under either a
different setting or different code. About 35-45 s per test trial and dataset
and about 20 s per connectome trial (12 matrices; Apple silicon, 5 BLAS
threads): about 2.5 h in total.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import flyhash as fh
from . import replication as rp
from .config import ROOT, Config

KS = tuple(rp.HASH_LENGTHS)          # (2, 4, 8, 16, 32)
DATASETS = tuple(rp.DATASETS)        # sift, glove, mnist, odours
TRIALS = rp.TRIALS                   # 50
N_ITEMS = rp.N_DATA                  # 10,000
N_QUERIES = rp.N_QUERIES             # 1,000
EXPANSION = 10                       # m = 10 d
VALUE_BITS = 4
VAL_ITEMS, VAL_QUERIES, VAL_TRIALS = 2000, 200, 10
PQ_SUB_BITS, PQ_ITER = 8, 25
CONN_KS, CONN_TRIALS, CONN_NULLS = (4, 16), 20, 10   # the paper's MaleCNS R controls
BOOT_DRAWS = 2000
SEED_TEST, SEED_VAL, SEED_VALSET, SEED_BOOT = 20_000, 30_000, 40_000, 50_000
OUT_NAME = "e2_readout.json"
EXIT_MORE = 75                       # exit code: time budget used, run again
LOAD_MARGIN = 120.0                  # seconds kept free for loading a dataset

FAMILY = {"fly": 1, "gauss": 2, "densefly": 3, "vtag": 4, "pq": 5}
ACCOUNTING = {"info": 1, "fixed": 2}
VARIANTS = ("raw", "std")
READOUTS = ("sym", "asym_raw", "asym_std")
ASYM_FAMILIES = ("gauss", "densefly", "vtag")


# ---------------------------------------------------------------- budgets

def index_bits(m: int) -> int:
    """ceil(log2 m): bits of one cell index."""
    return (int(m) - 1).bit_length()


def info_bits(m: int, k: int) -> int:
    """ceil(log2 C(m, k)), exact (math.comb)."""
    return fh.storage_bits(int(m), int(k))


def fixed_bits(m: int, k: int) -> int:
    """k indices of ceil(log2 m) bits each."""
    return int(k) * index_bits(m)


def budget(accounting: str, m: int, k: int) -> int:
    return info_bits(m, k) if accounting == "info" else fixed_bits(m, k)


def value_tag_winners(bits: int, m: int, value_bits: int = VALUE_BITS) -> int:
    """k' = floor(B / (ceil(log2 m) + value_bits))."""
    return int(bits) // (index_bits(m) + value_bits)


def pq_layout(bits: int, d: int) -> tuple[list[int], list[np.ndarray]]:
    """Bits per subquantiser and the dimensions of each subspace."""
    M = min(-(-int(bits) // PQ_SUB_BITS), int(d))
    per = [bits // M + (1 if s < bits % M else 0) for s in range(M)]
    return per, np.array_split(np.arange(d), M)


# ---------------------------------------------------------------- helpers

def _rng(*key) -> np.random.Generator:
    return np.random.default_rng([int(k) for k in key])


def cell_stats(y: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population SD of every column, accumulated in float64."""
    n = len(y)
    mu = np.zeros(y.shape[1])
    for i in range(0, n, chunk):
        mu += y[i:i + chunk].sum(0, dtype=np.float64)
    mu /= n
    ss = np.zeros(y.shape[1])
    for i in range(0, n, chunk):
        dlt = y[i:i + chunk].astype(np.float64) - mu
        ss += np.einsum("ij,ij->j", dlt, dlt)
    return mu, np.sqrt(ss / n)


def standardise(u: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(u - mu) / sd per column; columns with sd == 0 become 0."""
    u = np.asarray(u, np.float64)
    return np.divide(u - mu, sd, out=np.zeros_like(u), where=sd > 0)


def _unique_rows(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the first row of every distinct code, and each row's code."""
    _, first, inv = np.unique(codes, axis=0, return_index=True, return_inverse=True)
    return first, np.asarray(inv).reshape(-1)


def _sparse_rows(idx: np.ndarray, vals: np.ndarray, m: int) -> sp.csr_matrix:
    n, k = idx.shape
    return sp.csr_matrix((np.asarray(vals, np.float64).ravel(), idx.ravel(),
                          np.arange(0, n * k + 1, k)), shape=(n, m))


# ---------------------------------------------------------------- scorers

def set_scores(idx: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Fly asymmetric score: (n_q, n) sum of each query vector u_q over each
    item's stored winner set idx[j] (rows sorted). Distinct codes are scored
    once, so identical codes tie exactly."""
    first, inv = _unique_rows(idx)
    a = _sparse_rows(idx[first], np.ones(idx[first].shape), u.shape[1])
    return np.asarray(a @ np.asarray(u, np.float64).T).T[:, inv]


def sign_scores(bits: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Asymmetric score of a binary code: u_q . (2 b_j - 1)."""
    first, inv = _unique_rows(bits)
    return (np.asarray(u, np.float64) @ (2.0 * bits[first] - 1.0).T)[:, inv]


def quantise(vals: np.ndarray, bits: int = VALUE_BITS) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Uniform scalar quantiser with 2**bits levels on [min, max] of `vals`;
    returns the integer levels, their bin-centre reconstructions and the range."""
    vals = np.asarray(vals, np.float64)
    lo, hi = float(vals.min()), float(vals.max())
    levels = 2 ** bits
    if not hi > lo:
        return np.zeros(vals.shape, np.int64), np.full(vals.shape, lo), [lo, hi]
    w = (hi - lo) / levels
    lev = np.clip(np.floor((vals - lo) / w), 0, levels - 1).astype(np.int64)
    return lev, lo + (lev + 0.5) * w, [lo, hi]


def value_tag(y: np.ndarray, kp: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FlyLSH value tag: the kp winners of each row of y (sorted indices),
    their quantised levels and dequantised values."""
    t = rp.winners(y, kp)
    t.sort_indices()
    idx = t.indices.reshape(len(y), kp).astype(np.int64)
    vals = np.take_along_axis(y, idx, 1)
    lev, rec, _ = quantise(vals)
    return idx, lev, rec


def value_scores(idx: np.ndarray, lev: np.ndarray, rec: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Asymmetric value-tag score: u_q . v_j (v_j the dequantised stored vector)."""
    first, inv = _unique_rows(np.hstack([idx, lev]))
    a = _sparse_rows(idx[first], rec[first], u.shape[1])
    return np.asarray(a @ np.asarray(u, np.float64).T).T[:, inv]


def value_adc(idx: np.ndarray, lev: np.ndarray, rec: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Euclidean ADC form of the value tag: ||u_q - v_j||^2 with v_j the
    dequantised stored vector, zero outside its k' winners."""
    u = np.asarray(u, np.float64)
    first, inv = _unique_rows(np.hstack([idx, lev]))
    a = _sparse_rows(idx[first], rec[first], u.shape[1])
    d = (u * u).sum(1)[:, None] - 2.0 * np.asarray(a @ u.T).T + (rec[first] ** 2).sum(1)[None, :]
    return d[:, inv]


def value_distances(idx: np.ndarray, lev: np.ndarray, rec: np.ndarray, queries: np.ndarray,
                    m: int) -> np.ndarray:
    """Symmetric value-tag distance: squared Euclidean distance between the
    dequantised sparse vectors of query and item."""
    first, inv = _unique_rows(np.hstack([idx, lev]))
    a = _sparse_rows(idx[first], rec[first], m)
    b = _sparse_rows(idx[queries], rec[queries], m)
    na = (rec[first] ** 2).sum(1)
    nb = (rec[queries] ** 2).sum(1)
    return (nb[:, None] + na[None, :] - 2.0 * (b @ a.T).toarray())[:, inv]


def pq_fit(x: np.ndarray, bits: int, rng: np.random.Generator,
           iters: int = PQ_ITER) -> tuple[np.ndarray, list[np.ndarray]]:
    """Product quantiser with `bits` bits in total (see `pq_layout`), k-means
    per subspace with scipy's kmeans2, fitted on the rows of x."""
    from scipy.cluster.vq import kmeans2
    n, d = x.shape
    per, chunks = pq_layout(bits, d)
    codes = np.empty((n, len(per)), np.int64)
    books = []
    for s, (b, cols) in enumerate(zip(per, chunks)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cent, lab = kmeans2(np.ascontiguousarray(x[:, cols]), min(2 ** b, n), iter=iters,
                                minit="points", rng=rng)
        books.append(np.asarray(cent, np.float64).reshape(len(cent), len(cols)))
        codes[:, s] = lab
    return codes, books


def pq_adc(xq: np.ndarray, codes: np.ndarray, books: list[np.ndarray]) -> np.ndarray:
    """(n_q, n) squared distance from each uncompressed query to each item's
    reconstruction (subspaces are contiguous and in order)."""
    first, inv = _unique_rows(codes)
    rec = np.hstack([books[s][codes[first, s]] for s in range(len(books))])
    return rp._sqdist(np.asarray(xq, np.float64), rec)[:, inv]


# ---------------------------------------------------------------- ranking

def rank_scores(scores: np.ndarray, queries: np.ndarray, top: int,
                rng: np.random.Generator) -> np.ndarray:
    """Indices of the `top` highest scores per query, best first, the query
    itself excluded; exact ties in random order."""
    s = np.array(scores, dtype=np.float64)
    if np.isnan(s).any():
        raise ValueError("NaN score")
    nq, n = s.shape
    s[np.arange(nq), queries] = -np.inf
    part = np.argpartition(s, n - top, axis=1)
    kth = np.take_along_axis(s, part[:, n - top:n - top + 1], 1)   # top-th largest
    count = (s >= kth).sum(1)
    out = np.empty((nq, top), np.int64)

    def order(rows, cand):            # best first, exact ties by a random priority
        cs = s[rows[:, None], cand]
        o = np.lexsort((rng.random(cand.shape), -cs), axis=-1)[:, :top]
        out[rows] = np.take_along_axis(cand, o, 1)

    plain = np.flatnonzero(count == top)          # no tie across the cut-off
    if len(plain):
        order(plain, part[plain, n - top:])
    wide = np.flatnonzero(count > top)            # widen to every item >= the cut-off
    if len(wide):
        w = int(count[wide].max())
        cand = (np.argpartition(s[wide], n - w, axis=1)[:, n - w:] if w < n
                else np.tile(np.arange(n), (len(wide), 1)))
        order(wide, cand)
    return out


def rank_hamming(bits: np.ndarray, queries: np.ndarray, top: int,
                 rng: np.random.Generator) -> np.ndarray:
    """`replication.rank_binary` for a dense boolean code (the same Hamming
    distances and the same random tie-breaking draws)."""
    b = np.asarray(bits, np.float32)
    ones = b.sum(1)
    dist = ones[queries][:, None] + ones[None, :] - 2.0 * (b[queries] @ b.T)
    return rp._nearest(np.rint(dist), queries, top, rng)


# ---------------------------------------------------------------- one trial

def codes_trial(x: np.ndarray, n_queries: int, ks, key: tuple, accountings=("info", "fixed"),
                pq: bool = True) -> dict:
    """Every code and scorer on one database `x` (raw features) with one
    draw of queries and matrices. Returns {"ap": {method: [per k]},
    "recall": {...}}; entries are None where a method is undefined."""
    n, d = x.shape
    m = EXPANSION * d
    top = int(rp.TOP * n)
    xc = rp.centre(x)
    x32 = xc.astype(np.float32)
    r0 = _rng(*key, 0)
    q = r0.choice(n, n_queries, replace=False)
    true = rp.truth(x, q, top)
    y = x32 @ rp.fly_matrix(d, m, r0)
    mu, sd = cell_stats(y)
    yq = y[q].astype(np.float64)
    drive_q = {"raw": yq, "std": standardise(yq, mu, sd)}
    ap: dict[str, list] = {}
    rc: dict[str, list] = {}

    def put(name, i, pred):
        ap.setdefault(name, [None] * len(ks))[i] = rp.average_precision(pred, true)
        rc.setdefault(name, [None] * len(ks))[i] = rp.recall(pred, true)

    def sign_family(name, i, proj, bits, skey):
        put(f"{name}_sym", i, rank_hamming(bits, q, top, _rng(*skey, 1)))
        pmu, psd = cell_stats(proj)
        uq = proj[q].astype(np.float64)
        for v, var in enumerate(VARIANTS):
            u = uq if var == "raw" else standardise(uq, pmu, psd)
            put(f"{name}_asym_{var}", i, rank_scores(sign_scores(bits, u), q, top, _rng(*skey, 2 + v)))

    for i, k in enumerate(ks):
        fkey = (*key, k, FAMILY["fly"], 0)
        t = rp.winners(y, k)
        t.sort_indices()
        put("fly_sym", i, rp.rank_binary(t, q, top, _rng(*fkey, 1)))
        idx = t.indices.reshape(n, k)
        for v, var in enumerate(VARIANTS):
            put(f"fly_asym_{var}", i, rank_scores(set_scores(idx, drive_q[var]), q, top, _rng(*fkey, 2 + v)))
        for acc in accountings:
            a = ACCOUNTING[acc]
            B = budget(acc, m, k)
            gkey = (*key, k, FAMILY["gauss"], a)
            g = xc @ _rng(*gkey, 0).normal(size=(d, B))
            sign_family(f"gauss_{acc}", i, g, g > 0, gkey)
            dkey = (*key, k, FAMILY["densefly"], a)
            yd = (x32 @ rp.fly_matrix(d, B, _rng(*dkey, 0))).astype(np.float64)
            sign_family(f"densefly_{acc}", i, yd, yd >= 0, dkey)
            vkey = (*key, k, FAMILY["vtag"], a)
            kp = value_tag_winners(B, m)
            if kp >= 1:
                vi, lev, rec = value_tag(y, kp)
                put(f"vtag_{acc}_sym", i, rank_scores(-value_distances(vi, lev, rec, q, m), q, top,
                                                      _rng(*vkey, 1)))
                for v, var in enumerate(VARIANTS):
                    put(f"vtag_{acc}_asym_{var}", i,
                        rank_scores(value_scores(vi, lev, rec, drive_q[var]), q, top, _rng(*vkey, 2 + v)))
                put(f"vtag_{acc}_adc", i, rank_scores(-value_adc(vi, lev, rec, yq), q, top, _rng(*vkey, 4)))
            else:
                for nm in (f"vtag_{acc}_sym", f"vtag_{acc}_adc", *(f"vtag_{acc}_asym_{v}" for v in VARIANTS)):
                    ap.setdefault(nm, [None] * len(ks))
                    rc.setdefault(nm, [None] * len(ks))
            if pq:
                pkey = (*key, k, FAMILY["pq"], a)
                codes, books = pq_fit(xc, B, _rng(*pkey, 0))
                put(f"pq_{acc}_adc", i, rank_scores(-pq_adc(xc[q], codes, books), q, top, _rng(*pkey, 1)))
    return {"ap": ap, "recall": rc}


def asym_families(names) -> list[str]:
    """The asymmetric scorers with a raw/std query variant among `names`:
    "fly" and "<family>_<accounting>" for every competitor family computed."""
    return ["fly", *(f"{f}_{a}" for f in ASYM_FAMILIES for a in ACCOUNTING
                     if f"{f}_{a}_asym_raw" in names)]


def choose(val_trials: dict, families=None) -> tuple[dict, dict]:
    """Validation rule, the same for the fly and every competitor's
    asymmetric scorer: "std" iff its AP averaged over validation trials and
    all k is strictly higher than that of "raw"."""
    if families is None:
        families = asym_families(next(iter(val_trials.values()))["ap"])
    choice, means = {}, {}
    for fam in families:
        pre = "fly_asym" if fam == "fly" else f"{fam}_asym"
        mm = {}
        for var in VARIANTS:
            vals = np.array([[np.nan if v is None else v for v in tr["ap"][f"{pre}_{var}"]]
                             for tr in val_trials.values()], float)
            mm[var] = float(np.nanmean(vals)) if np.isfinite(vals).any() else None
        means[fam] = mm
        choice[fam] = "std" if (mm["std"] is not None and mm["raw"] is not None
                                and mm["std"] > mm["raw"]) else "raw"
    return choice, means


# ---------------------------------------------------------------- validation data

def _row_keys(x: np.ndarray) -> set:
    return {hashlib.sha1(np.ascontiguousarray(r).tobytes()).digest() for r in x}


def _fresh(cand: np.ndarray, test: np.ndarray, n_val: int) -> tuple[np.ndarray, int]:
    """First n_val rows of `cand` equal to no test row and to no earlier pick."""
    seen = _row_keys(test)
    keep, dropped = [], 0
    for i, r in enumerate(cand):
        h = hashlib.sha1(np.ascontiguousarray(r).tobytes()).digest()
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        keep.append(i)
        if len(keep) == n_val:
            break
    if len(keep) < n_val:
        raise RuntimeError("not enough validation candidates")
    return cand[keep], dropped


def validation_set(cfg: Config, name: str, n_val: int, n_test: int,
                   test: np.ndarray) -> tuple[np.ndarray, dict]:
    """Validation items, disjoint from the test items of every trial (the
    fixed subset `load_benchmark(cfg, name, n_test)`, passed as `test`)."""
    d = cfg.raw_dir / "bench"
    rng = np.random.default_rng(SEED_VALSET)
    spare = 2 * n_val + 100
    if name == "mnist":
        allx = np.load(d / "mnist.npz")["x_train"].reshape(-1, 784).astype(float)
        tid = np.random.default_rng(0).choice(len(allx), n_test, replace=False)
        rebuilt = allx[tid]
        pool = rng.permutation(np.setdiff1d(np.arange(len(allx)), tid))[:spare]
        cand = allx[pool]
        source = (f"MNIST training images outside the test subset "
                  f"({len(allx) - n_test} candidates), seed {SEED_VALSET}")
    elif name == "sift":
        def fvecs(f):
            return np.fromfile(d / "siftsmall" / f, dtype=np.float32).reshape(-1, 129)[:, 1:].astype(float)
        base = fvecs("siftsmall_base.fvecs")
        rebuilt = base[np.random.default_rng(0).choice(len(base), n_test, replace=False)] \
            if len(base) > n_test else base
        learn = fvecs("siftsmall_learn.fvecs")
        cand = learn[rng.permutation(len(learn))[:spare]]
        source = f"siftsmall_learn.fvecs ({len(learn)} vectors), seed {SEED_VALSET}"
    elif name == "glove":
        with zipfile.ZipFile(d / "glove.6B.zip") as z:
            lines = z.read("glove.6B.300d.txt").decode("utf8").splitlines()
        tid = np.sort(np.random.default_rng(0).choice(len(lines), n_test, replace=False))
        rebuilt = np.array([[float(v) for v in lines[i].split(" ")[1:]] for i in tid])
        pool = rng.permutation(np.setdiff1d(np.arange(len(lines)), tid))[:spare]
        cand = np.array([[float(v) for v in lines[i].split(" ")[1:]] for i in pool])
        source = (f"GloVe 6B 300d words outside the test subset "
                  f"({len(lines) - n_test} candidates), seed {SEED_VALSET}")
    elif name == "odours":
        from .experiments import context
        src = context(cfg).odours.x
        rebuilt = fh.mixtures(src, n_test, seed=0)
        cand = fh.mixtures(src, spare, seed=SEED_VALSET)
        source = (f"fresh Dirichlet mixtures of the same DoOR profiles "
                  f"(flyhash.mixtures seed {SEED_VALSET}; test items: seed 0)")
    else:
        raise ValueError(f"unknown dataset {name!r}")
    if not np.array_equal(rebuilt, test):
        raise RuntimeError(f"{name}: could not reproduce the test subset of load_benchmark")
    if name == "sift":
        test = np.vstack([test, base])          # exclude every base vector, not only the subset
    xv, dropped = _fresh(cand, test, n_val)
    return xv, {"source": source, "excluded_duplicates": int(dropped),
                "test_subset_reproduced": True, "n_items": int(len(xv))}


# ---------------------------------------------------------------- criterion (c)

def connectome_trial(name: str, z: np.ndarray, wiring: np.ndarray, nulls: list[np.ndarray],
                     pb: fh.Projection, t: int, ks, n_queries: int) -> dict:
    """One trial of the MaleCNS R comparison (seeds, queries, permutations,
    Hamming tie draws and control draws as `replication.trial_controls`, so
    the symmetric readout reproduces the paper's values): symmetric and both
    asymmetric readouts for the connectome, every null and one even-fan-out
    control."""
    g, m = wiring.shape
    n = len(z)
    top = int(rp.TOP * n)
    rng = np.random.default_rng(5000 + t)
    zp = z if name == "odours" else z[:, rng.permutation(g)]
    z32 = rp.centre(zp).astype(np.float32)
    q = rng.choice(n, n_queries, replace=False)
    true = rp.truth(zp, q, top)
    if list(ks) != sorted(set(ks)):
        raise ValueError("ks must be increasing")
    tie_seed = int(rng.integers(2**31))
    crng = np.random.default_rng(80_000 + t)
    # the paper draws one seed per control in CONTROLS order (in, out, both)
    cseed = {c: int(crng.integers(2**31)) for c in rp.CONTROLS}
    # the paper scores every size in this order with one tie stream per matrix;
    # the draws of the sizes not scored here are drawn and discarded, so that
    # the symmetric readout reproduces the paper's per-trial values exactly
    sizes = sorted(set(rp.HASH_LENGTHS) | {int(round(0.05 * m))} | set(ks))

    def fly(w):
        y = z32 @ w
        mu, sd = cell_stats(y)
        yq = y[q].astype(np.float64)
        u = {"raw": yq, "std": standardise(yq, mu, sd)}
        tie = np.random.default_rng(tie_seed)
        out = {r: [] for r in READOUTS}
        for k in sizes:
            if k > max(ks):
                break
            if k not in ks:
                tie.random((len(q), n))          # the draw rank_binary spends on size k
                continue
            tg = rp.winners(y, k)
            tg.sort_indices()
            out["sym"].append(rp.average_precision(rp.rank_binary(tg, q, top, tie), true))
            idx = tg.indices.reshape(n, k)
            for v, var in enumerate(VARIANTS):
                pred = rank_scores(set_scores(idx, u[var]), q, top, _rng(tie_seed, k, 1 + v))
                out[f"asym_{var}"].append(rp.average_precision(pred, true))
        return out

    real = fly(wiring)
    each = [fly(w) for w in nulls]
    even = fh.margin_control(pb, seed=cseed["out_equal"], equal_out=True).matrix.astype(np.float32)
    return {"real": real, "out_equal": fly(even),
            "null_mean": {r: np.mean([e[r] for e in each], axis=0).tolist() for r in READOUTS},
            "null": {r: [e[r] for e in each] for r in READOUTS}}


# ---------------------------------------------------------------- analysis

def ratio_ci(fly: np.ndarray, comps: np.ndarray, draws: int = BOOT_DRAWS, seed: int = 0) -> dict:
    """mean(fly) / max_c mean(comp_c) over trials, with a 95% percentile
    bootstrap over trials in which the maximum is re-taken."""
    fly = np.asarray(fly, float)
    c = np.atleast_2d(np.asarray(comps, float))
    means = c.mean(1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(fly), (draws, len(fly)))
    boot = fly[idx].mean(1) / c[:, idx].mean(2).max(0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"ratio": float(fly.mean() / means.max()), "ci95": [float(lo), float(hi)],
            "best": int(np.argmax(means))}


COMPARISONS = ("primary", "raw_asym", "all_variants")


def competitor_set(accounting: str, table: str = "primary", choice: dict | None = None) -> list[str]:
    """Training-free competitors at one accounting.

    primary       the criterion's set: symmetric fly overlap; for the Gaussian
                  sign code, DenseFly and the value tag the symmetric scorer
                  and the asymmetric scorer with its validated query variant
                  (`choice`, from `choose`); the value tag's ADC form.
    raw_asym      reported: the same with every competitor's raw-query form.
    all_variants  reported: every competitor scorer computed.
    """
    out = ["fly_sym"]
    for f in ASYM_FAMILIES:
        out.append(f"{f}_{accounting}_sym")
        if table == "primary":
            if choice is None:
                raise ValueError("the primary set needs the validation choice")
            out.append(f"{f}_{accounting}_asym_{choice[f'{f}_{accounting}']}")
        elif table == "raw_asym":
            out.append(f"{f}_{accounting}_asym_raw")
        elif table == "all_variants":
            out += [f"{f}_{accounting}_asym_{v}" for v in VARIANTS]
        else:
            raise ValueError(f"unknown table {table!r}")
    if table in ("primary", "all_variants"):
        out.append(f"vtag_{accounting}_adc")
    return out


def _nanmean0(v: np.ndarray) -> list:
    """Column means ignoring undefined (NaN) entries; NaN for an all-NaN column."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(v, 0).tolist()


def _arrays(trials: dict, what: str = "ap") -> dict[str, np.ndarray]:
    keys = sorted(trials, key=int)
    names = trials[keys[0]][what].keys()
    return {nm: np.array([[np.nan if v is None else v for v in trials[t][what][nm]] for t in keys], float)
            for nm in names}


def analyse_dataset(ds: dict, ks, ds_index: int = 0) -> dict:
    choice = ds["validation"]["choice"]
    ap = _arrays(ds["trials"])
    rc = _arrays(ds["trials"], "recall")
    fly_name = f"fly_asym_{choice['fly']}"
    out = {"fly_readout": fly_name, "validation_choice": dict(choice),
           "n_trials": len(ds["trials"]), "ks": list(ks),
           "mean_ap": {nm: _nanmean0(v) for nm, v in ap.items()},
           "mean_recall": {nm: _nanmean0(v) for nm, v in rc.items()},
           "comparisons": {}}
    accs = sorted({nm.split("_")[1] for nm in ap if nm.startswith("gauss_")},
                  key=lambda a: ACCOUNTING[a])
    for acc in accs:
        for label in COMPARISONS:
            comps = competitor_set(acc, label, choice)
            rows = []
            for i, k in enumerate(ks):
                avail = [c for c in comps if c in ap and np.isfinite(ap[c][:, i]).all()]
                missing = [c for c in comps if c not in avail]
                seed = int(np.random.SeedSequence([SEED_BOOT, ds_index, ACCOUNTING[acc], i])
                           .generate_state(1)[0])
                r = ratio_ci(ap[fly_name][:, i], np.array([ap[c][:, i] for c in avail]), seed=seed)
                per = {c: ratio_ci(ap[fly_name][:, i], ap[c][:, i][None, :], seed=seed) for c in avail}
                rows.append({"k": int(k), "bits": None, "ratio": r["ratio"], "ci95": r["ci95"],
                             "lower_ge_1": bool(r["ci95"][0] >= 1.0),
                             "best_competitor": avail[r["best"]], "competitors": comps,
                             "missing_competitors": missing,
                             "per_competitor": {c: {"ratio": v["ratio"], "ci95": v["ci95"]}
                                                for c, v in per.items()}})
            out["comparisons"].setdefault(acc, {})[label] = rows
        pq_name = f"pq_{acc}_adc"
        if pq_name in ap and np.isfinite(ap[pq_name]).all():
            out["comparisons"][acc]["pq_reference"] = [
                {"k": int(k), **{kk: v for kk, v in ratio_ci(ap[fly_name][:, i], ap[pq_name][:, i][None, :],
                                                             seed=i).items() if kk != "best"}}
                for i, k in enumerate(ks)]
    for acc in accs:
        for rows in out["comparisons"][acc].values():
            for i, r in enumerate(rows):
                r["bits"] = ds["bits"][acc][i]
    return out


def analyse_connectome(cn: dict, choice: str) -> dict:
    keys = sorted(cn["trials"], key=int)
    ks = cn["ks"]

    def arr(kind, readout):
        return np.array([cn["trials"][t][kind][readout] for t in keys], float)
    rows, inside_all = [], True
    for i, k in enumerate(ks):
        for gap in ("real", "out_equal"):
            sym = rp.contrast(arr(gap, "sym")[:, i], arr("null_mean", "sym")[:, i], seed=i)
            asy = {var: rp.contrast(arr(gap, f"asym_{var}")[:, i], arr("null_mean", f"asym_{var}")[:, i],
                                    seed=i) for var in VARIANTS}
            chosen = asy[choice]
            inside = bool(sym["ci95"][0] <= chosen["estimate"] <= sym["ci95"][1])
            inside_all &= inside
            rows.append({"k": int(k), "gap": "connectome-null" if gap == "real" else "even_fanout-null",
                         "symmetric": sym, "asymmetric": chosen, "asymmetric_readout": f"asym_{choice}",
                         "asymmetric_other": asy["std" if choice == "raw" else "raw"],
                         "inside": inside})
    return {"n_trials": len(keys), "ks": list(ks), "rows": rows, "dataset_passes": bool(inside_all),
            "mean_ap": {kind: {r: arr(kind, r).mean(0).tolist() for r in READOUTS}
                        for kind in ("real", "null_mean", "out_equal")}}


def paper_check(cn: dict, paper: dict | None) -> dict:
    """Largest absolute difference between the symmetric readout's per-trial
    AP (connectome, null mean, even fan-out) and the paper's
    `replication.trial_controls` values for the same dataset (results/
    controls.json, hemispheres.malecns_R.<dataset>). Comparable only with
    the paper's trial and null counts."""
    if not paper:
        return {"available": False}
    keys = sorted(cn["trials"], key=int)
    if len(keys) != paper["trials"] or cn.get("n_nulls") != paper["B"] \
            or not set(cn["ks"]) <= set(paper["sizes"]):
        return {"available": False, "reason": "different trial or null counts"}
    cols = [paper["sizes"].index(k) for k in cn["ks"]]
    diff = {}
    for kind, pk in (("real", "real"), ("null_mean", "null"), ("out_equal", "out_equal")):
        ours = np.array([cn["trials"][t][kind]["sym"] for t in keys], float)
        theirs = np.array(paper[pk], float)[:len(keys), cols]
        diff[kind] = float(np.abs(ours - theirs).max())
    return {"available": True, "max_abs_diff": diff,
            "identical": bool(max(diff.values()) < 1e-12)}


def analyse(res: dict, controls_path: str | Path | None = None) -> dict:
    st = res["settings"]
    ks = st["ks"]
    out = {"datasets": {}, "connectome": {}, "criteria": {}}
    for name, ds in res["datasets"].items():
        j = DATASETS.index(name) if name in DATASETS else len(DATASETS)
        out["datasets"][name] = analyse_dataset(ds, ks, ds_index=j)
    cpath = Path(controls_path) if controls_path else ROOT / "results" / "controls.json"
    paper = (json.loads(cpath.read_text()).get("hemispheres", {}).get("malecns_R", {})
             if cpath.exists() else {})
    for name, cn in res.get("connectome", {}).items():
        out["connectome"][name] = analyse_connectome(cn, res["datasets"][name]["validation"]["choice"]["fly"])
        out["connectome"][name]["paper_check"] = paper_check(cn, paper.get(name))
    full = tuple(ks) == KS
    cr = out["criteria"]
    if full and "glove" in out["datasets"]:
        rows = out["datasets"]["glove"]["comparisons"]["info"]["primary"]
        n_ok = sum(r["lower_ge_1"] for r in rows)
        cr["a_glove"] = {"k_passing": [r["k"] for r in rows if r["lower_ge_1"]], "n_passing": n_ok,
                         "required": 4, "pass": bool(n_ok >= 4)}
    if "mnist" in out["datasets"] and {2, 4} <= set(ks):
        rows = {r["k"]: r for r in out["datasets"]["mnist"]["comparisons"]["info"]["primary"]}
        cr["b_mnist"] = {"k2": rows[2]["lower_ge_1"], "k4": rows[4]["lower_ge_1"],
                         "pass": bool(rows[2]["lower_ge_1"] and rows[4]["lower_ge_1"])}
    if "a_glove" in cr and "b_mnist" in cr:
        cr["primary_pass"] = bool(cr["a_glove"]["pass"] and cr["b_mnist"]["pass"])
    crit_rows = [r for nm in ("glove", "mnist") if nm in out["datasets"]
                 for r in out["datasets"][nm]["comparisons"].get("info", {}).get("primary", [])]
    cr["all_competitors_present"] = not any(r["missing_competitors"] for r in crit_rows)
    if out["connectome"]:
        n_ok = sum(c["dataset_passes"] for c in out["connectome"].values())
        cr["c_connectome"] = {"datasets_passing": [nm for nm, c in out["connectome"].items()
                                                   if c["dataset_passes"]],
                              "n_passing": n_ok, "n_datasets": len(out["connectome"]),
                              "required": 3, "pass": bool(n_ok >= 3)}
    cr["protocol_as_preregistered"] = bool(
        full and st["trials"] == TRIALS and st["n_items"] == N_ITEMS and st["n_queries"] == N_QUERIES
        and tuple(st["datasets"]) == DATASETS and st["val_items"] == VAL_ITEMS
        and st["val_queries"] == VAL_QUERIES and tuple(st["conn_ks"]) == CONN_KS
        and st["conn_trials"] == CONN_TRIALS and st["conn_nulls"] == CONN_NULLS
        and tuple(st["conn_datasets"]) == DATASETS)
    return out


# ---------------------------------------------------------------- runner

def _sha256(path: Path) -> str | None:
    return fh.sha256(path) if path.exists() else None


def _save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def run(cfg: Config, out: str | Path | None = None, datasets=DATASETS, trials: int = TRIALS,
        n_items: int = N_ITEMS, n_queries: int = N_QUERIES, ks=KS, val_trials: int = VAL_TRIALS,
        val_items: int = VAL_ITEMS, val_queries: int = VAL_QUERIES, conn_datasets=DATASETS,
        conn_trials: int = CONN_TRIALS, conn_nulls: int = CONN_NULLS, conn_ks=CONN_KS,
        pq: bool = True, max_minutes: float | None = None, log=None) -> dict:
    """Run (or resume) E2 and criterion (c), checkpointing to `out` after
    every validation trial, test trial and connectome trial. With
    `max_minutes`, no new unit is started once the next one would overrun;
    the returned dict then has status "incomplete" and a later call resumes."""
    t_start = time.time()
    if max_minutes is not None and 60.0 * max_minutes <= LOAD_MARGIN:
        raise ValueError(f"--max-minutes must exceed {LOAD_MARGIN / 60:.0f} (the margin kept for loading "
                         "a dataset), or no invocation would make progress")
    deadline = None if max_minutes is None else t_start + 60.0 * max_minutes
    log = log or (lambda s: print(s, flush=True))
    path = Path(out) if out else ROOT / "results" / OUT_NAME
    if not set(conn_datasets) <= set(datasets):
        raise ValueError("connectome datasets must be among the datasets (they need its validation choice)")
    settings = json.loads(json.dumps({
        "datasets": list(datasets), "trials": trials, "n_items": n_items, "n_queries": n_queries,
        "ks": list(ks), "expansion": EXPANSION, "sampling": rp.SAMPLING, "top_fraction": rp.TOP,
        "value_bits": VALUE_BITS, "val_trials": val_trials, "val_items": val_items,
        "val_queries": val_queries, "pq": pq, "pq_sub_bits": PQ_SUB_BITS, "pq_iter": PQ_ITER,
        "conn_datasets": list(conn_datasets), "conn_trials": conn_trials, "conn_nulls": conn_nulls,
        "conn_ks": list(conn_ks), "bootstrap_draws": BOOT_DRAWS}))
    code_sha = _sha256(Path(__file__))
    if path.exists():
        res = json.loads(path.read_text())
        if res["settings"] != settings:
            raise SystemExit(f"{path} was started with other settings; use another --out")
        if res["code_sha256"] != code_sha:
            raise SystemExit(f"{path} was started with another version of {Path(__file__).name}; "
                             "results from two code versions are not mixed")
    else:
        res = {"experiment": "E2: graded-query readout of stored winner sets",
               "preregistration": "experiments/PREREGISTRATION_2026-09-28.md (UPDATES.md section 14)",
               "preregistration_sha256": _sha256(ROOT / "experiments" / "PREREGISTRATION_2026-09-28.md"),
               "code_sha256": code_sha, "settings": settings, "status": "running",
               "datasets": {}, "connectome": {}, "invocations": []}
    res["invocations"].append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "max_minutes": max_minutes})
    last: dict[str, float] = {}

    def over(kind: str) -> bool:
        return deadline is not None and time.time() + last.get(kind, 0.0) > deadline

    def near_end() -> bool:                     # before loading a dataset
        return deadline is not None and time.time() + LOAD_MARGIN > deadline

    def stop() -> dict:
        res["status"] = "incomplete"
        _save(res, path)
        log(f"  time budget reached; progress saved to {path}; run again to resume")
        return res

    for name in datasets:
        ds = res["datasets"].setdefault(name, {})
        val = ds.get("validation", {})
        if "choice" in val and len(ds.get("trials", {})) >= trials:
            continue
        if near_end():
            return stop()
        x = rp.load_benchmark(cfg, name, n=n_items)
        n, d = x.shape
        m = EXPANSION * d
        ds.update({"d": d, "m": m, "s": max(1, int(round(rp.SAMPLING * d))), "n_items": n,
                   "top": int(rp.TOP * n),
                   "bits": {a: [budget(a, m, k) for k in ks] for a in ACCOUNTING},
                   "value_tag_winners": {a: [value_tag_winners(budget(a, m, k), m) for k in ks]
                                         for a in ACCOUNTING},
                   "pq_subquantisers": {a: [len(pq_layout(budget(a, m, k), d)[0]) for k in ks]
                                        for a in ACCOUNTING}})
        ds.setdefault("trials", {})
        if "choice" not in val:
            xv, info = validation_set(cfg, name, val_items, n_items, x)
            val = ds.setdefault("validation", {**info, "n_queries": val_queries,
                                               "top": int(rp.TOP * len(xv)), "trials": {}})
            for v in range(val_trials):
                if str(v) in val["trials"]:
                    continue
                if over("val"):
                    return stop()
                t0 = time.time()
                val["trials"][str(v)] = codes_trial(xv, val_queries, ks, (SEED_VAL, v), pq=False)
                last["val"] = time.time() - t0
                _save(res, path)
            val["choice"], val["mean_ap"] = choose(val["trials"])
            val["chosen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            _save(res, path)
            log(f"  {name}: validation choice {val['choice']}")
        if ds["trials"]:
            last[f"trial_{name}"] = ds["trials"][max(ds["trials"], key=int)]["seconds"]
        for t in range(trials):
            if str(t) in ds["trials"]:
                continue
            if over(f"trial_{name}"):
                return stop()
            t0 = time.time()
            r = codes_trial(x, n_queries, ks, (SEED_TEST, t), pq=pq)
            r["seconds"] = time.time() - t0
            last[f"trial_{name}"] = r["seconds"]
            ds["trials"][str(t)] = r
            _save(res, path)
            log(f"  {name}: trial {t + 1}/{trials} ({r['seconds']:.0f} s)")

    if conn_datasets:
        from .experiments import context, null_pool
        ctx = context(cfg)
        for name in conn_datasets:
            cn = res["connectome"].setdefault(name, {"ks": list(conn_ks), "trials": {}})
            if len(cn["trials"]) >= conn_trials:
                continue
            if near_end():
                return stop()
            p = ctx.proj if name == "odours" else ctx.full
            pb = fh.Projection("b", p.binary, p.glomeruli)
            pool = null_pool(p, 200) if name == "odours" else null_pool(pb, 50)
            if len(pool) < conn_nulls:
                raise ValueError("not enough cached nulls")
            nulls = [qq.matrix.astype(np.float32) for qq in pool[:conn_nulls]]
            wiring = p.binary.astype(np.float32)
            g, mc = wiring.shape
            x = rp.load_benchmark(cfg, name, n=n_items)
            z = x if name == "odours" else rp.pca(x, g)
            cn.update({"n_glomeruli": g, "n_cells": mc, "nnz": int(wiring.sum()), "n_nulls": conn_nulls,
                       "input": "DoOR mixtures, measured glomeruli" if name == "odours"
                       else "PCA, random component-glomerulus assignment per trial"})
            if cn["trials"]:
                last[f"conn_{name}"] = cn["trials"][max(cn["trials"], key=int)]["seconds"]
            for t in range(conn_trials):
                if str(t) in cn["trials"]:
                    continue
                if over(f"conn_{name}"):
                    return stop()
                t0 = time.time()
                r = connectome_trial(name, z, wiring, nulls, pb, t, conn_ks, n_queries)
                r["seconds"] = time.time() - t0
                last[f"conn_{name}"] = r["seconds"]
                cn["trials"][str(t)] = r
                _save(res, path)
                log(f"  connectome {name}: trial {t + 1}/{conn_trials} ({r['seconds']:.0f} s)")

    res["analysis"] = analyse(res)
    res["status"] = "complete"
    res["completed"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save(res, path)
    log(f"  complete; results in {path}")
    log("  criteria: " + json.dumps(res["analysis"]["criteria"]))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m flypath.readout",
                                 description="E2 graded-query readout (pre-registered); "
                                             "resumable, checkpoints to results/e2_readout.json")
    ap.add_argument("--config", help="YAML config overriding config.example.yaml")
    ap.add_argument("--out", help="results file (default results/e2_readout.json)")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--items", type=int, default=N_ITEMS)
    ap.add_argument("--queries", type=int, default=N_QUERIES)
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--val-trials", type=int, default=VAL_TRIALS)
    ap.add_argument("--val-items", type=int, default=VAL_ITEMS)
    ap.add_argument("--val-queries", type=int, default=VAL_QUERIES)
    ap.add_argument("--conn-datasets", default=",".join(DATASETS), help="empty to skip criterion (c)")
    ap.add_argument("--conn-trials", type=int, default=CONN_TRIALS)
    ap.add_argument("--conn-nulls", type=int, default=CONN_NULLS)
    ap.add_argument("--conn-ks", default=",".join(map(str, CONN_KS)))
    ap.add_argument("--no-pq", action="store_true", help="skip the PQ-ADC reference")
    ap.add_argument("--max-minutes", type=float,
                    help="stop starting new units after this many minutes (exit code 75 = run again)")
    a = ap.parse_args(argv)

    def split(s):
        return [v for v in s.split(",") if v]
    from . import config
    cfg = config.load(a.config)
    res = run(cfg, out=a.out, datasets=split(a.datasets), trials=a.trials, n_items=a.items,
              n_queries=a.queries, ks=[int(v) for v in split(a.ks)], val_trials=a.val_trials,
              val_items=a.val_items, val_queries=a.val_queries, conn_datasets=split(a.conn_datasets),
              conn_trials=a.conn_trials, conn_nulls=a.conn_nulls,
              conn_ks=[int(v) for v in split(a.conn_ks)], pq=not a.no_pq, max_minutes=a.max_minutes,
              log=lambda s: print(s, flush=True))
    return 0 if res["status"] == "complete" else EXIT_MORE


if __name__ == "__main__":
    sys.exit(main())
