# FlyHash++: methods and efficiency ablations

This is a separate, exploratory algorithm-development study. It does not change the connectome paper's result files. Implementations live in `flypath/plusplus.py`; execute with `python -m flypath.plusplus`. No new method is assumed to improve retrieval, and no novelty/priority claim is established by implementing it.

## Implemented ideas

| Idea | Methods and ablation | Question |
|---|---|---|
| A: balanced coverage | `fly` → `balanced` | Does even fan-out help at identical per-cell input counts and total connections? |
| New: projection diversity | `balanced` → `diverse` | Do degree-preserving switches that reduce four-cycles improve retrieval beyond balancing? |
| New: winner calibration | `fly` → `zscore`, `calibrated`; `balanced` → `balanced_calibrated`; `diverse` → `diverse_calibrated` | Does training-only activation calibration help, and does it interact with balance/diversity? |
| B: operation-optimal design | Cartesian grid of expansion, fan-in, winner count; validation Pareto frontier | Which constructions provide more quality at a given arithmetic AND storage budget? |
| C: rank codes | `balanced` → `rank` | Is rank information worth its storage and retrieval cost? |
| D: adaptive inhibition | `balanced` → `adaptive_relative`, `adaptive_external`, `scale_tag` | Do variable-size tags help? Does explicitly retaining input scale help? |
| E: two tables | `two_global` → `two_concat` → `two_or` | Does splitting a fixed projection into two independent tables and allocating winners per table help? |
| E: actual hemispheres | optional `measured_*` versus `null_*` | Do MaleCNS left/right tables help compared with independent degree-preserving randomizations with identical margins? |
| F: constrained learning | `sparse_untrained` → `sparse_bio`, also on measured/null masks in bilateral mode | Does a fixed-support learning rule help beyond its signed random initialization? |
| Published learned baseline | `biohash_untrained` → `biohash` | What does unconstrained BioHash learning buy at its actual dense arithmetic/storage/training cost? |
| Published random baseline | `densefly`, `densefly_probe` | Sparse binary projection, dense sign code, and pseudo-hash candidate generation. |
| Gaussian baselines | `gaussian_real`, `gaussian_sign`, `gaussian_ops`, `gaussian_storage` | Compare k real coordinates, k sign bits, arithmetic-matched real coordinates, and storage-matched sign codes. |

All ablations keep their paired seed, splits, truth and relevant construction fixed. Trade-offs such as rank-code storage, calibration state or learned multiplication costs are explicitly charged rather than described as free improvements.

### Algorithm details and provenance

**FlyHash:** exactly s binary input connections per cell; seeded fixed cell priorities resolve ties without perturbing unequal values. Retrieval-distance ties use seeded fixed gallery priorities. These policies are separate.

**Balance:** `flyhash.margin_control` preserves column sums and makes row sums as equal as integer totals permit. Pairing uses ten curveball sweeps here; this is a construction heuristic, not a claim of exact uniform sampling.

**Diversity:** minimizes the count of four-cycles, equivalently the sum of choose(overlap, 2) over pairs of projection columns. Greedy 2×2 edge switches preserve both margins. Only strict objective reductions are accepted. This is an original experimental construction in this repository, not an established novel algorithm in the literature. Construction time is included in fit time. The objective need not correlate with retrieval, so record both accepted switches and before/after energy.

**Calibration:** z scores or per-cell empirical CDF ranks fit on the training split only. Calibration overhead and stored training activation distributions are charged. The CDF implementation keeps the full sorted calibration data; a quantile-table approximation would be a separate efficiency improvement.

**Rank code:** store the ordered k winning indices; score using Manhattan distance between rank-valued sparse vectors (absent=0, weakest winner=1). Fixed-width storage is k·ceil(log2 m), as for an ordered active-index list; a combinatorially compressed unordered binary tag could be smaller, but such a codec is not implemented. No claim of a free rank encoding is made.

**Adaptive thresholds:** fit a scalar threshold on training activations to target average k, either after per-input mean/SD standardization or on unstandardized projected drives. Ties can prevent exact target occupancy; achieved occupancy is reported. The relative threshold remains scale-invariant. `scale_tag` instead appends an explicitly quantized 8-bit log input norm to the balanced fixed-k code; the 1st/99th percentile range is training-only. This is the revised concentration-preserving proposal, not a claim that removed information can be recovered. Input norm is measured before row centring/PCA. Scale weight is fixed at one on [0,1] in this pilot; any tuning must use validation.

**Two tables:** synthetic tables are independent random draws with a total of m cells and m·s edges. `two_global` applies global k-WTA to their concatenated drives; `two_concat` uses k/2 winners per table (integer split); `two_or` unions candidate lists from each table, then ranks their codes. Measured hemispheres use all shared glomeruli and preserve actual hemisphere sizes. Their independent nulls keep each hemisphere's exact margins; global and two-table codes share the same concatenated matrix. This controls total computation instead of silently doubling it. Hemispheric biological independence is not assumed.

**BioHash:** implements the generalized Hebbian/anti-Hebbian current and update in Eqs. 1–2 and the top-k inference rule in Eq. 6 of [Ryali et al., ICML 2020](https://proceedings.mlr.press/v119/ryali20a/ryali20a.pdf). Weights initialize N(0,1). A minibatch sums updates, scales by their largest absolute entry, and uses a linearly decaying learning rate. Configuration exposes p, anti-Hebbian strength, competing rank, epochs and batch size. The pilot uses p=2, rank=2 and delta=.4. This is a documented NumPy implementation of the rule, **not a reproduction of published performance**: our geometric retrieval task, preprocessing and training schedule differ. `sparse_bio` masks these updates to fixed support and is explicitly our constrained variant. It does not learn which edges exist. Learned rewiring with movable support remains an extension; do not label fixed-mask learning as learned topology. The zero-epoch counterparts isolate learning from initialization.

**DenseFly:** implements the fixed-input-count construction and sign code from Algorithm 1 of [Sharma & Navlakha](https://arxiv.org/html/1812.01844). The reference repository instead uses Bernoulli masks in its DenseFly class; this suite deliberately uses the paper's fixed-count construction for exact connection accounting. `densefly_probe` also constructs pseudo-hashes by signs of sums of contiguous activation blocks (up to 16 pseudo-bits). It scans occupied bins in Hamming order until reaching a candidate cap, truncating the last bin deterministically. This cap-based policy differs from the paper's fixed-radius experiments and is labelled as an adaptation. It is not a reproduction of their query-time claims.

## Evaluation protocol

- Disjoint training, validation-query, test-query and gallery rows; split IDs saved per seed.
- PCA, when requested, fits on **training rows only**. Row-centred vectors feed every hash, but **ground-truth Euclidean neighbours always use the original features**, including with PCA. This intentionally avoids changing truth between algorithms or dimensions.
- Training and calibration never use gallery, validation or test queries. All methods see the same original-feature truth within a seed.
- AP@R divides by R ground-truth relevant neighbours, including misses. Recall@R is also reported. Missing candidates are padded with −1 and count as misses; difficult queries are never dropped.
- Report hash-only retrieval and, separately, retrieval after original-feature reranking with a common maximum candidate budget. Candidate recall and actual candidate counts are recorded. OR may return fewer distinct candidates because of overlap.
- Training/construction, gallery encoding, query encoding and search timings are separate. Search uses median timings over repeated batches. PCA preprocessing time is recorded separately. Additional candidate-generation-plus-reranking time includes the raw-feature distance calculations.
- Projection arithmetic counts additions and multiplications separately. Binary projections cost nnz additions analytically; learned/Gaussian projections cost two operations per nonzero. Runtime currently uses NumPy dense matrix multiplication, so analytic sparse arithmetic is **not** a measurement of the executed BLAS instruction count.
- Stored-code bits describe fixed-width active-index or packed-bit representations; execution uses float32 arrays. Actual working-array bytes, weight/calibration state and reranker gallery-feature bytes are also recorded. Packed codecs and production ANN indexes are not implemented; no optimized latency or peak-RSS claim is justified.
- Synthetic OR retrieval scans both tables; it is not a sublinear index. DenseFly probing scans occupied pseudo-bins. Candidate discovery costs are timed and never treated as free.
- Save per-query metrics and per-seed results. Paired differences use seed-level Student-t 95% intervals, conditional on this dataset. These are exploratory, unadjusted intervals and do not make seeds independent biological samples. Three pilot seeds are insufficient for firm conclusions.
- The joint quality/arithmetic/storage frontier is selected on **validation AP**, not test AP. Test results are displayed for all configurations for exploratory diagnosis; use a fresh holdout for a later confirmatory claim.

## Running

From the repository root, using the existing flybrain environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m flypath.plusplus \
  --config experiments/flyhashpp_pilot.json --output results/flyhashpp/pilot.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m flypath.plusplus \
  --config experiments/flyhashpp_full.json --output results/flyhashpp/full.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m flypath.plusplus \
  --config experiments/flyhashpp_bilateral.json --output results/flyhashpp/bilateral.json
```

An existing output requires explicit `--overwrite`, otherwise the command refuses to replace it. Each completed dataset/seed checkpoints JSON plus a Markdown report. Outputs record configuration, source hashes, commit and runtime environment. They do not overwrite the paper's `replication.json` or its figures.

The pilot uses SIFT and MNIST, 32 training-fitted PCs, 3 seeds, 512 training items, 1,000 gallery items, 64 validation and 64 test queries, AP@20, five learning epochs. It is a screening experiment, not AP@200 replication. The larger manifest proposes original-dimensional SIFT/GloVe/MNIST/odour experiments, 10 seeds and 100 learning epochs. It is computationally expensive, not automatically a scientifically sufficient convergence schedule. Check training curves and validation performance before fixing a publication protocol.

The bilateral pilot uses SIFT with training-fitted 51-dimensional PCA and actual MaleCNS masks. It requires locally built MaleCNS data. Setting `methods: []` runs only the optional anatomical table study. The same adapter supports odours on their measured glomeruli. For a biological robustness claim, separately test imputation and a second animal; this suite is an algorithm study.

## Decision criteria

1. Does diversity beat balancing, rather than merely beat the original random matrix?
2. Do calibration and diversity provide independent gains, or only redundant gains?
3. Do any gains remain at equal arithmetic AND storage? Use the full frontier rather than comparing only equal k.
4. Are improvements worth training, construction, calibration and retrieval costs? Compare deployment scenarios with many versus few queries.
5. Do results transfer across datasets and hold out under additional seeds? Compare learning curves before claiming superiority to BioHash.
6. Does the two-table advantage survive independent random-table controls and the same candidate budget?

A null or negative answer is a valid ablation result. Do not add FlyHash++ superiority claims to the paper until these questions have been answered with adequate replication.

## Completed exploratory runs

See [the results interpretation](../results/flyhashpp/RESULTS.md). The 24-method pilot, actual-hemisphere study and ten-seed follow-up have completed (1,422 rows total). The follow-up manifest fixes m=256, s=6, k=8 on SIFT/MNIST/GloVe and reuses the first three pilot seeds; it is exploratory. The larger original-dimensional 100-epoch manifest has not been run. Importantly, the follow-up did not establish a diversity benefit, and currently motivates rank/magnitude ablations more strongly.

Generate standalone trade-off figures without touching the paper:

```bash
MPLCONFIGDIR=/tmp/flyhashpp-mpl python experiments/plot_flyhashpp.py results/flyhashpp/pilot.json
MPLCONFIGDIR=/tmp/flyhashpp-mpl python experiments/plot_flyhashpp.py results/flyhashpp/followup.json
```
