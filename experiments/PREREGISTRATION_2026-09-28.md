
---

## 14. Pre-registration of follow-up experiments (2026-09-28, before any run)

Written before implementation or any full run. Pilot numbers in the idea report
(scratchpad `ideas_report.md`) motivated these designs; the criteria below are
fixed now and are not to be changed after results are seen. Deviations, if
unavoidable (e.g. a bug), are recorded as deviations with a reason.

### E2. Graded-query readout of stored winner sets (algorithmic)

- **Protocol.** `flypath.replication` 2017 protocol: 10,000 items, 1,000 test
  queries, truth = top 2% (200) by Euclidean distance on raw features, AP@200
  (all 200 true neighbours in the denominator) and recall@200, 50 trials.
  Datasets SIFT, GloVe, MNIST, odour mixtures. k ∈ {2, 4, 8, 16, 32}. Random fly
  matrices m = 10d, s = round(0.1d) inputs per cell, row-centred input.
- **Validation split.** A disjoint split (2,000 items, 200 queries) chooses one
  thing only: raw vs per-cell-standardised query drive, per dataset. Never
  chosen on test.
- **Method.** Store each item's k winner indices only. Score item j for query q
  as the sum of q's graded drive over j's winners (k lookups).
- **Storage accounting.** Primary: information-theoretic B(k) = ceil(log2
  C(m, k)) bits. Secondary, reported: fixed-width k * ceil(log2 m) bits.
- **Training-free competitors at B(k) bits.** Each gets symmetric and
  asymmetric (graded query · stored bits) scoring:
  - Gaussian sign code with B bits (asymmetric = Gordo et al. 2014 style);
  - DenseFly: sparse binary expansion to B cells with s inputs each, one sign
    bit per cell;
  - FlyLSH value tag: k' winners with 4-bit quantised values,
    k' = floor(B / (ceil(log2 m) + 4)), asymmetric scoring;
  - symmetric fly overlap (the paper's scorer).
- **Trained upper reference, reported not scored.** PQ-ADC at B bits.
- **Pass (primary, both required).**
  - (a) GloVe: the ratio AP(fly asymmetric) / AP(best training-free
    competitor) has a trial-bootstrap 95% lower bound ≥ 1.0 for ≥ 4 of the 5
    values of k.
  - (b) MNIST: the same at k = 2 and k = 4.
- **Secondary (paper robustness).** (c) MaleCNS R, k ∈ {4, 16}: the
  connectome−null and even-fan-out−null gaps under the asymmetric readout lie
  inside the symmetric readout's 95% interval on ≥ 3 of 4 datasets.

### E3. Collision law of the fly hash (theory)

- **(a) Equivalence.**
  - Setup: SIFT, GloVe, MNIST raw; 10 trials × 1,000 queries; m = 10d,
    s = 0.1d, k ∈ {4, 16}. AP@200 of the sparse binary fly hash vs
    Gaussian-WTA (dense Gaussian projection, same m, same top-k).
  - Pass: |AP_fly − AP_GWTA| / AP_GWTA ≤ 5% in ≥ 5 of 6 cells.
- **(b) Collapse.**
  - θ½ is the angle at which the mean winner-set overlap fraction
    |T(x) ∩ T(y)| / k equals 0.5, from ≥ 10^5 pairs binned by angle, with
    linear interpolation between bins.
  - Predictor: θ½ = c / t with t = Φ̄⁻¹(k/m).
  - One c is fitted on MNIST random matrices (all k, m cells), then frozen.
  - Held-out grid: k ∈ {4, 16, 64, 256} × m ∈ {20k, 10d, 40d} on SIFT, GloVe
    and odours; plus the 7 connectome matrices at their own m with k = 5%, on
    odour mixtures.
  - Pass: |θ½ − c/t| < 3° in ≥ 90% of held-out random cells and in ≥ 6 of 7
    connectome matrices.
- **(c) Fan-in boundary.**
  - Inputs: PCA-51 SIFT, MNIST, GloVe and odours (d = 35); m = 1,838;
    k ∈ {16, 92}; s ∈ {1, 2, 3, 5, 8, 13, 26, d} (26 capped at d).
  - Deficit(s) = 1 − AP_fly(s) / AP_GWTA; d_eff = participation ratio of the
    input covariance.
  - Pass on ≥ 3 of 4 inputs, both k: Spearman(s, deficit) ≤ −0.9; deficit
    > 10% for every s ≤ 5; deficit < 5% for every s ≥ 0.3·d_eff.

### E4. Spectral fan-out allocation (algorithmic, within sparse WTA hashes)

- **Stage 0.** Derive f*(Λ, k) under a Gaussian-drive, order-statistic top-k
  model at nnz = 9,967 with MaleCNS R inputs per cell. Freeze it (code plus
  SHA-256 in UPDATES.md) before any new AP run.
- **Stage 1: synthetic.**
  - Setup: d = 51, λ_j ∝ j^−β, β ∈ {0, 0.5, 1, 1.5, 2}, k ∈ {4, 16, 94},
    10 trials, AP@200.
  - Pass: at β = 0, f* is flat (CV < 0.05) with AP within ±1% of even
    fan-out; the best grid exponent α decreases in β and in k
    (Spearman ≤ −0.8).
- **Stage 2: real inputs.**
  - Setup: PCA-51 SIFT, MNIST, GloVe; Λ fitted on a disjoint training split;
    20 trials.
  - Pass at k ∈ {4, 16}: f* lies inside the best-grid-α 95% interval on 3 of
    3 datasets, AND f* beats both even fan-out and the best λ^(α/2) input
    pre-scaling (paired 95% intervals excluding 0) on ≥ 2 of 3 datasets.

### F1. Weak-connection audit (paper revision)

- Q z-score of every hemisphere against ≥ 100 curveball nulls at
  connection-weight thresholds w ≥ 1, 2, 3, 5 synapses; descriptive.
- Fixes the paper's hemibrain statement if its Q departs from null at w ≥ 2.

### F2. Larger null ensembles (paper revision)

- Odour primary endpoint for all 7 hemispheres with B = 2,000 curveball nulls
  (p floor 0.001), Holm across the 7. Replaces the B = 100 results in the
  paper whatever the outcome.
