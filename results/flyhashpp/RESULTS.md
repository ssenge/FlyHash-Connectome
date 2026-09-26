# FlyHash++: initial ablation results (2026-09-26)

The suite is implemented and three exploratory studies completed. These results concern new held-out geometric retrieval protocols, **not** the paper's original AP@200 experiment. They do not yet establish a new state-of-the-art algorithm.

## Completed studies

| Study | Coverage | Seeds | Evaluated method/config/seed rows |
|---|---|---:|---:|
| [Pilot](pilot.md) / [JSON](pilot.json) | 24 methods, SIFT + MNIST, 2 expansions × 2 fan-ins × 2 winner counts | 3 | 1,152 |
| [Focused follow-up](followup.md) / [JSON](followup.json) | 8 methods, SIFT + MNIST + GloVe, m=256, s=6, k=8 | 10 | 240 |
| [Actual hemispheres](bilateral.md) / [JSON](bilateral.json) | MaleCNS left/right, global/concatenated/OR retrieval, learned/untrained masks, independent degree-preserving controls; SIFT | 3 | 30 |

Total: **1,422 evaluated rows**. Each retains per-query AP/recall, split IDs, model parameters, costs and timings. Paired differences and validation-selected arithmetic/storage frontiers are in JSON. Run configuration and implementation hashes are saved.

The two general studies fit PCA to 32 dimensions on 512 training rows only, use 1,000 gallery items, 64 validation queries and 64 held-out test queries, and score AP@20 against original-feature Euclidean neighbours. The bilateral study fits 51 PCs. Ground truth never changes between methods within a study/seed. Learned methods have only five epochs in the general pilot and two in the bilateral pilot: they are deliberately **not converged published-baseline reproductions**.

## Ten-seed follow-up

The follow-up fixes m=256, s=6, k=8. Values below are mean held-out **hash-only AP@20**, before any original-feature reranking.

| Method | SIFT | MNIST | GloVe | Analytical projection ops/item | Code bits/item estimate |
|---|---:|---:|---:|---:|---:|
| Original FlyHash | 0.1920 | 0.2792 | 0.0234 | 1,536 | 64 |
| Balanced | 0.1907 | 0.2877 | 0.0226 | 1,536 | 64 |
| Balanced + diversity | 0.1939 | 0.2862 | 0.0219 | 1,536 | 64 |
| Rank-coded balanced | 0.1986 | 0.2936 | 0.0227 | 1,536 | 64 |
| Balanced + 8-bit scale | 0.1929 | 0.3042 | 0.0215 | 1,536 | 72 |
| DenseFly | 0.4830 | 0.5895 | 0.0512 | 1,536 | 256 |
| Gaussian, matched arithmetic, real coordinates | 0.3612 | 0.4477 | 0.0493 | 1,536 | 768 |
| Gaussian signs, matched 64-bit storage | 0.2968 | 0.4095 | 0.0349 | 4,096 | 64 |

Bits are explicit fixed-width representations, not measured serialized codecs. Binary top-k uses eight 8-bit indices here; rank uses their order. An optimally compressed unordered binary code could be smaller, so this is not proof that rank information is free under every storage model. Execution uses float32 working arrays; their actual bytes are recorded separately. Projection arithmetic excludes winner selection and retrieval, which are included in measured phase times. Reranking costs and scores are reported separately in JSON.

### What the follow-up says

1. **The diversity hypothesis is not established.** Its promising three-seed SIFT result weakens at ten seeds. Diversity-minus-balance AP differences and unadjusted paired 95% intervals are SIFT +0.00320 [−0.00681, +0.01321], MNIST −0.00150 [−0.01552, +0.01251], GloVe −0.00062 [−0.00244, +0.00120]. Do not market this as a demonstrated improvement.
2. **Rank and explicit scale are worth a more targeted test.** On MNIST, rank-minus-balance is +0.00590 [+0.00177, +0.01003]; scale-minus-balance is +0.01646 [+0.01070, +0.02222]. Both are exploratory unadjusted intervals after screening, not confirmatory findings. Scale costs eight more bits, and neither gain transfers convincingly to GloVe.
3. **DenseFly is essential.** It substantially outperforms sparse top-k tags at this projection budget, while storing four times as many bits. It also exceeds the tested equal-arithmetic real Gaussian baseline in these means. That does not contradict the earlier paper's FlyHash result: DenseFly is a different readout. Joint arithmetic/storage frontiers, rather than equal-k comparisons, are necessary.
4. **Calibration and adaptive thresholds are not automatic upgrades.** The broader pilot contains mixed or negative results. Do not combine every ingredient into a single method without component ablations.
5. **Learning is inconclusive at this schedule.** BioHash and constrained learning are implemented and exercised, but five/two epochs cannot support claims against fully trained BioHash. Increase training duration based on validation curves before drawing a learned-versus-random conclusion.
6. **Two hemispheres did not improve this pilot.** Measured global WTA scored 0.1931; per-table concatenation 0.1875; OR candidates 0.1873. At the same maximum candidate budget, original-feature-reranked AP was 0.6305, 0.6193, 0.5810 respectively. Three seeds and one dataset are insufficient for a biological conclusion, but they provide no reason to prioritize this method as the next algorithmic claim.

## Efficiency interpretation

See [pilot trade-offs](pilot_tradeoffs.png) and [follow-up trade-offs](followup_tradeoffs.png). The plots separate analytical projection operations, code bits, and measured reference-code search time. The JSON includes construction/training, gallery encoding, query encoding, search, candidate/reranking times and working-memory estimates.

The timing backend is unoptimized NumPy/dense code scanning, with occupied pseudo-bin scanning for DenseFly probing. These measurements are useful implementation diagnostics, **not production ANN latency or actual sparse-kernel operation counts**. Other work was active in the shared workspace, so use a quiet machine and optimized packed/sparse kernels for definitive latency comparisons. No peak-RSS measurement is claimed.

## Next experiment to prioritize

Focus on **rank and magnitude information at genuinely matched storage**, with a stronger DenseFly baseline, rather than assuming balance plus diversity is already the winning design. Compare:

- binary top-k with increased k under the same 72-bit budget;
- rank coding and the explicit scale channel;
- DenseFly at the same storage budget, varying fan-in to explore the arithmetic trade-off;
- converged BioHash, with training cost and amortization reported;
- raw, normalized and angular ground truths as separate declared tasks.

Use validation to select configurations, then a fresh test set for the final claim. The ten-seed follow-up reuses pilot seeds 0–2 and is an exploratory extension, not an independent confirmation. The full manifest is provided but has **not** been run; it should be refined using these results before spending its much larger compute budget.

Implementation/protocol: [experiments/FLYHASHPP.md](../../experiments/FLYHASHPP.md). Validation: **93 tests passed**, including the new algorithm/scoring/leakage/budget invariants and the existing repository suite.
