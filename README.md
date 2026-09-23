# Random by design

**The fly's mushroom body is a good hash function because of its shape, not its
wiring.**

In 2017 Dasgupta, Stevens and Navlakha showed that the fruit fly's olfactory
circuit implements a locality-sensitive hash that beats classical LSH at
similarity search. They had no connectome, so they modelled the
projection-neuron to Kenyon-cell matrix as a *sparse binary random matrix*.

The MaleCNS v1.0 connectome, released September 2026, makes that assumption
testable for the first time. This repository substitutes the measured wiring
for their random one and asks whether it matters.

It does not.

![result](results/flyhash.png)

## The finding

Against a configuration-model null that preserves both degree sequences and
destroys only the pairing, the measured wiring shows **no advantage**
(z = −1.1 to −1.6; 35 to 38 of 40 random nulls score higher). At the sharpest
hash size the test resolves a 1.8% relative effect at 2σ, so any true advantage
is smaller than that.

Meanwhile the architecture reproduces its advantage: sparse expansion plus
winner-take-all beats classical LSH by about 5×.

Two conclusions follow. The 2017 result stands. And their simplifying
assumption was not merely convenient — it was **right**: for this task the real
matrix and a degree-matched random one are interchangeable. Where a random
projection is what the computation needs, evolution appears to have supplied
one.

## Results

Full connectome, no synapse threshold: 25,563,096 connections, 124M synapses.
The right mushroom body is 51 olfactory glomeruli × 1,886 Kenyon cells at 5.28
claws per cell, against an anatomical expectation of about six. Restricted to
the 35 glomeruli the odour data covers: 35 × 1,838.

Mean average precision at 10 nearest neighbours, 4,000 odour items, 40 seeds
per control:

| hash size | fly wiring | shuffle_kc | shuffle_both (null) | random claws | classical LSH |
|---|---|---|---|---|---|
| 8 | 0.1965 | 0.2102 | 0.2110 | 0.2469 | 0.0182 |
| 16 | 0.2882 | 0.3015 | 0.2966 | 0.3065 | 0.0662 |
| 32 | 0.3531 | 0.3699 | 0.3593 | 0.3535 | 0.1514 |
| 64 | 0.3960 | 0.4137 | 0.4000 | 0.3847 | 0.2462 |

### Can the test see a difference at all?

A null result is worthless if the instrument is blunt, so the same pipeline is
run on matrices whose answer is known in advance, and the null's own spread
gives a resolution bound:

| hash size | resolves at 2σ | balanced fan-out | wrecked wiring |
|---|---|---|---|
| 8 | 8.6% | z = −0.42 | z = −22.9 |
| 16 | 4.1% | z = −0.05 | z = −48.6 |
| 32 | 2.4% | z = +0.83 | z = −83.8 |
| 64 | **1.8%** | z = +1.61 | z = −110.0 |

Harm is detected overwhelmingly. An *improvement* the size of a balanced
fan-out sits just under the threshold, which is why the claim is stated as a
bound — "no advantage above 1.8%" — rather than "no advantage".

The conclusion survived a fourfold change in how much of the connectome is
used. An earlier run kept only connections of at least 5 synapses (6.2M
instead of 25.6M) and gave z = −1.1 to −1.8; dropping the threshold moved the
claw count from 4.79 to 5.28 and left the result where it was.

One asymmetry worth noting: the real fan-out is lopsided — DA1, the pheromone
glomerulus, feeds 473 Kenyon cells while the quietest feeds 19 — and the
controls that flatten it score slightly *higher*. Specialisation appears to
cost general-purpose hashing performance.

## Reproduce

```bash
conda env create -f environment.yml && conda activate flybrain
python -m flypath build                       # downloads ~1.1 GB, once
python -m flypath flyhash --seeds 40 --save   # regenerates the figure and JSON
pytest -m data                                # the result is pinned by a test
```

`flyhash --dataset odours` runs on the 172 measured odorants alone. That is the
real data but it is underpowered: at that size nothing short of a wrecked
matrix is detectable, which is why the headline uses mixtures.

## Method

**The matrix.** Rows are glomeruli (odour input channels), columns are Kenyon
cells, an entry is 1 where the connectome records a connection. Right
hemisphere: 51 olfactory glomeruli × 1,878 cells, 4.79 claws per cell — the
anatomically expected ~6. Restricted to the 35 glomeruli the odour data covers:
35 × 1,826.

**The hash**, exactly as in the 2017 paper: divisive normalisation, then
expansion through the matrix, then winner-take-all keeping the top *k* cells as
a sparse binary tag.

**The task.** Nearest-neighbour retrieval. For each odour, the 10 nearest by
Euclidean distance in glomerulus space are the ground truth; the 10 nearest by
Hamming distance between tags are the prediction; mean average precision scores
the overlap.

**The controls.** Only the matrix changes; input, normalisation, sparsity and
scoring are identical throughout.

| control | preserves | destroys |
|---|---|---|
| `shuffle_kc` | claws per cell | which glomeruli, and the lopsided fan-out |
| `shuffle_both` | claws per cell **and** cells per glomerulus | only the pairing |
| `random_claws` | the 2017 paper's own construction | — |
| `balanced` | claws per cell, fan-out made uniform | sensitivity check, upward |
| `degenerate` | nothing useful | sensitivity check, downward |

`shuffle_both` is the null that matters: it holds every degree constant, so a
difference can only come from *which* glomerulus pairs with *which* cell.

**The data.** Odour responses from DoOR 2.0, fetched at run time. Its consensus
matrix is ~17% filled, so the 172 usable odorants are those measured across at
least 12 glomeruli. Mixtures are Dirichlet-weighted blends of 2–5 measured
odorants, which preserves the natural correlation structure while giving enough
items for the null to tighten; from 172 to 4,000 items the null's spread
narrows about 3× and the result does not move.

## Limitations

Stated plainly, because they bound the claim.

- **Bounded, not absolute.** The claim is "no advantage above 1.8% relative",
  not "no advantage". Tightening it needs more items, and mean average
  precision costs O(n²).
- **One task, one metric.** Nearest-neighbour retrieval scored by mean average
  precision. Novelty detection, the circuit's other documented job, is untested.
- **Not a replication of the 2017 benchmark numbers.** The real matrix has fixed
  dimensions (35–51 in, ~1,878 out), so SIFT, GLOVE and MNIST cannot be passed
  through it without mangling them first. The architecture comparison is in kind,
  at the fly's own scale, on the fly's own input.
- **One animal.** MaleCNS is a single male fly. Whether this holds across
  individuals is unknown; FlyWire would give a female comparison.
- **Item independence.** 4,000 mixtures derive from 172 measured odorants, so
  they are not 4,000 independent measurements.
- **35 of 51 glomeruli.** Bounded by DoOR coverage, not by the connectome.

## Layout

```
web/mushroom-body-hash.html   an interactive demo: run the real wiring, then
                              scramble it and watch retrieval barely move
paper/flyhash.tex       the write-up (IEEE conference format, 4 pages)
flypath/flyhash.py      the experiment: wiring, controls, hash, scoring, figure
flypath/data.py         builds the connectome graph from the MaleCNS release
flypath/config.py       YAML config loading
tests/test_flyhash.py   12 tests, one of which pins the headline result
tests/test_data.py      sanity checks on the built graph
results/flyhash.json    the committed result behind the figure
```

Two commands: `build` and `flyhash`. `python -m flypath --help` has the flags.
Build the paper with `cd paper && pdflatex flyhash.tex` (twice, for references);
it picks up the figure from `results/`.

## Data and citation

- Connectome: MaleCNS v1.0 (HHMI Janelia, Google Research, University of
  Cambridge, MRC LMB), CC-BY. <https://male-cns.janelia.org/>
- Odours: DoOR 2.0, Münch & Galizia, *Scientific Reports* 6:21841 (2016).
- The result being tested: Dasgupta, Stevens & Navlakha, "A neural algorithm for
  a fundamental computing problem", *Science* 358:793–796 (2017).

Code is MIT; the datasets carry their own licences and are downloaded rather
than redistributed.
