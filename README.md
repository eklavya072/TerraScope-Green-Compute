# TerraScope

**What does a percentage point of land-cover accuracy cost in energy?**

[![CI](https://github.com/eklavya072/TerraScope-Green-Compute/actions/workflows/ci.yml/badge.svg)](https://github.com/eklavya072/TerraScope-Green-Compute/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-run%20the%20models-1b4332)](https://eklavya072.github.io/TerraScope-Green-Compute/)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/results%20data-CC--BY--4.0-blue)](LICENSE-DATA)
[![EuroSAT](https://img.shields.io/badge/dataset-EuroSAT%20(MIT)-green)](https://github.com/phelber/eurosat)

A Green AI benchmark. Five architectures, three numeric precisions, five seeds,
one identical recipe, measured on CPU only across 300 windows. Every published
number regenerates from committed data, and CI fails if one drifts.

---

## The finding

|                      | ResNet-50 (fp32) | EfficientNet-Lite0 (int8) |    difference |
| -------------------- | ---------------: | ------------------------: | ------------: |
| Top-1 accuracy       |           98.12% |                    97.45% |     −0.67 pp |
| Energy per 1k images |          60.84 J |                    3.18 J |     19× less |
| p95 latency          |         12.24 ms |                   0.43 ms |   28× faster |
| Model on disk        |          94.0 MB |                    3.8 MB |  25× smaller |

Switching to the quantised mobile model costs **0.67 percentage points** of
accuracy. It buys **19× less energy**, **28× faster** inference and
**25× less disk**.

![Accuracy versus energy, with the Pareto frontier marked](results/pareto.png)

Accuracy spans a 1.25 pp spread across all five architectures, and
8 of 10 pairwise comparisons survive Holm-Bonferroni correction. The
differences are real, but small. Energy spans a factor of 19.

> **On CPU-only hardware, deployment is an energy decision, not an accuracy one.**

Energy is *estimated* from on-die power telemetry, not metered at the wall. That
distinction is kept everywhere it appears. See [Limitations](#limitations).

---

## Why this question

Land-cover classification runs in public-sector and development organisations on
the hardware they already own, which is usually a CPU server with no GPU. UNDP's
Accelerator Labs apply earth-observation models to exactly this task, and UNDP
names Green Compute as one of five foundations for national AI ecosystems.

So the operative question is not which model scores highest. It is what the
score costs to run. Papers report accuracy; deployments pay for joules, latency
and memory.

Schwartz et al. argued this in [Green AI](https://doi.org/10.1145/3381831)
(CACM 2020): efficiency belongs in the evaluation, because reporting accuracy
alone rewards whoever spends the most compute. Strubell et al. (ACL 2019) costed
the training side, and Henderson et al.
([JMLR 2020](https://jmlr.org/papers/v21/20-312.html)) pushed for systematic
energy reporting. This is one complete instance of that, applied to inference,
where a deployed model spends most of its lifetime energy.

---

## Try it

**[Run the models in your browser](https://eklavya072.github.io/TerraScope-Green-Compute/)**

No install. Real ONNX graphs execute on a held-out test tile via WebAssembly and
are timed on your machine. Accuracy and energy are looked up from the committed
benchmark, never invented.

Or reproduce it locally:

```bash
make setup     # locked environment (uv + uv.lock)
make data      # fetch EuroSAT, write a sha256 manifest
make split     # regenerate the committed split, verified byte-identical
make train     # 5 architectures x 5 seeds, one shared recipe
make export    # ONNX fp32 + int8 dynamic/static
make bench     # latency, memory and energy matrix (CPU only)
make report    # rebuild every table, figure and README number
```

Training uses a GPU where one exists, purely to make the matrix tractable. All
benchmarking is CPU-only, and no reported figure depends on the training device.

<details>
<summary>Energy measurement needs a privileged sampler</summary>

Apple Silicon exposes on-die power only to root, so a sampler runs alongside the
benchmark:

```bash
sudo ./scripts/energy_sampler.sh
```

Start it before `make bench` and leave it running. Without it the benchmark
still records accuracy, latency and memory, and reports every energy column as
`null` rather than substituting an estimate.
</details>

---

## Results

<!-- BEGIN:results_table -->
### Results (ONNX Runtime CPU EP, intra-op threads=1, batch=1)

Accuracy is the mean over 5 seeds with a Student-t 95% confidence interval. Energy figures are ESTIMATED from on-die power telemetry, not metered at the wall. CO2e assumes 481 gCO2e/kWh (world average grid carbon intensity, ~481 gCO2e/kWh).

| Model | Precision | Params | Accuracy % (mean ± 95% CI) | p95 latency (ms) | Model RSS (MB) | Model (MB) | Energy/1k inf (J, estimated) | CO2e/1M inf (g, estimated) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| efficientnet_lite0 | fp32 | 3.38M | 97.61 ± 0.13 | 3.74 | 31 | 13.5 | 19.28 | 2.58 |
| efficientnet_lite0 | int8_dynamic | 3.38M | 63.44 ± 3.97 | 4.43 | 19 | 3.6 | 25.75 | 3.44 |
| efficientnet_lite0 | int8_static | 3.38M | 97.45 ± 0.32 | 0.43 | 17 | 3.8 | 3.18 | 0.42 |
| mobilenetv3_large | fp32 | 4.21M | 97.10 ± 0.17 | 2.73 | 35 | 16.8 | 14.46 | 1.93 |
| mobilenetv3_large | int8_dynamic | 4.21M | 70.47 ± 8.10 | 2.88 | 19 | 4.4 | 19.32 | 2.58 |
| mobilenetv3_large | int8_static | 4.21M | 91.10 ± 1.11 | 0.52 | 21 | 4.7 | 3.60 | 0.48 |
| mobilenetv3_small | fp32 | 1.53M | 97.19 ± 0.10 | 1.40 | 18 | 6.1 | 6.97 | 0.93 |
| mobilenetv3_small | int8_dynamic | 1.53M | 14.57 ± 2.95 | 1.54 | 14 | 1.7 | 8.02 | 1.07 |
| mobilenetv3_small | int8_static | 1.53M | 32.53 ± 10.40 | 0.32 | 19 | 1.9 | 1.72 | 0.23 |
| mobilevit_s | fp32 | 4.94M | 98.36 ± 0.44 | 4.46 | 37 | 20.0 | 26.31 | 3.51 |
| mobilevit_s | int8_dynamic | 4.94M | 58.41 ± 10.82 | 3.91 | 30 | 5.5 | 24.53 | 3.28 |
| mobilevit_s | int8_static | 4.94M | 49.50 ± 9.13 | 2.11 | 28 | 5.8 | 13.77 | 1.84 |
| resnet50 | fp32 | 23.53M | 98.12 ± 0.30 | 12.24 | 162 | 94.0 | 60.84 | 8.13 |
| resnet50 | int8_dynamic | 23.53M | 80.49 ± 8.15 | 5.36 | 39 | 23.7 | 27.08 | 3.62 |
| resnet50 | int8_static | 23.53M | 97.22 ± 0.38 | 2.49 | 67 | 24.0 | 14.39 | 1.92 |
<!-- END:results_table -->

ONNX export is accuracy-neutral: all 25 exported fp32 graphs reproduce their
PyTorch checkpoint's test accuracy exactly, 25/25 identical, maximum difference
0.000000 pp. Latency is measured on the exported graph while accuracy is
attributed to the model, so without this check a row could describe two
different models.

---

## What quantisation costs

<!-- BEGIN:quantisation -->
### Accuracy cost of int8 quantisation

Paired per-seed differences against each model's own fp32 export, mean with a Student-t 95% confidence interval. Negative means quantisation lost accuracy.

| Model | Precision | fp32 % | int8 % | Δ (pp, mean ± 95% CI) |
|---|---|---:|---:|---:|
| efficientnet_lite0 | int8_dynamic | 97.61 | 63.44 | -34.17 ± 4.07 |
| efficientnet_lite0 | int8_static | 97.61 | 97.45 | -0.16 ± 0.23 |
| mobilenetv3_large | int8_dynamic | 97.10 | 70.47 | -26.63 ± 8.12 |
| mobilenetv3_large | int8_static | 97.10 | 91.10 | -6.01 ± 1.08 |
| mobilenetv3_small | int8_dynamic | 97.19 | 14.57 | -82.62 ± 2.92 |
| mobilenetv3_small | int8_static | 97.19 | 32.53 | -64.66 ± 10.34 |
| mobilevit_s | int8_dynamic | 98.36 | 58.41 | -39.94 ± 11.18 |
| mobilevit_s | int8_static | 98.36 | 49.50 | -48.86 ± 9.28 |
| resnet50 | int8_dynamic | 98.12 | 80.49 | -17.63 ± 7.89 |
| resnet50 | int8_static | 98.12 | 97.22 | -0.90 ± 0.50 |
<!-- END:quantisation -->

Quantisation is architecture-specific, not a uniform tax.

- EfficientNet-Lite0 loses **0.16 ± 0.23 pp**. The interval contains zero, so its
  int8 form is statistically indistinguishable from its fp32 parent while using
  6× less energy. ResNet-50 loses 0.90 pp.
- MobileNetV3-Small loses **64.66** pp and MobileViT-S **48.86** pp. Post-training
  quantisation destroys both. Do not deploy either in int8 without
  quantisation-aware training.
- int8 *dynamic* is strictly dominated: worse accuracy and worse energy than fp32
  for every model here. Reported because a negative result saves someone the
  experiment.

<details>
<summary>We tried to recover MobileNetV3-Small, and could not</summary>

Per-channel weights, min-max / percentile / entropy calibration, restricting
quantisation to Conv/Gemm, excluding depthwise convolutions, signed and unsigned
activations, and batch sizes 1 and 64 all fail. Its hard-swish and squeeze-excite
activation distributions are the textbook case post-training quantisation cannot
represent. Recovering them needs quantisation-aware training, which is out of
scope for a post-training benchmark.
</details>

---

## Are the accuracy differences real?

<!-- BEGIN:significance -->
### Which accuracy differences are statistically distinguishable?

Welch's t-test over seeds, Holm-Bonferroni corrected across all pairwise comparisons (family-wise alpha = 0.05).

| Comparison | Δ accuracy (pp) | p | Holm threshold | Distinguishable? |
|---|---:|---:|---:|---|
| efficientnet_lite0 vs mobilenetv3_small | +0.41 | 0.0001 | 0.0050 | **yes** |
| mobilenetv3_large vs resnet50 | -1.01 | 0.0001 | 0.0056 | **yes** |
| efficientnet_lite0 vs mobilenetv3_large | +0.50 | 0.0002 | 0.0063 | **yes** |
| mobilenetv3_small vs resnet50 | -0.93 | 0.0005 | 0.0071 | **yes** |
| mobilenetv3_large vs mobilevit_s | -1.25 | 0.0006 | 0.0083 | **yes** |
| mobilenetv3_small vs mobilevit_s | -1.16 | 0.0014 | 0.0100 | **yes** |
| efficientnet_lite0 vs resnet50 | -0.51 | 0.0063 | 0.0125 | **yes** |
| efficientnet_lite0 vs mobilevit_s | -0.75 | 0.0073 | 0.0167 | **yes** |
| mobilenetv3_large vs mobilenetv3_small | -0.09 | 0.2480 | 0.0250 | no |
| mobilevit_s vs resnet50 | +0.24 | 0.2565 | 0.0500 | no |

8 of 10 pairwise accuracy differences are statistically distinguishable after correction.
<!-- END:significance -->

Two cautions. Statistically distinguishable is not operationally meaningful: the
whole best-to-worst spread is 1.25 pp, and a difference can be reliable yet far
too small to justify changing a deployment. And these compare architectures under
one fixed recipe and budget, so a model that trains poorly here might do better
with tuning it was deliberately not given.

The Pareto frontier above is the true mathematical one, so it includes
`mobilenetv3_small int8_static` purely because nothing is cheaper. At 32.5%
accuracy that configuration is useless in practice. Above 97% accuracy the
frontier is EfficientNet-Lite0 int8_static (97.45%, 3.18 J/1k), with ResNet-50
int8_static (97.22%, 14.39 J/1k) dominated by it, and MobileViT-S fp32 (98.36%,
26.31 J/1k) buying the last 0.9 pp for 8.3× the energy.

---

## Deployment recommendation

For a CPU-only server classifying Sentinel-2 RGB tiles, deploy
**EfficientNet-Lite0 quantised to int8 with static calibration**.

It gives up 0.67 pp against the ResNet-50 fp32 baseline and 0.90 pp against the
most accurate model measured. In exchange: 19× less energy, 28× lower p95
latency, a 3.8 MB artefact, roughly 2,300 images/second on one thread, and a
17 MB resident footprint that leaves the machine free for other work.

If the last 0.9 pp genuinely matters, and given the dataset caveats below that
should be argued rather than assumed, MobileViT-S fp32 is accuracy-optimal at
8.3× the energy. Do not deploy a quantised MobileNetV3 or MobileViT without
quantisation-aware training.

> **This recommendation is Apple Silicon-specific.** CI re-times the committed
> graphs on x86 every push and the ordering shifts (rho ≈ 0.68). fp32 gains
> substantially there, to the point that `mobilenetv3_small fp32` becomes the
> fastest configuration measured, at 97.19% accuracy and with no quantisation
> risk. Re-measure on your target hardware.

<details>
<summary><b>Is the energy column just the latency column in different units?</b></summary>

A fair objection, answered with the data. Energy is power × time, so if package
power were constant the energy axis would carry nothing latency does not.

Across all 75 measurement windows at 1 thread, batch 1:

| Quantity | Range | Ratio |
|---|---|---|
| p50 latency | 0.251 → 12.239 ms | 48.8× |
| Energy per 1,000 inferences | 1.56 → 63.12 J | 40.5× |
| Mean package power | 4.86 → 10.24 W | 2.1× |

Energy correlates with latency at r = 0.983. Most of the energy spread is the
latency spread, and saying so plainly beats implying two independent findings.

The remaining 2.1× is not nothing:

- Quantised models draw systematically more power: 5.51 W for fp32, 6.16 W for
  int8-dynamic, 6.95 W for int8-static, a 26% increase for static int8 over fp32.
  Quantisation does not simply make the same work shorter. It makes the CPU work
  harder while it runs, so a latency-only reading overstates int8's advantage.
- It reorders one pair. By latency `mobilenetv3_large int8_dynamic` beats
  `efficientnet_lite0 fp32`; by energy the order reverses. One swap in fifteen is
  a small effect, and reporting it as small is the honest framing.
- Across thread counts the two decouple. Package power spans 3.7× among the
  4-thread windows against 2.1× at 1 thread. That is also where core placement
  confounds the comparison, so no cross-model conclusion is drawn from it.

At a fixed thread count, energy is largely a restatement of latency with a real
second-order power term that matters most when comparing precisions. The energy
axis earns its place because the deployment question is energy, and because the
power term moves opposite to the intuition that int8 is uniformly cheaper. It is
not an independent axis and this README does not claim it is.

Thread counts: MobileNetV3-Small fp32 goes from 0.94 to 0.79 J/1k at 4 threads
(1.50× faster, 16% less energy) at batch 1, and from 6.64 to 3.06 J/1k
(2.19× faster, 54% less energy) at batch 32.
</details>

---

## Why you can trust these numbers

- **Pre-registered.** Rejection criteria were committed before any measurement.
  `bd06ff5` (26 Aug) added PROTOCOL.md and `bench/exclusion.py`; `1c9fa33` (2 Sep)
  is the first commit carrying results. Seven days apart, checkable with
  `git log bd06ff5..1c9fa33`.
- **One recipe.** Identical training recipe, preprocessing, split and seed
  protocol for every architecture. The recipe is hashed into every result row, so
  a changed recipe cannot masquerade as the old one.
- **Committed split.** EuroSAT ships no official split. Ours is deterministic,
  sha256-hashed and version-controlled, and CI verifies the hash every run.
- **Variance, not point estimates.** Five seeds per configuration, Student-t 95%
  confidence intervals, Welch's t-test with Holm-Bonferroni correction.
- **Nothing hand-typed.** Every published table and headline number is generated
  from `results/bench.jsonl`. CI regenerates them and fails on drift.
- **86 tests** against committed artefacts. No GPU, no dataset download.

<details>
<summary>Exclusions, and what was excluded</summary>

**4 of 300 windows were excluded**: two for latency p95/p50 > 1.50 (contention),
two for energy sample coverage below 0.95. None fall in the primary reporting
configuration, so no headline figure changes when they are removed, verified by
recomputing the summary both ways. Excluded rows stay in `results/bench.jsonl`
and are listed in `results/summary.json`. Nothing is deleted.

[PROTOCOL.md](PROTOCOL.md) records five deviations from the original
pre-registration, including two criteria that were never instrumented and the
fact that failing windows were not re-run.

Energy is reported gross, not baseline-subtracted. The idle baseline measured
over 338 s immediately after the matrix was 0.036 W, which is 0.78% of the
lowest-power window and less for every other one, well inside seed-to-seed
variation.
</details>

<details>
<summary>Training recipe and hardware</summary>

<!-- BEGIN:recipe -->
Every architecture is trained under this identical recipe. There is no
supported way to give one model a tuned recipe of its own.

| Setting | Value |
|---|---|
| augmentation | `['random_hflip', 'random_vflip', 'random_rot90']` |
| batch_size | `128` |
| early_stopping | `{'mode': 'max', 'monitor': 'val_acc', 'patience': 4, 'restore_best_weights': True}` |
| finetune | `full` |
| input_size | `64` |
| label_smoothing | `0.1` |
| lr | `0.0003` |
| max_epochs | `20` |
| norm_mean | `[0.485, 0.456, 0.406]` |
| norm_std | `[0.229, 0.224, 0.225]` |
| optimizer | `adamw` |
| schedule | `cosine` |
| warmup_epochs | `2` |
| weight_decay | `0.0001` |
<!-- END:recipe -->

<!-- BEGIN:hardware -->
All measurements in this repository come from ONE machine. Latency,
memory and energy figures are properties of the model AND this hardware;
they are not portable claims.

| Property | Value |
|---|---|
| CPU | Apple M2 |
| Cores | 8 physical / 8 logical |
| RAM | 8 GiB |
| OS | Darwin 23.6.0 (Darwin Kernel Version 23.6.0) |
| Python | 3.12.11 |
| PyTorch | 2.9.1 |
| ONNX Runtime | 1.29.0 |
| timm | 1.0.28 |
| NumPy | 2.5.2 |
| Measurement date (UTC) | 2026-09-01T17:09:50+00:00 |
| Power source during measurement | AC |
| macOS Low Power Mode | 0 (AC) |
| Training environment (affects no reported figure) | 2026-08-27T01:52:08+00:00, battery |

Split file: `splits/eurosat_split_seed42.csv`  
Split sha256: `b77443792ba4b11439ed220b3dea699ce61e48e0a0af49c51fb6a8cf43b0595d`  
Recipe hash: `746abf440ef1`

Inference is measured through ONNX Runtime's **CPU execution provider only**.
Apple's GPU (MPS) and CoreML providers are excluded deliberately, not merely
left unused: the question is what CPU-only hardware achieves. Training used
the GPU, which affects no reported figure -- training cost is not part of the
deployment claim being made.
<!-- END:hardware -->
</details>

```bash
make test      # run the suite against the committed artefacts
```

---

## Limitations

Most important first.

- **Single platform, and the ordering does not survive leaving it.** Every
  published figure comes from one Apple M2. CI re-times the seven committed graphs
  on x86 Linux each push
  ([`scripts/crossplatform_latency.py`](scripts/crossplatform_latency.py)) and the
  ordering changes: Spearman rho ≈ 0.68, not 1.0. Both fp32 graphs move sharply
  up on x86, and `mobilenetv3_small fp32`, 4th on the M2 at 1.40 ms, becomes the
  fastest measured, beating its own int8 form. Quantisation buys much less on x86.
  Those runs are indicative only: one pass on a shared runner, latency only. They
  are still enough to say the ranking here is an Apple M2 ranking.
- **Energy is estimated, not metered.** On-die CPU package power sampled at 200 ms
  and integrated over each window. `codecarbon` cannot cross-check it: it reads
  Intel RAPL, which Apple Silicon lacks, so it degrades to a hardcoded-TDP model
  whose output is a linear function of runtime, which is latency wearing a
  different unit. What the CPU-package figure leaves out is now bounded rather
  than merely disclosed: `powermetrics` also logged the GPU and ANE rails, and
  across 89,983 samples the CPU drew **99.80%** of on-die compute power, leaving
  0.20% in GPU and ANE
  ([`scripts/power_composition.py`](scripts/power_composition.py)). That also
  confirms the matrix really did run on the CPU. DRAM, display and PSU losses
  remain genuinely unmeasured, since no on-die counter sees them, so these are
  not wall-socket figures.
- **Scene leakage costs 0.3 to 0.8 pp**, measured rather than guessed. EuroSAT
  tiles are cut from larger Sentinel-2 scenes and the corpus carries no scene
  identifier, so the split is stratified by class but cannot be grouped by scene.
  [`scripts/leakage_check.py`](scripts/leakage_check.py) quantifies the result.
  <details><summary>How it was measured, and what it found</summary>

  *The folds are not separated.* Each test tile's cosine similarity to its nearest
  train tile is indistinguishable from the same statistic computed inside the
  train fold: median 0.6510 against a control of 0.6548, p99 0.9984 against
  0.9983. A test tile sits as close to the training data as a train tile sits to
  its own same-scene neighbours.

  *Near-duplicates are common.* 8.8% of test tiles have a train neighbour at
  cosine ≥ 0.99, and 88.2% of those pairs share a class, rising monotonically from
  42% at ≥ 0.90. That rise is what separates real duplication from two tiles of
  flat texture resembling each other.

  *The cost.* Dropping those tiles moves the committed EfficientNet-Lite0 int8
  graph from 97.33% to 97.08%, a fall of 0.26 pp. Dropping everything at ≥ 0.90
  gives 96.56%, a fall of 0.78 pp. This is a lower bound: it catches visible
  duplication, not same-scene tiles that happen to look different.

  Every model is affected equally, so the comparisons and the ranking stand. It is
  the absolute numbers that are not a geographic-generalisation claim.
  </details>
- **CO₂e rests on a stated assumption**: 481 gCO₂e/kWh, world average. Carbon
  scales linearly, so substituting your own grid is one multiplication. The
  recommended configuration's 0.42 g per million inferences becomes about 0.04 g
  at 50 gCO₂e/kWh and 0.61 g at 700. That range is wider than any difference this
  benchmark measures between models.
- **EuroSAT is near-saturated**, so architecture differences are small in absolute
  terms even when statistically reliable. That is itself the finding.
- **Geographic bias.** EuroSAT covers 34 European countries. Nothing here supports
  a claim about performance elsewhere, where land cover, agriculture, settlement
  morphology and phenology all differ.
- **RGB only.** EuroSAT's 13-band multispectral form is not benchmarked, and
  closing that gap is not a small change: the multispectral corpus is a separate
  download, a 13-channel stem cannot reuse the ImageNet-pretrained first
  convolution, and the full 25-run matrix would need retraining. Absolute
  accuracies would not be comparable to the ones here. The energy and latency
  ordering is driven by architecture rather than by ten extra input channels, so
  it would plausibly survive, but that is an expectation and not a measurement.


---

## The wider point

Green AI is usually argued at training time, where the headline numbers are
largest. But a model is trained once and served for years, so most of its
lifetime energy goes on inference, which is the part most benchmarks leave
unmeasured.

Here the entire accuracy spread across five architectures is 1.25 pp while energy
spans a factor of 19. Wherever that shape holds, reaching for the largest model
that fits is not a cautious default. It is a large and invisible energy bill for
a difference that may not survive contact with the deployment.

The contribution is not that EfficientNet-Lite0 wins, because on another chip it
may not. It is that the trade-off was made measurable under one honest recipe,
and that every number can be checked by anyone who clones the repository.
Efficiency only becomes a design decision once someone reports it.

---

## Repository map

`bench/` is the benchmark: `config.py` holds the zoo, the shared recipe and the
grid-intensity constant as a single source of truth, and every other module reads
from it. `scripts/` covers data prep, split generation, isolated memory
measurement, the cross-platform latency check and the leakage check. `tests/`
runs against committed artefacts, so it needs no GPU and no dataset. `splits/`,
`results/` and `web/` hold the committed split, the measurements and the demo
site.

Deeper documentation: [PROTOCOL.md](PROTOCOL.md) for the measurement protocol,
outcomes and deviations, and [DATASHEET.md](DATASHEET.md) following *Datasheets
for Datasets* (Gebru et al.).

---

## Licences and attribution

Code is MIT ([LICENSE](LICENSE)). Results data and the split file are CC-BY-4.0
([LICENSE-DATA](LICENSE-DATA)).

EuroSAT is distributed under the MIT licence:

> Helber, P., Bischke, B., Dengel, A., & Borth, D. *EuroSAT: A Novel Dataset and
> Deep Learning Benchmark for Land Use and Land Cover Classification.*

EuroSAT is derived from Copernicus Sentinel-2 imagery, provided under terms
granting free access, including reproduction, distribution and modification.

ESA WorldCover is *not* used here, so its attribution string is deliberately
omitted rather than included for completeness. Printing an attribution for data
one has not used is a false provenance claim.
