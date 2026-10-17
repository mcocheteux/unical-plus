# KITTI-C2L and sequence calibration assessment

Assessment, 2026-10-17. The planned three-seed dataset, implementation and mean
sequence comparisons are complete, including the longer unchanged-main controls.
These experiments assess an optimized training recipe, not a reproduction of the
published UniCal recipe.

## Decision

**KITTI-C2L supports useful calibration with unchanged main.** With longer
training, it improves both metrics versus usual KITTI and identity correction on
both common held-out protocols across seeds42/43/44, including each test drive.
However, the reproduced usual-KITTI baseline remains weak, so this does not
establish parity with advertised UniCal accuracy.

**Sequence inference improves accuracy modestly.** Three-frame feature averaging
improves both metrics using the same frame-trained checkpoint across three seeds
on both training corpora. Dedicated three-frame mean training does not establish
an accuracy gain and uses three times the encoder exposure.

**Broad parity for the frozen perturbation policy is unsupported.** With the
improved branch under the matched short recipe, frozen C2L performs better on
native targets but worse beyond the 10% margin on usual-relative targets.
Regenerating usual-style perturbations on the same C2L frames meets the 10%
noninferiority screen under both protocols. This supports the frame/ground-truth
content under that policy, rather than parity of the frozen targets. These
results do not justify advertising the frozen dataset as a validated reference
benchmark matching published UniCal accuracy.

## Design and limits

- Unchanged baseline: main commit `7c16fd81e4889d6f16076b1ec5b58835d876807b`.
- Three paired training seeds: 42, 43, 44. Checkpoints selected by validation loss.
- All configured usual-KITTI training drives versus full C2L stage 5, with C2L
  sequence 07 held out for validation. Shared test sequences are 09 and 10:
  2,792 frame anchors, or 2,604 valid window anchors for temporal comparisons.
- Frame training uses 256px inputs, 8,192 loss points sampled after dense depth-map
  construction, BF16 mixed precision, batch 8, four spawned workers, pinned
  memory, uniform learning rate 3e-5 and one warmup epoch.
- Short frame runs process 185,070 encoder frames on C2L and 198,978 on usual
  KITTI. Whole-epoch rounding gives usual training approximately 7.5% more
  exposure. Dedicated window T1/T3 runs have equal updates and 3x encoder
  exposure for T3.
- Native C2L targets use frozen dual-rig errors. Relative targets use usual-style
  camera-relative errors. Their distributions differ; report both separately.
- Intervals use 5,000 paired bootstrap draws over training seeds and shared
  temporal blocks/windows. Conclusions are conditional on the held-out drives.
  Noninferiority requires the paired upper bound for candidate minus 1.10 times
  reference to be below zero for each metric.
- Report translation MAE in centimetres and geodesic rotation error in degrees.
  Geodesic rotation is not interchangeable with the paper's Euler rotation MAE.
- C2L training sequence 08 overlaps the usual paper-alpha test drive. It cannot
  be used as an unseen test for models trained on full C2L.

## Matched short-recipe dataset comparisons across three seeds

Means are translation cm / geodesic degrees. Lower is better.

| Implementation and targets | Usual KITTI | Frozen C2L | Finding |
| --- | ---: | ---: | --- |
| Unchanged main, native | 6.8449 / 1.3298 | 6.9364 / 1.3269 | Both within 10%; weak identity-level baselines |
| Unchanged main, relative | 5.2190 / 0.9556 | 5.3051 / 0.9587 | Both within 10%; weak identity-level baselines |
| Improved branch, native | 4.4919 / 0.6524 | 4.2528 / 0.6148 | C2L improves both metrics |
| Improved branch, relative | 3.2897 / 0.4555 | 3.7726 / 0.5199 | C2L worsens both beyond the 10% margin |

The improved branch's frozen-C2L relative deficit is 14.68% in translation and
14.14% in rotation. Both deficits exceed the tolerance with paired uncertainty,
and repeat separately on drives 09 and 10. Main's identity reference is
6.6925 / 1.3287 on native targets and 5.0536 / 0.9542 on relative targets;
matching those weak baselines does not establish useful calibration parity.

With usual-style online perturbations regenerated on C2L, the improved branch
reaches 4.4169 / 0.6436 on native targets and 3.2632 / 0.4368 on relative targets.
Both metrics meet 10% noninferiority versus usual KITTI under both protocols,
overall and on each held-out drive. No overall improvement versus usual KITTI is
established. Training and validation perturbation policies change together, so
this does not isolate training perturbations as the sole cause.

## Implementation and temporal effects

Under matched short recipes, the improved single-frame implementation beats
unchanged main on both datasets
and both protocols across three seeds. On C2L, native translation/rotation errors
fall by 38.7% / 53.7%, and relative errors by 28.9% / 45.8%. This is an
implementation bundle, including input normalization, projection geometry and
encoder training-mode fixes; it is not an isolated temporal effect.

Inference-only feature averaging uses the same frame-trained checkpoint for T1
and T3 within each seed, CPU FP32, and identical window anchors and targets:

| Training corpus | T1 | T3 mean | Finding |
| --- | ---: | ---: | --- |
| C2L | 4.3470 / 0.6343 | 4.1036 / 0.5944 | Both improve, including each drive |
| Usual KITTI | 4.4792 / 0.6347 | 4.3875 / 0.6110 | Both improve, including each drive |

Dedicated mean training instead gives T1 5.9607 / 0.9667 versus T3
5.9956 / 0.9378. Neither metric establishes improvement across three seeds.
Both meet the 10% tolerance, but T3 consumes three times the encoder exposure.
The capacity-matched Transformer pilot also failed to establish a useful gain;
that pilot is single-seed evidence.

## Longer unchanged-main controls across three seeds

Every C2L arm completed 50 epochs, 925,350 encoder frames and 115,700 optimizer
updates. Every usual-KITTI arm completed 61 epochs, 933,666 encoder frames and
116,754 updates. Whole-epoch rounding gives usual training approximately 0.9%
more exposure. Both update count and cosine-schedule horizon change versus the
short runs; this does not isolate training duration alone.

| Held-out targets | Usual KITTI | Frozen C2L | C2L minus usual paired 95% intervals |
| --- | ---: | ---: | --- |
| Native | 7.2228 / 1.3284 | 4.0628 / 0.7746 | [-4.1324, -2.3403] cm / [-0.5986, -0.5135] degrees |
| Relative | 5.6084 / 0.9543 | 3.5458 / 0.6061 | [-3.1650, -1.1924] cm / [-0.3870, -0.3095] degrees |

C2L improves both metrics versus longer usual training, short C2L training and
identity under both protocols, overall and separately on drives09/10. This
resolves the weak-learning concern from short unchanged-main C2L runs: the
frozen data supports useful calibration with this implementation and schedule.

Longer usual-KITTI training establishes no improvement versus short training or
identity for either metric under either protocol. Relative translation is worse
than identity, with an increase interval of [0.0047, 1.8244] cm. Translation
noninferiority within 10% versus short training or identity is not established
under either protocol. The usual baseline remains a weak comparator.

The audit loaded all six best and final recovery checkpoints on CPU, verified
optimizer/scheduler state and completed updates, checked every validation epoch,
matched initial weights within seeds and against short controls, and recomputed
controller aggregates independently. Numeric reductions agree within 1e-12;
report structure and metadata match exactly. Validation selected C2L epochs
45/34/40 and usual epochs60/53/25 for seeds42/43/44, respectively (one-based).
Validation curves were non-monotonic, especially C2L seed43; these are results of
validation-selected checkpoints, not proof of stable end-of-schedule convergence.

The implementation improvements reported above use matched short recipes.
Comparing them directly with these longer main runs would confound implementation
with training budget. The temporal inference comparison isolates fusion by
using the same checkpoint and anchors within each paired seed.

## Evidence and reproduction

The tracked `assessment_results.json` provides a compact numeric evidence summary
with source-report hashes. `comparison_protocol.json`, `run_comparison.py`,
`evaluate_comparison.py` and `paired_analysis.py` define the training, evaluation
and comparison machinery. Local experiment outputs are intentionally ignored by
Git. `logs/comparison/current-assessment-index.json` indexes report paths and
SHA-256 hashes; reports retain prediction-source hashes, paired intervals,
per-drive results and control audits. Reproduction requires KITTI data and the
frozen C2L snapshot; no private producer package is required at runtime.

Primary local reports are `main-dataset-assessment-seeds42-44.json`,
`strong-dataset-assessment-uniformlr-seeds42-44.json`,
`aligned-perturbation-assessment-seeds42-44.json`,
`implementation-{c2l,usual}-assessment-seeds42-44.json`,
`cpu-window-inference-{c2l,usual}-assessment-seeds42-44.json`,
`temporal-mean-uniformlr-assessment-seeds42-44.json` and
`long-main-assessment-seeds42-44.json`, under `logs/comparison`.
