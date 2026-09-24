# Experiment design and interpretation

## What the detector measures

Every agent takes one spatial sample per frame and stores all four channel values.
Once its memory contains `CORR_WINDOW` observations, it evaluates the six unique
channel pairs with Pearson correlation. A detection occurs when `abs(r)` exceeds
the threshold. Constant channels contribute no detections. The three operator
names currently select no different behavior.

The shared graph is undirected because this estimator has no directional test.
Each pair retains its strongest recent absolute correlation, then decays every
frame. Edge weights are refreshed even when no new detection occurs; weak edges
are removed. Counts include repeated observations of the same relationship by
multiple agents and across overlapping windows. They are not independent findings.

## Synthetic paired control protocol

1. Choose the configuration and starting seed before reading the output.
2. Use at least `CORR_WINDOW` frames. Prefer several windows to allow evolution.
3. For each seed, run the coupled field and reset the RNG to that same seed before
   running the uncoupled field.
4. Disable only `F[0] += 0.045 * F[1] * env_factor` and the coherence-window mixture
   `F[3] = 0.7 * F[3] + 0.3 * F[2]` in the uncoupled arm.
5. Retain all other dynamics, initial draws, moderator draws, and agent movements.
6. Compare all six pairs, including the four pairs without direct injected links.

The primary descriptive output is the mean paired difference in total detections.
Per-pair rates expose whether that difference is concentrated on the injected
relationships or spread across other pairs. Rates divide counts by
`AGENTS * (FRAMES - CORR_WINDOW + 1)`, the eligible windows for each pair.

The report includes each run, its seed, both arms, and coherence frames. The engine
restores the caller's RNG state after the study. Tests verify matching agent paths
and random schedules across arms, so rendering/layout randomness cannot alter the
comparison. Studies with too few frames or seeds outside the supported range fail
instead of emitting an uninformative success report.

Do not interpret the difference as a p-value or evidence for real-world causation.
The study is a sensitivity check against known changes in a synthetic generator.
Increasing the number of agents does not create independent experimental units.
A future inferential protocol needs an explicitly justified null model, independent
replicates, effect uncertainty, held-out evaluation, and multiple-testing handling.

## Checked example: eight paired seeds

The committed [report](benchmarks/control-study.json) was generated with:

```bash
ions-x --quick --control-study 8 --frames 100 --agents 40 --field-res 32 \
  --seed 42 --output outputs/control-study.json
```

| Channel pair | Coupled detection rate | Uncoupled detection rate |
| --- | ---: | ---: |
| 0–1 (continuous injected coupling) | 100.00% | 7.52% |
| 2–3 (coherence-window coupling) | 60.32% | 5.20% |
| 1–2 (no direct injected coupling) | 5.28% | 5.28% |

The unchanged 1–2 rate is also a useful check: those two channels are unaffected
by either ablated term and the sampling paths match. These are descriptive results
for seeds 42–49 and this configuration, not general performance guarantees.
The report retains the other three pairs and all per-seed values. Its passport
identifies the exact implementation and runtime used.

## Baseline preset is a separate experiment

`--preset baseline` creates independent Gaussian series in all four channels. The
previous three-zero-channel baseline could not meaningfully exercise cross-channel
correlation detection. With a CSV, measured channels are replaced while its
reference channel, timestamps, and covariates are retained.

The baseline threshold remains the legacy quantile of absolute correlations
between overlapping reference windows and a circularly shifted reference series.
That is a heuristic threshold, not family-wise error control or a calibrated
significance level. It does not substitute for the paired synthetic comparison,
and a white-noise baseline is not a suitable universal control for autocorrelated
empirical telemetry.

## Empirical data boundaries

The CSV importer requires each measured sensor channel and at least one numeric
value in each. It rejects infinities. Partial sensor gaps are filled forward then
backward and counted in `telemetry_quality`. Missing covariates default to zero.
Timestamps retain the legacy fill/synthetic-index behavior documented in README.

Standardization uses the entire series. Spatial projection is synthetic. These
choices make the analysis offline and can create relationships not present in
untransformed streams. The spatial basis, agent trajectory, and moderator model
are part of the experiment, not physical facts about the measurements.

The `consciousness_proxy` name is a historical schema identifier. REG telemetry
alone is not a measure or demonstration of consciousness. In synthetic mode the
fourth channel participates in an injected coupling; it must not be interpreted as
an independent control in that mode.

## Reproducibility contract

`main()` resets configuration to defaults before applying the experiment bundle,
seed, quick settings, and explicit flags. Reports record the effective configuration,
source SHA-256, input SHA-256 when present, Python and dependency versions, backend,
and actual processed frames. Output paths, timestamps, and longitudinal file IDs
are run-specific. The saved CSV itself is not claimed to be byte-reproducible.

`SimulationEngine.step(frame)` accepts only the next sequential frame, or an
idempotent repeat of its current frame. It rejects earlier frames, skipped frames,
and out-of-bounds indices. Animation uses a separate initializer that does not
advance state. Export HTML or GIF once per animation instance; construct a new
seeded run for another export format. The engine uses module-level configuration
and RNG, so callers must keep settings unchanged and run engines sequentially.

Headless mode and both render formats are tested against the same engine metrics.
The GPU path is optional and recorded, but CPU/GPU numerical identity is not
promised. Tests in CI exercise CPU execution.
