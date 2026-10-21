# Experiment design and interpretation

## What the detector measures

Every agent takes one spatial sample per frame and stores all four channel values.
Once its memory contains `CORR_WINDOW` observations, it evaluates the six unique
channel pairs with Pearson correlation. A detection occurs when `abs(r)` exceeds
the threshold. Constant channels contribute no detections. The three operator
names currently select no different behavior.

The engine evaluates all agents at once. Positions and memories live in shared
arrays: a ring buffer holds the latest `MEMORY` observations of every agent, and
one batched correlation covers every full window. Random draws happen in the same
order as the original per-agent loop, so batching changes speed, not results.
`Agent.discover` uses the same correlation function on a single window and remains
the reference implementation.

Every evaluated |r| is recorded, detected or not, in a histogram with 100 bins of
width 0.01, one per pair. Every two-decimal threshold is therefore a bin edge, and
detection rates, ROC curves and recalibrated thresholds can be computed exactly
after the run without re-simulating.

The shared graph is undirected because this estimator has no directional test.
Each pair retains its strongest recent absolute correlation from any agent, then
decays every frame. Edge weights are refreshed even when no new detection occurs;
weak edges are removed. With many agents testing six pairs, the strongest |r| is a
maximum over many chance comparisons, and false alarms alone keep every edge
alive. The animation therefore draws **operator consensus**, the share of agents
whose current window detects each pair, which does separate injected couplings
from chance. Counts include repeated observations of the same relationship by
multiple agents and across overlapping windows. They are not independent findings.

## The synthetic generator

### Driven dynamics (default since 0.5)

Each frame, for every channel `c`, in Fourier space:

```
F̂ ← FIELD_MEMORY · exp(−D_c · (2πk)² · m) · F̂  +  √(1 − FIELD_MEMORY²) · g(k) · ξ̂
```

`m` is the moderator factor, `D_c` = (0.10, 0.075, 0.09, 0.06), and `ξ` is fresh
standard normal noise. `g(k)` is a Gaussian filter that gives the forcing
unit variance and a spatial scale of `FIELD_CORRELATION_LENGTH` pixels. Then:

- the coupled arm adds `COUPLING_STRENGTH · m · F₁` to `F₀` (continuous);
- inside a coherence window, the coupled arm sets
  `F₃ ← (1 − COHERENCE_MIXING) · F₃ + COHERENCE_MIXING · F₂`;
- a gentle saturation `F ← F + 0.1 · (tanh F − F)` keeps the field nonlinear and bounded.

The initial field is a draw of the forcing texture itself, so there is no burn-in
transient. The field is stationary, changes on a 20-frame correlation time
(`1 / (1 − 0.95)`), and has spatial structure. Because each channel keeps relaxing
and being renewed, the imprint of a coherence window fades after the window closes.
The coupling is transient, as documented.

### Legacy dynamics (≤ 0.4, `--dynamics legacy`)

The legacy generator is retained bit for bit so that every earlier result can be
regenerated, but it should not be used for new work:

- **It freezes.** `0.92 · F + 0.08 · tanh(4F)` has gain 1.24 near zero and stable
  fixed points near ±1. Every pixel grows until it saturates. In the checked
  configuration, 91% of pixels are saturated by frame 50 and all of them by
  frame 99. The field's correlation with itself one window later approaches 1.00.
- **It has no spatial structure.** Its diffusion kernel `exp(−D · k² · 0.5 · m)` omits
  the (2π)² of the Laplacian, and with `D ≤ 0.002` it removes at most 0.065% of
  the Nyquist component per frame. Neighbouring pixels correlate at 0.00.
- **Its coherence coupling is permanent.** Repeated mixing during one window makes
  channel 3 converge to channel 2. The two then evolve under identical dynamics and
  stay identical (field correlation 1.000) for the rest of the run. In the 0.4
  benchmark, the seeds without any coherence window (47 and 49) show zero ch2–ch3
  difference, and most others show 90–96%.

## Synthetic paired control protocol

1. Choose the configuration and starting seed before reading the output.
2. Use at least `CORR_WINDOW` frames. Prefer several windows to allow evolution.
3. For each seed, run the coupled field and reset the RNG to that same seed before
   running the uncoupled field.
4. Disable only the two injected terms in the uncoupled arm: ch1 → ch0 and the
   coherence-window ch2 → ch3 mixing.
5. Retain all other dynamics, initial draws, forcing noise, moderator draws, and agent movements.
6. Compare all six pairs, including the four pairs without direct injected links.

The primary descriptive output is the mean paired difference in total detections.
Per-pair rates expose whether that difference is concentrated on the injected
relationships or spread across other pairs. Rates divide counts by
`AGENTS * (FRAMES - CORR_WINDOW + 1)`, the eligible windows for each pair; every
run also records `evaluated_windows`, the count the engine actually evaluated.

The engine restores the caller's RNG state after the study. Tests verify matching
agent paths and random schedules across arms, so rendering or layout randomness
cannot alter the comparison. Studies with too few frames or seeds outside the
supported range fail instead of emitting an uninformative success report.

## Seed-level statistics

Seeds are independent replicates of the generator. Agents and overlapping windows
are not: they observe one field. Every uncertainty statement therefore comes from
seeds, and agent or window counts are never used as a sample size.

- **Standard error** is the standard deviation of the per-seed rate differences
  divided by √(seeds).
- **Sign-flip p** tests whether the per-seed differences are symmetric about zero.
  Under that null, each seed's difference is as likely negative as positive. For up
  to 18 seeds all 2ⁿ sign assignments are enumerated exactly. Beyond that a seeded
  20,000-draw Monte Carlo estimate is used, counting the observed assignment.
- **Holm p** adjusts the six sign-flip p-values for family-wise error. With n seeds
  the smallest possible two-sided p is 2/2ⁿ, so the smallest Holm-adjusted p for
  six pairs is 12/2ⁿ. Fewer than 8 seeds cannot reach 0.05; plan studies accordingly.
- **AUC** is threshold-free. It is the probability that a coupled-arm window shows a
  stronger |r| than an uncoupled-arm window for the same pair, with ties within one
  histogram bin counted half. It is reported pooled over seeds and per seed.

These statistics describe this generator and configuration only. They are not
evidence about empirical data, and a different configuration is a different experiment.

## Detector calibration

The **nominal false-alarm rate** is what the threshold implies under the textbook
assumption of independent samples. For `n` independent samples of two independent
Gaussian channels, `r · √(n − 2) / √(1 − r²)` follows Student's t with `n − 2`
degrees of freedom. So `P(|r| > τ) = I_{1−τ²}((n − 2)/2, 1/2)`, the regularized incomplete beta
function. For `n = 50`, `τ = 0.32` this is 2.35%. A test checks this against the
t distribution and against the detector's own code on 40,000 independent windows.

The **observed false-alarm rate** is the uncoupled arm's detection rate pooled over
all pairs and seeds. Every uncoupled pair is unrelated by construction. The ratio
of the two is the inflation. The **effective samples per window** invert the
nominal formula: the number of independent samples that would produce the observed rate.

**Held-out recalibration** splits the seeds. Even-indexed seeds choose the lowest
two-decimal threshold whose pooled uncoupled false-alarm rate does not exceed the
nominal rate. Odd-indexed seeds, which played no part in that choice, report the
false-alarm rate and the sensitivity to each injected coupling at that threshold,
next to the sensitivity at the configured threshold.

## Field diagnostics

In synthetic runs the engine also measures the field directly, over every pixel:

- `pair_correlation_mean`: the pixelwise correlation of each channel pair, averaged
  over frames. This is the ground truth the agents are trying to recover.
- `window_memory`: the mean, over channels, of the field's correlation with its own
  state `CORR_WINDOW` frames earlier, sampled once per window. 1 means frozen.
- `neighbour_correlation`: the mean correlation between each pixel and its
  neighbour along each axis. 0 means no spatial structure.

## Checked example: eight paired seeds

The committed [JSON report](benchmarks/control-study.json) and [offline HTML companion](benchmarks/control-study.html) were generated with:

```bash
ions-x --quick --control-study 8 --frames 100 --agents 40 --field-res 32 \
  --seed 42 --output outputs/control-study.json
```

| Channel pair | Coupled | Uncoupled | Holm p | AUC |
| --- | ---: | ---: | ---: | ---: |
| 0–1 (continuous injected coupling) | 56.99% | 9.07% | 0.047 | 0.83 |
| 2–3 (coherence-window coupling) | 26.86% | 8.99% | 0.078 | 0.64 |
| 0–2 (no direct injected coupling) | 11.00% | 8.30% | 0.094 | 0.52 |
| 1–2 (no direct injected coupling) | 11.19% | 11.19% | 1.000 | 0.50 |

Field ground truth: ch0–ch1 correlates at 0.35 in the coupled field and −0.01 in the
uncoupled field. ch2–ch3 correlates at 0.16, averaged over all frames, in the coupled field.
Memory across one window is 0.00, and neighbouring pixels correlate at 0.82.
Calibration: nominal false alarms 2.35%, observed 8.76% (3.7×), about 30 effective
samples per 50-sample window. Held out, |r| ≥ 0.42 gives 2.08% false alarms.

The unchanged 1–2 rate is a built-in check: neither ablated term touches those
channels, and the sampling paths match. The small 0–2 rise is an indirect effect.
Adding channel 1 into channel 0 changes channel 0's texture and so its chance
agreement with channel 2, and Holm adjustment correctly keeps it from being reported
as a finding. These are descriptive results for seeds 42–49 and this configuration,
not general performance guarantees. The report retains every pair and every seed.
Its passport identifies the exact implementation and runtime used.

The 0.4 benchmark is kept as a [historical record](benchmarks/legacy-0.4/control-study.json)
and is regenerated exactly by the same command with `--dynamics legacy`.

## Baseline preset is a separate experiment

`--preset baseline` creates independent Gaussian series in all four channels. With a
CSV, measured channels are replaced while its reference channel, timestamps, and
covariates are retained.

The baseline threshold remains the legacy quantile of absolute correlations
between overlapping reference windows and a circularly shifted reference series.
That is a heuristic threshold, not family-wise error control or a calibrated
significance level. It does not substitute for the paired synthetic comparison,
and a white-noise baseline is not a suitable universal control for autocorrelated
empirical telemetry. The calibration section above shows why: the false-alarm rate
depends on how redundant consecutive samples are.

## Empirical data boundaries

The CSV importer requires each measured sensor channel and at least one numeric
value in each. It rejects infinities. Partial sensor gaps are filled forward then
backward and counted in `telemetry_quality`. Missing covariates default to zero.
Timestamps retain the legacy fill/synthetic-index behavior documented in README.

Standardization uses the entire series. Spatial projection is synthetic: frame `t`
of channel `c` is the standardized value times a fixed spatial basis, computed on
demand. These choices make the analysis offline and can create relationships not
present in the untransformed streams. The spatial basis, agent trajectory, and
moderator model are part of the experiment, not physical facts about the measurements.

The `consciousness_proxy` name is a historical schema identifier. REG telemetry
alone is not a measure or demonstration of consciousness. In synthetic mode the
fourth channel participates in an injected coupling; it must not be interpreted as
an independent control in that mode.

## Reproducibility contract

`main()` resets configuration to defaults before applying the experiment bundle,
seed, quick settings, explicit flags and `--dynamics`. Reports record the effective
configuration, the lab version, source SHA-256, input SHA-256 when present, Python
and dependency versions, backend, actual processed frames, and the settings that
could not influence the run. Output paths, timestamps, and longitudinal file IDs are
run-specific. The saved CSV itself is not claimed to be byte-reproducible.

`SimulationEngine.step(frame)` accepts only the next sequential frame, or an
idempotent repeat of its current frame. It rejects earlier frames, skipped frames,
and out-of-bounds indices. Animation uses a separate initializer that does not
advance state. Export HTML or GIF once per animation instance; construct a new
seeded run for another export format. The engine uses module-level configuration
and RNG, so callers must keep settings unchanged and run engines sequentially.

Headless mode and both render formats are tested against the same engine metrics.
Both committed benchmarks are regenerated exactly by the test suite. The GPU path is
optional and recorded, but CPU/GPU numerical identity is not promised. Tests in CI
exercise CPU execution.

## Report schemas

**Single-run schema 4.** `evaluation` holds `status`, `correlation_window`,
`opportunities_per_pair` (the agent-windows actually evaluated), and six
`association_rates`. An ineligible run has null rates and status
`insufficient_observations`; an eligible run with no detections has zero rates and
status `evaluated`. `field_diagnostics` holds the field measurements above for
synthetic runs and is null for telemetry.

**Paired-study schema 3.** It adds `dynamics`. Each arm of each run gains
`evaluated_windows`, `abs_r_histogram` (100 counts per pair) and
`field_diagnostics`. `paired_association_effects` gains `standard_error`,
`sign_flip_p`, `sign_flip_method` and `holm_p`. New top-level sections are
`detector_calibration`, `discrimination` (pooled and per-seed AUC), and
`field_ground_truth`. Existing counts, rates and arm records remain. Numeric
arrays are written on one line. Passports fingerprint both the engine and the
report implementation. Reports from earlier schemas still render.
