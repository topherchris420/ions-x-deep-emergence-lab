# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-10-21

The synthetic generator, the detector and the report were audited against ground truth
measured directly from the field. The new default measures what the documentation
always claimed; the old behavior remains reproducible bit for bit.

### Fixed
- **The synthetic field froze.** The legacy update had gain 1.24 near zero, so every pixel
  saturated near +-1 within about 60-100 frames and the "changing field" became a fixed pattern.
- **The field had no spatial structure.** The legacy diffusion kernel omitted (2*pi)^2 and
  attenuated the finest scale by at most 0.065% per frame; neighbouring pixels correlated at 0.00.
- **"Coherence windows only" coupling was permanent.** One window copied channel 2 into
  channel 3 for the rest of the run, producing the bimodal 0%/96% ch2-ch3 seeds of 0.4.
- **The association map could not distinguish signal from chance.** The shared graph keeps the
  strongest |r| from any agent, which false alarms alone keep high for every pair. The animation
  now draws operator consensus: the share of agents detecting each pair this frame.
- Telemetry projection materialized every frame (about 5 GB for 20,000 rows at the default
  resolution). Frames are now projected on demand with identical values.
- Coherence lookups were O(events) per call and empirical runs can log an event every frame,
  making long CSV runs quadratic. Lookups now bisect a sorted list.

### Added
- **Driven-dissipative field** (`--dynamics driven`, the new default): damped, diffusing through
  a true Laplacian, renewed by spatially correlated forcing that both arms share. Its parameters
  (`FIELD_MEMORY`, `FIELD_CORRELATION_LENGTH`, `COUPLING_STRENGTH`, `COHERENCE_MIXING`) are recorded.
- `--dynamics legacy` reproduces 0.4 bit for bit. The 0.4 benchmark is kept under
  `docs/benchmarks/legacy-0.4/` and a test regenerates it exactly.
- **Field diagnostics**: pixel-level pair correlations, memory across one window, and neighbour
  correlation, in headless reports and for every arm of a paired study.
- **Detector calibration**: the textbook false-alarm rate for the window and threshold, the rate
  the uncoupled arm actually produced, their ratio, the effective independent samples per window,
  and a threshold recalibrated on even seeds and evaluated on held-out odd seeds.
- **Seed-level inference**: exact sign-flip tests over seeds with Holm adjustment across all six
  pairs, plus standard errors. Agents and windows are still never treated as replicates.
- **Threshold-free discrimination**: |r| histograms for every evaluated window and pair, pooled
  and per-seed AUC, and ROC curves in the report.
- The offline report gains a per-seed effect plot, calibration and held-out results, ROC small
  multiples, and a field ground-truth table, all inline SVG with no scripts.
- The animation shows operator positions, a stable colour scale, and coherence-window state.
- `__version__`, recorded in every passport and used as the single source of the package version.

### Changed
- The engine samples, stores and correlates all agents as one batch per frame. Random draws keep
  the original order, so results are identical, about eight times faster.
- Engine agents keep `pos`, `memory`, `discover` and `observe`; their positions and memories are
  views into shared arrays.
- Single-run reports are schema 4: `opportunities_per_pair` counts the windows actually evaluated,
  and `field_diagnostics` is added. Paired studies are schema 3. Numeric JSON arrays are written on one line.
- Passports list inactive settings for the actual run (generator settings are inactive for
  telemetry and for legacy dynamics) and record the scipy version.
- `TelemetryTargetField` stores `signals` and `bases`; `fields` is materialized on request.

## [0.4.0] - Inspectable experiment evidence

- Added offline HTML companions for paired studies, preserving all six pairs and every seed.
- Added per-pair mean effects, observed ranges, and positive/zero/negative seed counts.
- Distinguished insufficient observations from evaluated runs with zero detections.
- Added report-source provenance, boundary regressions, and wheel-build validation.
- Separate sequential simulation from rendering; process each frame once in HTML,
  GIF, and new headless JSON mode. Close rendered figures after export.
- Add paired multi-seed synthetic coupling ablations with per-pair rates and complete
  run records. Replay matching random draws and explicitly distinguish descriptive
  comparisons from statistical significance.
- Correct undirected association semantics and stale graph weights; stabilize graph
  layout and refresh the observation dashboard.
- Record configuration, source/input hashes, runtime versions, actual frames, unique
  associations, and sensor-imputation counts. Isolate CLI configuration between runs.
- Reject missing/invalid sensor channels and infinities; make baseline channels
  independently variable; prevent longitudinal filename collisions.
- Vectorize pairwise Pearson evaluation with safe constant-channel handling.
- Document inactive legacy settings, shared operator behavior, offline preprocessing,
  heuristic calibration, and limits of scientific interpretation.

## [0.3.0] - 2026-08-20

### Added
- **Deterministic seed control** via `--seed N` (and `set_seed()`). Sets module RNG deterministically,
  ensuring identical metrics sidecars across runs with matching seeds and reproducible empirical experiments.
- **Live metrics dashboard** via `--live`. Streams real-time frame progress, cumulative discoveries,
  active coherence factor, REG variance deviation, and multi-scale sensor anomaly ratios without blocking animation rendering.
- **Dedicated CLI entry point** (`cli()`) preventing return-object exit code issues when running `ions-x`.
- **Packaging build support and modern SPDX license specification** in `pyproject.toml`, fully validated with `python -m build`.
- **Test suite additions** covering seed reproducibility, seed variance, live dashboard stdout streaming, and CLI UX.

### Changed
- Refactored `CFG` class attributes to use immutable tuples for mutable defaults (`AGENT_TYPES`, `LAG_FRAMES`), achieving 100% clean ruff compliance.
- Upgraded `build_run_summary` and metrics sidecar schema to record the active simulation `seed`.
- Cleaned up redundant casts in `RealWorldModerator` and `LongitudinalMetricsRecorder`.

## [0.2.0] - 2026-07-11

### Added
- **Named experiment presets** via `--experiment {balanced,quick,arv,coherence,dense-agents}`.
  Each preset is a documented parameter bundle so users can start from a meaningful
  configuration without tuning raw numbers. Explicit flags (`--frames`, `--agents`,
  `--field-res`) always override the preset.
- **GIF output.** `--output run.gif` renders a shareable clip via Pillow; any other
  suffix still writes the interactive HTML animation. `--fps` controls the frame rate.
- **Metrics sidecar.** Every run writes `<output>.metrics.json` beside the animation with
  frame/agent counts, backend, total discoveries, discoveries per operator type, coherence
  frames, and the per-frame discovery-rate history. Disable with `--no-metrics-sidecar`.
- **Guided notebook** at `notebooks/quickstart.ipynb` that walks through the field,
  operators, moderators, and emergent graph with quick settings.
- **Generated demo GIF** (`docs/assets/demo.gif`) produced from a real quick run, shown at
  the top of the README.
- **Packaging and tooling.** `pyproject.toml` with project metadata, an `ions-x` console
  entry point, ruff configuration, and pytest configuration.
- **Continuous integration.** `.github/workflows/ci.yml` runs ruff plus the test suite on
  Python 3.10-3.12 and executes a quick end-to-end smoke render.
- **Contributor guide** (`CONTRIBUTING.md`).

### Changed
- Modernized type hints across the module (`Optional[X]` -> `X | None`, `Dict`/`List`/`Tuple`
  -> builtins) and cleaned imports; the module is now ruff-clean.
- `save_animation_html` is retained as a backward-compatible alias for the new
  `save_animation`, which dispatches on the output suffix.
- README documents experiment presets, GIF output, and the metrics sidecar.

### Notes
- No changes to the simulation's numerical behavior: the synthetic, baseline, and empirical
  presets produce the same fields and discoveries as before for identical settings.
