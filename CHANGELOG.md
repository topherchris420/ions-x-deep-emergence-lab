# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
