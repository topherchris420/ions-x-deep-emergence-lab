# Project Tasks

## Done

- Added a renderer-independent sequential engine, headless JSON runs, and paired
  multi-seed synthetic coupling studies with a committed eight-seed reference report.
- Fixed duplicate animation steps, directed-correlation semantics, decaying graph
  weights, dark-render contrast, actual frame counts, and configuration leakage.
- Added experiment passports, input checks, imputation counts, nondegenerate baseline
  channels, collision-resistant longitudinal outputs, and 24 regression cases.

- Standardized the project name as `IONS-X Deep Emergence Lab`.
- Moved runtime behavior behind `main()` and `if __name__ == "__main__"`.
- Added deterministic tests for agent discovery, coherence windows, and import safety.
- Added a quick command-line run path with saved HTML output.
- Added dependency files for runtime and test setup.
- Added a README preview asset and clearer onboarding docs.
- **Generated a real demo GIF** (`docs/assets/demo.gif`) from an actual quick run and placed it
  at the top of the README. (Was "Next UX Task 1".)
- **Added named experiment presets** (`--experiment quick|arv|coherence|dense-agents|balanced`),
  documented in the README, covered by unit tests. (Was "Next UX Task 2".)
- **Exported metrics beside visual output**: every run writes `<output>.metrics.json` with frame
  count, agent count, backend, total discoveries, per-type discoveries, and coherence frames;
  tests verify the shape. (Was "Next UX Task 3".)
- **Added a guided notebook** (`notebooks/quickstart.ipynb`) that runs top-to-bottom with quick
  settings; the README links to it. (Was "Next UX Task 4".)
- Added packaging (`pyproject.toml`, `ions-x` entry point), ruff configuration, CI
  (`.github/workflows/ci.yml`: lint + tests on 3.10-3.12 + smoke render), a `CHANGELOG.md`, and a
  `CONTRIBUTING.md`.
- Modernized type hints and made the module ruff-clean.
- **Added live metrics dashboard** (`--live`): real-time streaming stats for discoveries, coherence factor,
  REG variance deviation, and multi-scale sensor anomaly ratios in console and notebook handles during rendering. (Was "Next UX Task 1".)
- **Added deterministic seed control** (`--seed N`): seedable module RNG, identical metrics sidecar reproduction across runs,
  and comprehensive test coverage. (Was "Next UX Task 2".)
- **Validated packaging and build** (`python -m build` & `ions-x` CLI): verified standalone build and console script execution. (Was "Next UX Task 3".)

## Next UX Tasks

### 1. Interactive Web UI / Streamlit App

Build a lightweight web interface for interactive parameter tuning, CSV upload, and real-time graph visualization.

### 2. Multi-Target Channel Expansion

Support dynamic target channel definitions beyond the default 4-channel schema for custom multi-sensor arrays.
