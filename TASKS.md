# Project Tasks

## Done

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

## Next UX Tasks

### 1. Live metrics dashboard

Stream discoveries and coherence factor to a lightweight live view during long empirical runs,
instead of only writing the sidecar at the end.

Acceptance criteria:
- A `--live` (or notebook widget) mode updates a small stats panel as frames render.
- Works without blocking the animation render.

### 2. Seed control

Add a `--seed` flag so runs can vary while remaining reproducible.

Acceptance criteria:
- `--seed N` sets the module RNG deterministically.
- Two runs with the same seed produce identical metrics sidecars.
- A test asserts seed reproducibility.

### 3. Package publish

Publish to PyPI so `pip install ions-x-deep-emergence-lab` and the `ions-x` command work.

Acceptance criteria:
- Build succeeds with `python -m build`.
- The `ions-x` entry point runs a quick simulation end to end.
