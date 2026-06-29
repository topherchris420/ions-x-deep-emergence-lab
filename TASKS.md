# Project Tasks

## Done

- Standardized the project name as `IONS-X Deep Emergence Lab`.
- Moved runtime behavior behind `main()` and `if __name__ == "__main__"`.
- Added deterministic tests for agent discovery, coherence windows, and import safety.
- Added a quick command-line run path with saved HTML output.
- Added dependency files for runtime and test setup.
- Added a README preview asset and clearer onboarding docs.

## Next UX Tasks

### 1. Add a real generated demo GIF

Create a short animation from a known quick run and place it near the top of the README.

Acceptance criteria:
- The GIF is generated from the actual simulation, not a conceptual mockup.
- README first screen shows the project name, one-sentence value prop, and demo media.

### 2. Add experiment presets

Add named presets such as `quick`, `arv`, `coherence`, and `dense-agents` so users do not need to tune raw numbers first.

Acceptance criteria:
- `python ions_x_deep_emergence.py --preset quick` works.
- Presets are documented in README.
- Unit tests cover at least two presets.

### 3. Export metrics beside visual output

Save discoveries, coherence frames, and final graph edges as JSON or CSV.

Acceptance criteria:
- Running with `--output outputs/demo.html` also writes `outputs/demo.metrics.json`.
- Metrics include frame count, agent count, backend, total discoveries, and coherence frames.
- Tests verify the metric file shape.

### 4. Add a guided notebook

Create a notebook that walks through the field, operators, moderators, and graph output in small cells.

Acceptance criteria:
- Notebook runs top-to-bottom in Colab or local Jupyter.
- It uses quick settings by default.
- README links to the notebook from the quickstart section.