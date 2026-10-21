# Contributing to IONS-X Deep Emergence Lab

Thanks for your interest. This is a small, deliberately readable research sandbox, so
the contribution bar is simple: keep it runnable, keep it deterministic, and keep it honest
about what is simulation versus claim.

## Development setup

```bash
git clone https://github.com/topherchris420/ions-x-deep-emergence-lab.git
cd ions-x-deep-emergence-lab
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

## Before you open a pull request

Run the same checks CI runs:

```bash
ruff check .
python -m pytest -q
```

Then confirm rendering and a paired experiment both work end to end:

```bash
python ions_x_deep_emergence.py --quick --frames 8 --agents 20 --field-res 32 --output outputs/check.html
python ions_x_deep_emergence.py --quick --control-study 2 --agents 4 --field-res 8 --output outputs/control.json
```

## Guidelines

- **Determinism.** Tests must not depend on wall-clock time or un-seeded randomness. The
  simulation seeds NumPy via `CFG.SEED`; keep new randomness routed through the module `rng`
  or an injected `RandomState`.
- **One concern per PR.** Separate feature work from formatting-only churn.
- **Test new behavior.** New CLI options, presets, or output formats need at least one test.
  Mirror the existing style in `tests/` (load the module via `importlib`, snapshot/restore
  `CFG` when you mutate it).
- **Honesty about scope.** This lab explores hypotheses; it does not prove nonlocal effects.
  Keep documentation framed as "repeatable experiment," not "scientific result."
- **Style.** Ruff enforces formatting and imports. Physics/field symbols such as `F` and `I`
  are intentional and are exempted in `pyproject.toml`.

## Adding an experiment preset

1. Add an entry to `EXPERIMENTS` in `ions_x_deep_emergence.py` mapping `CFG` attribute names
   to values. A short comment should say what the preset is *for*.
2. Document it in the README's experiment-presets table.
3. Add a test asserting the preset applies (see `tests/test_experiments_and_outputs.py`).

## Changing the generator or detector

The committed benchmark is evidence, and the test suite regenerates it exactly. If a change
alters results on purpose, regenerate it in the same pull request and say why in the changelog:

```bash
ions-x --quick --control-study 8 --frames 100 --agents 40 --field-res 32 \
  --seed 42 --output docs/benchmarks/control-study.json
```

`--dynamics legacy` must keep reproducing `docs/benchmarks/legacy-0.4/` bit for bit; never edit
that directory. Draw any new randomness identically in both arms, never conditionally on the
coupling, so the arms stay paired. When you tune a parameter, check the report's field
diagnostics and calibration sections, not only the detection rates.

## Experiment integrity

Read [the experiment protocol](docs/experiment-design.md) before modifying field
dynamics or detection. Keep paired random schedules matched; do not let rendering
consume simulation randomness. `main()` resets configuration; lower-level engine
callers own configuration and must run sequentially. Treat detections as dependent
observations, not independent experiments: seeds are the only independent units, and
statistics must come from them. Keep the shared detector and inactive legacy settings
accurately described.
