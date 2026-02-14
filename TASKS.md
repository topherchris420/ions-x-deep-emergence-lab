# Proposed Maintenance Tasks

## 1) Typo fix task
**Issue found:** The `README.md` title uses `IONS_X` while the repository/package naming elsewhere consistently uses `ions-x` / `ions_x`, which creates naming inconsistency for users copying commands and references.

**Task:** Standardize the project name in the README title to `IONS-X Deep Emergence Lab` (or another single canonical form used across the repo).

**Acceptance criteria:**
- README title and first paragraph use one canonical project name.
- The chosen name matches the clone URL/repository naming convention.

## 2) Bug fix task
**Issue found:** `ions_x_deep_emergence.py` executes plotting/animation side effects at import time (`FuncAnimation`, `display(...)`). This prevents clean reuse as an importable module and can break non-notebook execution workflows.

**Task:** Move runtime execution into a `main()` function and guard it with `if __name__ == "__main__":`.

**Acceptance criteria:**
- Importing `ions_x_deep_emergence` does not open figures or start animation.
- Running `python ions_x_deep_emergence.py` still produces the same visualization behavior.

## 3) Code comment/documentation discrepancy task
**Issue found:** The README installation section has a malformed markdown code block (the clone command fence is never closed before `### 1. Standard Installation`), so rendered docs are broken and steps appear inside a shell block.

**Task:** Repair markdown fencing and provide a complete, linear setup flow (clone, install deps, run command).

**Acceptance criteria:**
- All markdown code fences in README are properly opened/closed.
- Setup instructions render correctly on GitHub.
- A run command is documented after dependency installation.

## 4) Test improvement task
**Issue found:** There are currently no automated tests for core behavior (agent discovery logic, environment modulation, and CPU fallback path).

**Task:** Add a small `pytest` suite focused on deterministic unit coverage:
- `Agent.discover()` threshold behavior.
- `EnvironmentalModerators.is_coherence_active()` window logic.
- Import/runtime separation (ensuring no animation side effects on import after bug fix).

**Acceptance criteria:**
- `pytest` runs locally with at least 3 focused tests.
- Tests are deterministic by setting the random seed.
- Tests cover both positive and negative cases for discovery/coherence behavior.
