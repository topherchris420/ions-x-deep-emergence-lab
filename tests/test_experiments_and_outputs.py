"""Tests for experiment presets, the GIF output path, and the metrics sidecar."""

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'ions_x_deep_emergence.py'
MODULE_SPEC = importlib.util.spec_from_file_location('ions_x_deep_emergence_outputs', MODULE_PATH)


def load_module():
    module = importlib.util.module_from_spec(MODULE_SPEC)
    MODULE_SPEC.loader.exec_module(module)
    return module


CFG_KEYS = ('FIELD_RES', 'AGENTS', 'FRAMES', 'SAMPLE_PER_FRAME', 'MEMORY', 'CORR_WINDOW', 'DISCOVER_THRESH', 'CONFIDENCE_DECAY', 'LAG_FRAMES')


def snapshot_cfg(sim):
    return {key: getattr(sim.CFG, key) for key in CFG_KEYS}


def restore_cfg(sim, values):
    for key, value in values.items():
        setattr(sim.CFG, key, value)


def test_every_experiment_preset_only_sets_known_cfg_attributes():
    sim = load_module()
    for name, overrides in sim.EXPERIMENTS.items():
        for key in overrides:
            assert hasattr(sim.CFG, key), f'experiment {name!r} sets unknown CFG attribute {key!r}'


def test_apply_experiment_applies_named_bundle():
    sim = load_module()
    original = snapshot_cfg(sim)
    try:
        sim.apply_experiment('dense-agents')
        assert sim.CFG.AGENTS == 800
        assert sim.CFG.FIELD_RES == 96
        assert sim.CFG.FRAMES == 200
    finally:
        restore_cfg(sim, original)


def test_apply_experiment_none_is_a_noop():
    sim = load_module()
    original = snapshot_cfg(sim)
    try:
        sim.apply_experiment(None)
        assert snapshot_cfg(sim) == original
    finally:
        restore_cfg(sim, original)


def test_apply_experiment_rejects_unknown_name():
    sim = load_module()
    with pytest.raises(ValueError, match='Unknown experiment'):
        sim.apply_experiment('does-not-exist')


def test_explicit_flags_override_experiment_bundle():
    sim = load_module()
    original = snapshot_cfg(sim)
    try:
        args = sim.parse_args(['--experiment', 'dense-agents', '--agents', '11', '--frames', '5'])
        sim.apply_runtime_options(args)
        # dense-agents wants 800 agents / 200 frames, but explicit flags win.
        assert sim.CFG.AGENTS == 11
        assert sim.CFG.FRAMES == 5
        # Untouched-by-flag values still come from the bundle.
        assert sim.CFG.FIELD_RES == 96
    finally:
        restore_cfg(sim, original)


def test_parser_rejects_unknown_experiment_choice():
    sim = load_module()
    with pytest.raises(SystemExit):
        sim.parse_args(['--experiment', 'not-a-real-preset'])


def test_build_run_summary_has_expected_shape():
    sim = load_module()
    metrics = sim.PerformanceMetrics()
    metrics.log_discovery('perceiver')
    metrics.log_discovery('perceiver')
    metrics.log_frame(frame=0, discoveries=2, env_factor=1.1, is_coherence=True)
    result = sim.RunResult(
        output_path=Path('outputs/demo.html'),
        frames=1,
        agents=3,
        field_res=16,
        on_gpu=False,
        preset='synthetic',
        experiment='coherence',
    )

    summary = sim.build_run_summary(result, metrics)

    assert summary['experiment'] == 'coherence'
    assert summary['backend'] == 'CPU'
    assert summary['total_discoveries'] == 2
    assert summary['discoveries_by_operator_type'] == {'perceiver': 2}
    assert summary['coherence_frames'] == [0]
    assert summary['coherence_frame_count'] == 1
    assert summary['discovery_rate_history'] == [2]


def test_write_metrics_sidecar_writes_json_next_to_output(tmp_path):
    sim = load_module()
    metrics = sim.PerformanceMetrics()
    metrics.log_discovery('integrator')
    result = sim.RunResult(
        output_path=tmp_path / 'run.html',
        frames=1,
        agents=1,
        field_res=8,
        on_gpu=False,
    )

    summary_path = sim.write_metrics_sidecar(result, metrics)

    assert summary_path == tmp_path / 'run.metrics.json'
    payload = json.loads(summary_path.read_text(encoding='utf-8'))
    assert payload['total_discoveries'] == 1
    assert payload['discoveries_by_operator_type'] == {'integrator': 1}


def test_save_animation_writes_gif_for_gif_suffix(tmp_path):
    sim = load_module()
    saved = {}

    class FakeAnimation:
        def to_jshtml(self):
            raise AssertionError('HTML path should not be used for a .gif output')

        def save(self, path, writer=None):
            saved['path'] = path
            saved['fps'] = getattr(writer, 'fps', None)
            Path(path).write_bytes(b'GIF89a-fake')

    out = tmp_path / 'clip.gif'
    returned = sim.save_animation(FakeAnimation(), out, fps=15)

    assert returned == out
    assert saved['path'] == str(out)
    assert saved['fps'] == 15
    assert out.read_bytes().startswith(b'GIF89a')


def test_save_animation_writes_html_for_html_suffix(tmp_path):
    sim = load_module()

    class FakeAnimation:
        def to_jshtml(self):
            return '<section>anim</section>'

    out = tmp_path / 'clip.html'
    sim.save_animation(FakeAnimation(), out)
    assert out.read_text(encoding='utf-8') == '<section>anim</section>'


def test_main_writes_metrics_sidecar_when_metrics_present(tmp_path):
    sim = load_module()
    original = snapshot_cfg(sim)
    output_path = tmp_path / 'demo.html'

    class FakeAnimation:
        def to_jshtml(self):
            return '<section>IONS-X demo</section>'

    def fake_run_simulation(**kwargs):
        metrics = sim.PerformanceMetrics()
        metrics.log_discovery('forecaster')
        metrics.log_frame(frame=0, discoveries=1, env_factor=1.0, is_coherence=False)
        return sim.SimulationArtifacts(animation=FakeAnimation(), metrics=metrics)

    try:
        sim.run_simulation = fake_run_simulation
        result = sim.main([
            '--experiment', 'quick',
            '--frames', '1',
            '--output', str(output_path),
        ])

        assert result.experiment == 'quick'
        assert result.summary_path == tmp_path / 'demo.metrics.json'
        payload = json.loads(result.summary_path.read_text(encoding='utf-8'))
        assert payload['total_discoveries'] == 1
        assert payload['discoveries_by_operator_type'] == {'forecaster': 1}
    finally:
        restore_cfg(sim, original)


def test_main_respects_no_metrics_sidecar_flag(tmp_path):
    sim = load_module()
    original = snapshot_cfg(sim)
    output_path = tmp_path / 'demo.html'

    class FakeAnimation:
        def to_jshtml(self):
            return '<section>IONS-X demo</section>'

    def fake_run_simulation(**kwargs):
        return sim.SimulationArtifacts(animation=FakeAnimation(), metrics=sim.PerformanceMetrics())

    try:
        sim.run_simulation = fake_run_simulation
        result = sim.main([
            '--frames', '1',
            '--no-metrics-sidecar',
            '--output', str(output_path),
        ])

        assert result.summary_path is None
        assert not (tmp_path / 'demo.metrics.json').exists()
    finally:
        restore_cfg(sim, original)
