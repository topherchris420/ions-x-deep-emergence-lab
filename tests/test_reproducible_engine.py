"""Regression checks for actual frame execution and paired experiment validity."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sim():
    path = Path(__file__).resolve().parents[1] / 'ions_x_deep_emergence.py'
    spec = importlib.util.spec_from_file_location('ions_x_engine_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_runtime_options(module.parse_args(['--frames', '8', '--agents', '4', '--field-res', '8']))
    module.CFG.CORR_WINDOW = 3
    return module


def test_repeated_frame_does_not_advance_any_state(sim):
    recorder = sim.LongitudinalMetricsRecorder()
    engine = sim.SimulationEngine(recorder=recorder)
    first = engine.step(0)
    state = sim.rng.get_state()
    assert engine.step(0) is first
    assert len(recorder.rows) == 1
    assert len(engine.metrics.discovery_rate_history) == 1
    assert len(engine.agents[0].memory) == 1
    np.testing.assert_array_equal(sim.rng.get_state()[1], state[1])
    assert sim.rng.get_state()[2:] == state[2:]
    with pytest.raises(ValueError, match='sequentially'):
        engine.step(2)
    engine.run()
    assert len(engine.metrics.discovery_rate_history) == 8
    with pytest.raises(ValueError):
        engine.step(0)


def test_graph_is_undirected_and_weight_tracks_decay(sim):
    sim.CFG.DISCOVER_THRESH = 1.0  # No new detections, so isolate decay.
    engine = sim.SimulationEngine()
    engine.conf_map['ch0->ch1'] = 0.8
    engine.graph.add_edge('ch0', 'ch1', weight=0.8)
    engine.run()
    assert not engine.graph.is_directed()
    assert engine.graph['ch0']['ch1']['weight'] == pytest.approx(0.8 * sim.CFG.CONFIDENCE_DECAY**8)
    assert len(engine.graph.nodes) == 4


def test_constant_channels_do_not_emit_warnings_or_discoveries(sim):
    agent = sim.Agent(0, 'perceiver')
    for _ in range(3):
        agent.observe(sim.Observation((1.0, 1.0, 1.0, 1.0), 1.0))
    with np.errstate(all='raise'):
        assert agent.discover() == []


def test_vectorized_correlations_match_numpy(sim):
    agent = sim.Agent(0, 'perceiver')
    values = np.random.RandomState(17).normal(size=(3, 4))
    for row in values:
        agent.observe(sim.Observation(tuple(row), 1.0))
    reference = np.corrcoef(values.T)
    for item in agent.discover(threshold=0):
        i, j = (int(name[2:]) for name in item['edge'])
        assert item['pearson_r'] == pytest.approx(reference[i, j])


@pytest.mark.parametrize('suffix', ['html', 'gif'])
def test_rendering_and_headless_engine_produce_identical_metrics(sim, tmp_path, suffix):
    sim.set_seed(32)
    expected = sim.SimulationEngine().run().metrics
    sim.set_seed(32)
    artifacts = sim.run_simulation()
    sim.save_animation(artifacts.animation, tmp_path / f'run.{suffix}')
    assert artifacts.animation._fig.get_facecolor()[:3] == pytest.approx((8 / 255, 27 / 255, 32 / 255))
    assert len(artifacts.metrics.discovery_rate_history) == 8
    assert artifacts.metrics.discovery_rate_history == expected.discovery_rate_history
    assert artifacts.metrics.edge_counts == expected.edge_counts
    assert artifacts.metrics.env_history == expected.env_history


def test_cli_options_do_not_leak_between_runs(sim):
    sim.apply_runtime_options(sim.parse_args(['--experiment', 'arv', '--seed', '123']))
    sim.apply_runtime_options(sim.parse_args([]))
    assert {key: getattr(sim.CFG, key) for key in sim.DEFAULT_CONFIG} == sim.DEFAULT_CONFIG


def test_headless_report_has_effective_frame_count_input_hash_and_quality(sim, tmp_path):
    csv = tmp_path / 'sensors.csv'
    pd.DataFrame({'em_rf': [1, 2, 3], 'optical_ir': [3, None, 1], 'reg_variance': [0, 1, 2]}).to_csv(csv, index=False)
    result = sim.main(['--headless', '--input-data', str(csv), '--frames', '20', '--agents', '2',
                       '--field-res', '4', '--output', str(tmp_path / 'run.json')])
    report = json.loads(result.output_path.read_text())
    assert report['frames'] == report['frames_processed'] == 3
    assert report['preset'] == 'empirical'
    assert len(report['passport']['input_sha256']) == 64
    assert report['passport']['telemetry_quality']['imputed_sensor_cells']['optical_ir'] == 1
    assert result.metrics_path.parent == tmp_path


def test_headless_cli_does_not_call_renderer(sim, tmp_path, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail('Headless mode attempted to render')
    monkeypatch.setattr(sim, 'run_simulation', forbidden)
    result = sim.main(['--headless', '--frames', '3', '--agents', '2', '--field-res', '4',
                       '--output', str(tmp_path / 'run.json')])
    assert json.loads(result.output_path.read_text())['frames_processed'] == 3


def test_paired_study_is_deterministic_and_restores_rng(sim):
    before = sim.rng.get_state()
    report = sim.run_control_study(3, seed=100)
    assert report == sim.run_control_study(3, seed=100)
    assert [row['seed'] for row in report['runs']] == [100, 101, 102]
    for row in report['runs']:
        assert row['coupled']['coherence_frames'] == row['uncoupled']['coherence_frames']
        assert row['paired_difference'] == row['coupled']['total_discoveries'] - row['uncoupled']['total_discoveries']
    for rates in report['association_rates'].values():
        assert all(0 <= rate <= 1 for rate in rates.values())
    np.testing.assert_array_equal(sim.rng.get_state()[1], before[1])
    assert sim.rng.get_state()[2:] == before[2:]
    assert sim.CFG.SEED == 42


def test_coupling_ablation_preserves_agent_paths_and_random_schedule(sim):
    sim.set_seed(41)
    coupled = sim.SimulationEngine().run()
    coupled_state = sim.rng.get_state()
    sim.set_seed(41)
    uncoupled = sim.SimulationEngine(coupled=False).run()
    np.testing.assert_array_equal(sim.rng.get_state()[1], coupled_state[1])
    assert sim.rng.get_state()[2:] == coupled_state[2:]
    np.testing.assert_array_equal([a.pos for a in coupled.agents], [a.pos for a in uncoupled.agents])
    np.testing.assert_allclose(coupled.F[1], uncoupled.F[1])
    assert not np.allclose(coupled.F[0], uncoupled.F[0])


def test_control_study_rejects_inert_run_and_seed_overflow(sim):
    with pytest.raises(ValueError, match='seed range'):
        sim.run_control_study(2, 2**32 - 1)
    sim.CFG.FRAMES = 2
    with pytest.raises(ValueError, match='CORR_WINDOW'):
        sim.run_control_study(1, 42)


@pytest.mark.parametrize('flags', [
    ['--seed', '-1'], ['--seed', str(2**32)], ['--headless', '--show'],
    ['--headless', '--output', 'run.html'], ['--control-study', '2', '--preset', 'empirical'],
    ['--control-study', '2', '--input-data', 'data.csv'], ['--output', 'run.txt'],
])
def test_invalid_cli_combinations_are_rejected(sim, flags):
    with pytest.raises(SystemExit):
        sim.parse_args(flags)


@pytest.mark.parametrize('bad', [pd.DataFrame({'unrelated': [1, 2]}),
    pd.DataFrame({'em_rf': [1, np.inf], 'optical_ir': [1, 2], 'reg_variance': [1, 2]}),
    pd.DataFrame({'em_rf': ['bad', None], 'optical_ir': [1, 2], 'reg_variance': [1, 2]})])
def test_unusable_telemetry_is_not_silently_invented(sim, bad):
    with pytest.raises(ValueError):
        sim.TelemetryTargetField.from_dataframe(bad, field_res=4, rng=np.random.RandomState(2))


def test_baseline_contains_four_nonconstant_independent_channels(sim):
    target = sim.TelemetryTargetField.from_null_control(1000, 2, np.random.RandomState(3))
    assert (target.raw_values.std() > 0.5).all()
    correlations = np.corrcoef(target.raw_values.to_numpy().T)
    assert np.max(np.abs(correlations - np.eye(4))) < 0.12


def test_longitudinal_run_ids_do_not_collide(sim):
    assert sim.LongitudinalMetricsRecorder().run_id != sim.LongitudinalMetricsRecorder().run_id
