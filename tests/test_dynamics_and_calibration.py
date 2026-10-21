"""The generator, the detector's calibration, and the seed-level statistics, each checked against a known truth."""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import study_report as sr

ROOT = Path(__file__).resolve().parents[1]
LEGACY_BENCHMARK = ROOT / 'docs' / 'benchmarks' / 'legacy-0.4' / 'control-study.json'
BENCHMARK = ROOT / 'docs' / 'benchmarks' / 'control-study.json'
CH2_CH3 = 5  # index of ch2--ch3 in channel_pairs(4)


@pytest.fixture
def sim():
    spec = importlib.util.spec_from_file_location('ions_x_dynamics_tests', ROOT / 'ions_x_deep_emergence.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(sim, *flags):
    sim.apply_runtime_options(sim.parse_args(list(flags)))


def study_flags(report):
    config = report['passport']['configuration']
    return ['--quick', '--frames', str(config['FRAMES']), '--agents', str(config['AGENTS']),
            '--field-res', str(config['FIELD_RES'])]


def assert_close(fresh, committed):
    """Exact for structure and integers; floats to 1e-12, since scipy's special functions vary in the last digit."""
    if isinstance(committed, dict):
        assert fresh.keys() == committed.keys()
        for key in committed:
            assert_close(fresh[key], committed[key])
    elif isinstance(committed, list):
        assert len(fresh) == len(committed)
        for new, old in zip(fresh, committed, strict=True):
            assert_close(new, old)
    elif isinstance(committed, float):
        assert fresh == pytest.approx(committed, rel=1e-12)
    else:
        assert fresh == committed


def assert_same_runs(fresh, committed, keys):
    for new, old in zip(fresh['runs'], committed['runs'], strict=True):
        assert new['seed'] == old['seed']
        for arm in ('coupled', 'uncoupled'):
            for key in keys:
                assert new[arm][key] == old[arm][key], (new['seed'], arm, key)


def test_legacy_dynamics_reproduce_the_committed_040_benchmark_exactly(sim):
    committed = json.loads(LEGACY_BENCHMARK.read_text())
    configure(sim, *study_flags(committed), '--dynamics', 'legacy')
    report = sim.run_control_study(committed['repetitions'], committed['starting_seed'])
    assert_same_runs(report, committed, ('total_discoveries', 'edge_counts', 'coherence_frames'))
    assert report['association_rates'] == committed['association_rates']


def test_committed_driven_benchmark_regenerates_exactly(sim):
    committed = json.loads(BENCHMARK.read_text())
    configure(sim, *study_flags(committed))
    report = sim.run_control_study(committed['repetitions'], committed['starting_seed'])
    assert_same_runs(report, committed, ('total_discoveries', 'edge_counts', 'coherence_frames',
                                         'evaluated_windows', 'abs_r_histogram'))
    assert_close(report['detector_calibration'], committed['detector_calibration'])


def test_committed_benchmark_summaries_follow_from_its_runs():
    report = json.loads(BENCHMARK.read_text())
    assert report['dynamics'] == report['passport']['configuration']['DYNAMICS'] == 'driven'
    assert report['paired_association_effects'] == sr.paired_association_summary(
        report['runs'], report['opportunities_per_pair_per_run'])
    assert report['discrimination'] == sr.discrimination_summary(report['runs'])


@pytest.mark.parametrize('dynamics', ['driven', 'legacy'])
def test_field_diagnostics_expose_the_frozen_legacy_field(sim, dynamics):
    configure(sim, '--frames', '201', '--agents', '2', '--field-res', '32', '--seed', '5', '--dynamics', dynamics)
    field = sim.SimulationEngine().run().metrics.field
    assert len(field.window_memory) == 4
    if dynamics == 'driven':  # renewed within one window, smooth between neighbours
        assert max(field.window_memory) < 0.3
        assert min(field.neighbour_correlation) > 0.5
    else:  # frozen salt-and-pepper noise
        assert field.window_memory[-1] > 0.99
        assert max(abs(value) for value in field.neighbour_correlation) < 0.1


@pytest.mark.parametrize('dynamics', ['driven', 'legacy'])
def test_coherence_coupling_fades_only_in_the_driven_field(sim, dynamics):
    configure(sim, '--frames', '160', '--agents', '2', '--field-res', '32', '--seed', '8', '--dynamics', dynamics)
    engine = sim.SimulationEngine()
    engine.env_mod.coherence_events = [20]  # exactly one window: frames 6 through 34
    engine.env_mod.update = lambda t: None
    engine.run()
    r23 = [frame[CH2_CH3] for frame in engine.metrics.field.pair_correlations]
    assert max(r23[6:35]) > 0.5
    long_after = float(np.mean(r23[110:]))
    if dynamics == 'driven':
        assert long_after < 0.2
    else:  # one window made channel 3 a permanent copy of channel 2
        assert long_after > 0.9


def test_vectorized_engine_keeps_the_agent_api_and_accounts_for_every_window(sim):
    configure(sim, '--frames', '70', '--agents', '6', '--field-res', '16', '--seed', '3')
    sim.CFG.CORR_WINDOW = 20
    engine = sim.SimulationEngine().run()
    agent = engine.agents[2]
    assert len(agent.memory) == 70
    assert agent.memory[-1] == agent.memory[69]
    np.testing.assert_array_equal(agent.pos, engine.positions[2])
    field = engine.snapshot['field']
    assert agent.memory[-1].values == tuple(float(v) for v in field[:, agent.pos[0], agent.pos[1]])
    batch = sim.pairwise_correlations(engine.observations.windows(20))[2]
    for discovery in agent.discover(threshold=0):
        i, j = (int(name[2:]) for name in discovery['edge'])
        assert discovery['pearson_r'] == pytest.approx(batch[i, j], abs=1e-12)

    metrics = engine.metrics
    assert metrics.evaluated_windows == 6 * (70 - 20 + 1)
    assert metrics.abs_r_histogram.sum() == metrics.evaluated_windows * 6
    threshold_bin = round(sim.CFG.DISCOVER_THRESH * sr.HISTOGRAM_BINS)
    for k, (i, j) in enumerate(sim.channel_pairs(4)):
        assert sr.tail_count(metrics.abs_r_histogram[k], threshold_bin) == metrics.edge_counts.get(f'ch{i}--ch{j}', 0)


def test_nominal_false_alarm_rate_matches_the_t_distribution_and_the_detector_itself(sim):
    nominal = sim.nominal_false_alarm_rate(50, 0.32)
    t_statistic = 0.32 * np.sqrt(48) / np.sqrt(1 - 0.32**2)
    assert nominal == pytest.approx(2 * stats.t.sf(t_statistic, 48), rel=1e-10)
    independent = np.random.RandomState(0).normal(size=(40_000, 50, 2))
    observed = np.mean(np.abs(sim.pairwise_correlations(independent)[:, 0, 1]) > 0.32)
    assert observed == pytest.approx(nominal, abs=0.003)
    assert sim.effective_samples(nominal, 0.32) == pytest.approx(50, rel=1e-6)
    assert sim.effective_samples(4 * nominal, 0.32) < 50


def test_sign_flip_holm_auc_and_thresholds_match_hand_computed_values():
    assert sr.sign_flip_p_value([1, 2, 3]) == (0.25, 'exact')
    assert sr.sign_flip_p_value([0, 0, 0]) == (1.0, 'exact')
    assert sr.sign_flip_p_value([2.0, -2.0]) == (1.0, 'exact')
    p_value, method = sr.sign_flip_p_value([1.0] * 20)
    assert method == 'monte_carlo'
    assert p_value == pytest.approx(1 / (sr.MONTE_CARLO_DRAWS + 1))
    assert sr.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert sr.auc_from_histograms([1, 1], [1, 1]) == 0.5
    assert sr.auc_from_histograms([0, 2], [2, 0]) == 1.0
    assert sr.auc_from_histograms([2, 0], [0, 2]) == 0.0
    assert sr.threshold_bin_for_rate([50, 30, 15, 5], 0.06) == 3
    assert sr.threshold_bin_for_rate([50, 30, 15, 5], 1.0) == 0


def test_control_study_reports_calibration_discrimination_and_ground_truth(sim, tmp_path):
    result = sim.main(['--control-study', '4', '--frames', '60', '--agents', '6', '--field-res', '16',
                       '--output', str(tmp_path / 'study.json')])
    report = json.loads(result.summary_path.read_text())
    assert report['schema_version'] == 3
    assert report['dynamics'] == 'driven'
    assert report['passport']['version'] == sim.__version__
    calibration = report['detector_calibration']
    assert calibration['nominal_false_alarm_rate'] == pytest.approx(sim.nominal_false_alarm_rate(50, 0.32))
    assert set(calibration['held_out']['calibration_seeds']).isdisjoint(calibration['held_out']['evaluation_seeds'])
    assert set(report['discrimination']) == set(report['association_rates'])
    assert set(report['field_ground_truth']['pair_correlation_mean']) == set(report['association_rates'])
    for run in report['runs']:
        for arm in ('coupled', 'uncoupled'):
            assert run[arm]['evaluated_windows'] == report['opportunities_per_pair_per_run']
            assert all(sum(hist) == run[arm]['evaluated_windows'] for hist in run[arm]['abs_r_histogram'].values())
    html = result.report_path.read_text()
    for heading in ('Is the detector calibrated?', 'Threshold-free discrimination', 'What the field actually contained'):
        assert heading in html
    assert '<script' not in html


def test_reports_from_earlier_schemas_still_render(tmp_path):
    committed = json.loads(LEGACY_BENCHMARK.read_text())
    html = sr.render_study_report(committed, tmp_path / 'legacy.html').read_text()
    assert 'Every seed, every pair' in html
    assert 'Is the detector calibrated?' not in html


def test_telemetry_projection_is_lazy_and_exact(sim):
    rows = 20_000
    noise = np.random.RandomState(1)
    df = pd.DataFrame({name: noise.normal(size=rows) for name in ('em_rf', 'optical_ir', 'reg_variance')})
    target = sim.TelemetryTargetField.from_dataframe(df, field_res=128, rng=np.random.RandomState(2))
    # Materializing every frame would need rows * 4 * 128 * 128 * 4 bytes, about 5.2 GB.
    assert target.signals.nbytes + target.bases.nbytes < 2_000_000
    assert target.field_for_frame(rows - 1).shape == (4, 128, 128)
    small = sim.TelemetryTargetField.from_dataframe(df.head(5), field_res=6, rng=np.random.RandomState(2))
    for frame in range(5):
        np.testing.assert_array_equal(small.field_for_frame(frame), small.fields[frame])


def test_coherence_lookup_matches_brute_force_and_scales(sim):
    events = sorted(np.random.RandomState(4).randint(0, 5000, 300).tolist())
    synthetic = sim.EnvironmentalModerators()
    synthetic.coherence_events = events
    empirical = sim.RealWorldModerator(0.32, 0.995, 15)
    empirical.coherence_events = events
    for t in range(-20, 5020, 7):
        assert synthetic.is_coherence_active(t) == any(abs(t - event) < 15 for event in events)
        assert empirical.is_coherence_active(t) == any(abs(t - event) <= 15 for event in events)
    empirical.coherence_events = list(range(100_000))  # empirical runs can log an event on every frame
    start = time.perf_counter()
    assert all(empirical.is_coherence_active(t) for t in range(0, 100_000, 10))
    assert time.perf_counter() - start < 1.0


def test_dynamics_flag_is_synthetic_only_validated_and_recorded(sim):
    with pytest.raises(SystemExit):
        sim.parse_args(['--dynamics', 'legacy', '--preset', 'empirical'])
    configure(sim, '--dynamics', 'legacy')
    assert sim.CFG.DYNAMICS == 'legacy'
    assert set(sim.DRIVEN_SETTINGS) <= set(sim.experiment_passport()['inactive_configuration'])
    configure(sim)
    assert sim.CFG.DYNAMICS == 'driven'
    assert not set(sim.DRIVEN_SETTINGS) & set(sim.experiment_passport()['inactive_configuration'])
    assert 'DYNAMICS' in sim.experiment_passport(synthetic=False)['inactive_configuration']
    sim.CFG.FIELD_MEMORY = 1.0
    with pytest.raises(ValueError, match='FIELD_MEMORY'):
        sim.SimulationEngine()


def test_reports_keep_numeric_arrays_compact_without_touching_strings(sim, tmp_path):
    payload = {'counts': [1, 2, 3], 'nested': [[0.5, -1e-05], []], 'text': 'keep [1,\n 2] exactly', 'mixed': [None, 1]}
    path = sim.write_report(tmp_path / 'report.json', payload)
    assert json.loads(path.read_text()) == payload
    assert '[1, 2, 3]' in path.read_text()
