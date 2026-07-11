import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / 'ions_x_deep_emergence.py'
MODULE_SPEC = importlib.util.spec_from_file_location('ions_x_deep_emergence', MODULE_PATH)
sim = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(sim)


def test_agent_discover_finds_strong_correlation():
    original_window = sim.CFG.CORR_WINDOW
    original_channels = sim.CFG.CHANNELS
    original_thresh = sim.CFG.DISCOVER_THRESH
    try:
        sim.CFG.CORR_WINDOW = 5
        sim.CFG.CHANNELS = 2
        sim.CFG.DISCOVER_THRESH = 0.5

        agent = sim.Agent(1, 'perceiver')
        series = [0.0, 1.0, 2.0, 3.0, 4.0]
        for value in series:
            agent.observe(sim.Observation(values=(value, value), env_factor=1.0))

        discoveries = agent.discover()
        assert len(discoveries) == 1
        assert discoveries[0]['edge'] == ('ch0', 'ch1')
        assert discoveries[0]['confidence'] > sim.CFG.DISCOVER_THRESH
    finally:
        sim.CFG.CORR_WINDOW = original_window
        sim.CFG.CHANNELS = original_channels
        sim.CFG.DISCOVER_THRESH = original_thresh


def test_is_coherence_active_window_behavior():
    env = sim.EnvironmentalModerators()
    env.coherence_events = [100]

    assert env.is_coherence_active(86)
    assert env.is_coherence_active(114)
    assert not env.is_coherence_active(85)
    assert not env.is_coherence_active(115)


def test_import_has_no_runtime_animation_side_effects():
    reloaded = importlib.util.module_from_spec(MODULE_SPEC)
    MODULE_SPEC.loader.exec_module(reloaded)

    assert not hasattr(reloaded, 'anim')
    assert callable(reloaded.run_simulation)


def test_target_field_ingestion_forward_fills_missing_csv_values():
    df = pd.DataFrame(
        {
            'timestamp': ['2026-01-01T00:00:00Z', None, '2026-01-01T00:02:00Z'],
            'em_rf': [12.0, None, 18.0],
            'optical_ir': [3.0, None, 9.0],
            'reg_variance': [0.51, None, 0.54],
            'kp_index': [2.0, None, 5.0],
            'lunar_phase': [0.25, None, 0.5],
            'sidereal_time': [6.0, None, 8.0],
            'xray_flux': [1e-7, None, 2e-7],
        }
    )

    target = sim.TelemetryTargetField.from_dataframe(
        df,
        field_res=8,
        rng=np.random.RandomState(7),
        source='unit-test.csv',
    )

    assert target.frame_count == 3
    assert target.fields.shape == (3, 4, 8, 8)
    assert not np.isnan(target.fields).any()
    assert target.timestamps.isna().sum() == 0
    assert target.raw_values.iloc[1]['em_rf'] == 12.0
    assert target.raw_values.iloc[1]['optical_ir'] == 3.0
    assert target.raw_values.iloc[1]['consciousness_proxy'] == 0.51
    assert target.covariates_for_frame(1)['kp_index'] == 2.0
    assert target.source == 'unit-test.csv'


def test_real_world_moderator_scales_threshold_and_decay_from_covariates():
    moderator = sim.RealWorldModerator(
        base_threshold=0.32,
        base_decay=0.995,
        base_window=15,
    )

    moderator.update(
        10,
        {
            'kp_index': 8.0,
            'lunar_phase': 0.0,
            'sidereal_time': 12.0,
            'xray_flux': 8e-6,
        },
    )

    assert moderator.get_modulation(10) > 1.0
    assert moderator.discovery_threshold < 0.32
    assert moderator.confidence_decay < 0.995
    assert moderator.coherence_window >= 15
    assert moderator.is_coherence_active(10)


def test_baseline_calibration_returns_95_percent_control_threshold():
    rng = np.random.RandomState(12)
    df = pd.DataFrame(
        {
            'timestamp': pd.date_range('2026-01-01', periods=80, freq='min'),
            'em_rf': rng.normal(size=80),
            'optical_ir': rng.normal(size=80),
            'reg_variance': rng.normal(size=80),
        }
    )
    target = sim.TelemetryTargetField.from_dataframe(df, field_res=4, rng=np.random.RandomState(99))

    threshold = sim.calibrate_control_threshold(target, corr_window=12, confidence=0.95)

    assert 0.0 < threshold < 1.0
    assert threshold >= 0.2


def test_longitudinal_recorder_exports_discovery_schema(tmp_path):
    recorder = sim.LongitudinalMetricsRecorder(
        output_dir=tmp_path,
        run_id='20260101_000000',
    )
    recorder.log_frame(
        frame=0,
        timestamp=pd.Timestamp('2026-01-01T00:00:00Z'),
        discoveries=[
            {
                'edge': ('ch0', 'ch2'),
                'pearson_r': 0.81,
                'confidence': 0.81,
                'operator_type': 'perceiver',
            }
        ],
        moderator_values={'kp_index': 4.0, 'coherence_factor': 1.2},
        operator_density=0.125,
        total_discoveries=1,
        reg_variance_deviation=0.42,
        sensor_anomaly_ratios={'em_rf_short_long': 1.4, 'optical_ir_short_long': 0.9},
    )

    csv_path, metadata_path = recorder.save_summary(
        metadata={'preset': 'empirical', 'frames': 1},
    )

    exported = pd.read_csv(csv_path)
    assert csv_path.name == 'longitudinal_run_20260101_000000.csv.gz'
    assert metadata_path.name == 'metadata_20260101_000000.json'
    assert {
        'Timestamp',
        'Channel_A',
        'Channel_B',
        'Pearson_R',
        'Confidence_Score',
        'Active_Moderator_Values',
        'Operator_Density',
    }.issubset(exported.columns)
    assert exported.loc[0, 'Channel_A'] == 'ch0'
    assert exported.loc[0, 'Channel_B'] == 'ch2'
    assert exported.loc[0, 'Pearson_R'] == 0.81
