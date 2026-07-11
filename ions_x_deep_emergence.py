import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import cupy as cp

    xp = cp
    fft_rfft = cp.fft.rfft2
    ifft_irfft = cp.fft.irfft2
    on_gpu = True
except ImportError:
    cp = None
    xp = np
    fft_rfft = np.fft.rfft2
    ifft_irfft = np.fft.irfft2
    on_gpu = False


class CFG:
    FIELD_RES = 128
    CHANNELS = 4
    AGENTS = 300
    MEMORY = 300
    AGENT_TYPES = ['perceiver', 'forecaster', 'integrator']
    CORR_WINDOW = 50
    DISCOVER_THRESH = 0.32
    LAG_FRAMES = [5, 15, 30, 60]
    CONFIDENCE_DECAY = 0.995
    GEOMAG_INFLUENCE = True
    LUNAR_CYCLE = True
    COHERENCE_BOOST = True
    FRAMES = 500
    SAMPLE_PER_FRAME = 8
    SEED = 42
    STEP_SIZE = 3


CFG = CFG()
rng = np.random.RandomState(CFG.SEED)
DEFAULT_OUTPUT = Path('outputs/latest.html')
DEFAULT_METRICS_DIR = Path('outputs')
DEFAULT_GIF_FPS = 20

# Named experiment bundles. Each maps CFG attribute -> value and is applied
# before --quick and before any explicit numeric flags, so command-line
# overrides always win. 'balanced' is the documented default (empty = ship CFG).
EXPERIMENTS: Mapping[str, Mapping[str, Any]] = {
    'balanced': {},
    'quick': {'FIELD_RES': 64, 'AGENTS': 50, 'FRAMES': 60, 'SAMPLE_PER_FRAME': 4},
    # Associative-remote-viewing style: patient operators reading against
    # long temporal displacement, so widen memory, lag, and the correlation
    # window while lowering the discovery bar for weak, lagged structure.
    'arv': {
        'LAG_FRAMES': [15, 30, 60, 120],
        'MEMORY': 500,
        'CORR_WINDOW': 80,
        'DISCOVER_THRESH': 0.28,
        'FRAMES': 400,
    },
    # Emphasize environmental coherence windows: slower confidence decay and a
    # lower threshold let coherence-boosted structure accumulate and persist.
    'coherence': {
        'DISCOVER_THRESH': 0.26,
        'CONFIDENCE_DECAY': 0.997,
        'AGENTS': 400,
        'FRAMES': 300,
    },
    # Saturate the field with operators to study crowding and density effects.
    'dense-agents': {
        'AGENTS': 800,
        'FIELD_RES': 96,
        'FRAMES': 200,
    },
}
CHANNEL_NAMES = ('em_rf', 'optical_ir', 'consciousness_proxy', 'control_baseline')
COVARIATE_NAMES = ('kp_index', 'lunar_phase', 'sidereal_time', 'xray_flux')

SENSOR_ALIASES: Mapping[str, tuple[str, ...]] = {
    'em_rf': ('em_rf', 'electromagnetic_rf', 'magnetometer', 'magnetometer_noise', 'rf_noise', 'rf_spectrum_noise', 'channel_0'),
    'optical_ir': ('optical_ir', 'optical_ir_anomaly', 'pixel_variance', 'sky_pixel_variance', 'ir_anomaly', 'channel_1'),
    'consciousness_proxy': ('consciousness_proxy', 'reg_variance', 'reg_entropy', 'egg_variance', 'raw_entropy', 'entropy', 'channel_2'),
}

COVARIATE_ALIASES: Mapping[str, tuple[str, ...]] = {
    'kp_index': ('kp_index', 'kp', 'geomagnetic_kp', 'geomagnetic_index'),
    'lunar_phase': ('lunar_phase', 'moon_phase', 'lunar_cycle_phase'),
    'sidereal_time': ('sidereal_time', 'local_sidereal_time', 'lst'),
    'xray_flux': ('xray_flux', 'solar_xray_flux', 'solar_flare_xray_flux'),
}


@dataclass
class RunResult:
    output_path: Path
    frames: int
    agents: int
    field_res: int
    on_gpu: bool
    preset: str = 'synthetic'
    experiment: str = 'balanced'
    metrics_path: Path | None = None
    metadata_path: Path | None = None
    summary_path: Path | None = None
    calibration_threshold: float | None = None


@dataclass
class SimulationArtifacts:
    animation: Any
    recorder: Optional['LongitudinalMetricsRecorder'] = None
    calibration_threshold: float | None = None
    metrics: Optional['PerformanceMetrics'] = None


@dataclass
class Observation:
    values: tuple[float, ...]
    env_factor: float


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError('value must be a positive integer')
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the IONS-X Deep Emergence Lab simulation and save an HTML animation.'
    )
    parser.add_argument('--quick', action='store_true', help='Use a smaller, faster configuration for first runs and demos.')
    parser.add_argument(
        '--experiment',
        choices=tuple(EXPERIMENTS),
        help='Named parameter bundle to start from (see README). Explicit flags below still override it.',
    )
    parser.add_argument('--frames', type=positive_int, help='Number of animation frames to render.')
    parser.add_argument('--agents', type=positive_int, help='Number of autonomous agents to simulate.')
    parser.add_argument('--field-res', type=positive_int, dest='field_res', help='2D field resolution.')
    parser.add_argument(
        '--preset',
        choices=('synthetic', 'baseline', 'empirical'),
        default='synthetic',
        help='Run mode: synthetic sandbox, baseline control calibration, or empirical CSV analysis.',
    )
    parser.add_argument('--input-data', type=Path, help='CSV input file for empirical telemetry or baseline control calibration.')
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help='Output path. Use a .html suffix for an interactive animation or .gif for a shareable clip.',
    )
    parser.add_argument('--fps', type=positive_int, default=DEFAULT_GIF_FPS, help='Frames per second when writing a .gif output.')
    parser.add_argument(
        '--no-metrics-sidecar',
        action='store_true',
        help='Do not write the <output>.metrics.json summary next to the animation.',
    )
    parser.add_argument('--show', action='store_true', help='Also display inline when running in an IPython notebook.')
    return parser.parse_args(argv)


def apply_experiment(name: str | None) -> None:
    """Apply a named experiment bundle onto the global CFG in place."""
    if not name:
        return
    if name not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment {name!r}; choose from {sorted(EXPERIMENTS)}.')
    for key, value in EXPERIMENTS[name].items():
        setattr(CFG, key, value)


def apply_runtime_options(args: argparse.Namespace) -> CFG:
    # Precedence, lowest to highest: experiment bundle -> --quick -> explicit flags.
    apply_experiment(getattr(args, 'experiment', None))
    if args.quick:
        CFG.FIELD_RES = 64
        CFG.AGENTS = 50
        CFG.FRAMES = 60
        CFG.SAMPLE_PER_FRAME = min(CFG.SAMPLE_PER_FRAME, 4)
    if args.frames is not None:
        CFG.FRAMES = args.frames
    if args.agents is not None:
        CFG.AGENTS = args.agents
    if args.field_res is not None:
        CFG.FIELD_RES = args.field_res
    return CFG


def save_animation(animation: Any, output_path: Path, fps: int = DEFAULT_GIF_FPS) -> Path:
    """Render the animation to ``output_path``.

    A ``.gif`` suffix writes a shareable clip via Pillow; anything else writes a
    self-contained interactive HTML animation. Both drive the same frame updates,
    so metrics are fully populated afterward either way.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == '.gif':
        from matplotlib.animation import PillowWriter

        animation.save(str(output_path), writer=PillowWriter(fps=fps))
    else:
        output_path.write_text(animation.to_jshtml(), encoding='utf-8')
    return output_path


# Backward-compatible alias for callers that imported the original name.
save_animation_html = save_animation


def _canonical_lookup(columns: Sequence[str]) -> dict[str, str]:
    return {column.lower().strip(): column for column in columns}


def _find_column(df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    lookup = _canonical_lookup(df.columns)
    for alias in aliases:
        match = lookup.get(alias.lower())
        if match is not None:
            return match
    return None


def _filled_numeric_series(df: pd.DataFrame, aliases: Sequence[str], default: float = 0.0) -> pd.Series:
    column = _find_column(df, aliases)
    if column is None:
        return pd.Series(default, index=df.index, dtype='float64')
    numeric = pd.to_numeric(df[column], errors='coerce')
    return numeric.ffill().bfill().fillna(default).astype('float64')


def _filled_timestamps(df: pd.DataFrame) -> pd.Series:
    column = _find_column(df, ('timestamp', 'time', 'datetime', 'date_time', 'utc_timestamp'))
    if column is None:
        return pd.Series(pd.date_range('1970-01-01', periods=len(df), freq='min', tz='UTC'))
    timestamps = pd.to_datetime(df[column], errors='coerce', utc=True)
    if timestamps.isna().all():
        return pd.Series(pd.date_range('1970-01-01', periods=len(df), freq='min', tz='UTC'))
    return timestamps.ffill().bfill()


def _normalize_values(values: pd.Series) -> np.ndarray:
    array = values.astype('float64').to_numpy()
    std = float(np.std(array))
    if std < 1e-12:
        return np.zeros_like(array, dtype='float64')
    return (array - float(np.mean(array))) / std


def _spatial_bases(field_res: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, field_res)
    xx, yy = np.meshgrid(axis, axis, indexing='ij')
    radial = np.exp(-3.0 * (xx**2 + yy**2))
    return np.asarray(
        [
            np.sin(math.pi * xx) * np.cos(math.pi * yy),
            np.cos(2.0 * math.pi * xx) + np.sin(math.pi * yy),
            radial - radial.mean(),
            np.sin(2.0 * math.pi * (xx + yy)),
        ],
        dtype='float32',
    )

@dataclass
class TelemetryTargetField:
    fields: np.ndarray
    timestamps: pd.Series
    raw_values: pd.DataFrame
    covariates: pd.DataFrame
    source: str = 'dataframe'

    @classmethod
    def from_csv(cls, input_path: Path, field_res: int, rng: np.random.RandomState) -> 'TelemetryTargetField':
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f'Input telemetry CSV not found: {input_path}')
        return cls.from_dataframe(pd.read_csv(input_path), field_res=field_res, rng=rng, source=str(input_path))

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        field_res: int,
        rng: np.random.RandomState,
        source: str = 'dataframe',
    ) -> 'TelemetryTargetField':
        if df.empty:
            raise ValueError('Input telemetry data must contain at least one row.')

        timestamps = _filled_timestamps(df).reset_index(drop=True)
        raw_values = pd.DataFrame(index=range(len(df)))
        for canonical, aliases in SENSOR_ALIASES.items():
            raw_values[canonical] = _filled_numeric_series(df, aliases).reset_index(drop=True)
        raw_values['control_baseline'] = rng.normal(0.0, 1.0, len(df))

        covariates = pd.DataFrame(index=range(len(df)))
        for canonical, aliases in COVARIATE_ALIASES.items():
            covariates[canonical] = _filled_numeric_series(df, aliases).reset_index(drop=True)

        fields = cls._map_to_grid(raw_values, field_res=field_res, rng=rng)
        return cls(fields=fields, timestamps=timestamps, raw_values=raw_values, covariates=covariates, source=source)

    @classmethod
    def from_null_control(
        cls,
        frame_count: int,
        field_res: int,
        rng: np.random.RandomState,
        source: str = 'generated-control-baseline',
    ) -> 'TelemetryTargetField':
        df = pd.DataFrame(
            {
                'timestamp': pd.date_range('1970-01-01', periods=frame_count, freq='min', tz='UTC'),
                'em_rf': np.zeros(frame_count),
                'optical_ir': np.zeros(frame_count),
                'reg_variance': np.zeros(frame_count),
            }
        )
        return cls.from_dataframe(df, field_res=field_res, rng=rng, source=source)

    @staticmethod
    def _map_to_grid(raw_values: pd.DataFrame, field_res: int, rng: np.random.RandomState) -> np.ndarray:
        bases = _spatial_bases(field_res)
        texture = rng.normal(0.0, 0.015, size=(len(CHANNEL_NAMES), field_res, field_res)).astype('float32')
        fields = np.zeros((len(raw_values), len(CHANNEL_NAMES), field_res, field_res), dtype='float32')
        for channel_index, channel_name in enumerate(CHANNEL_NAMES):
            normalized = _normalize_values(raw_values[channel_name])
            basis = bases[channel_index] + texture[channel_index]
            for frame_index, value in enumerate(normalized):
                fields[frame_index, channel_index] = (value * basis).astype('float32')
        return fields

    @property
    def frame_count(self) -> int:
        return int(self.fields.shape[0])

    @property
    def field_res(self) -> int:
        return int(self.fields.shape[-1])

    def field_for_frame(self, frame: int) -> np.ndarray:
        index = min(max(frame, 0), self.frame_count - 1)
        return self.fields[index]

    def timestamp_for_frame(self, frame: int) -> pd.Timestamp:
        index = min(max(frame, 0), self.frame_count - 1)
        return pd.Timestamp(self.timestamps.iloc[index])

    def covariates_for_frame(self, frame: int) -> dict[str, float]:
        index = min(max(frame, 0), self.frame_count - 1)
        return {name: float(self.covariates.iloc[index][name]) for name in COVARIATE_NAMES}

    def channel_series(self, channel_index: int) -> np.ndarray:
        channel_name = CHANNEL_NAMES[channel_index]
        return self.raw_values[channel_name].astype('float64').to_numpy()

    def control_only_view(self, rng: np.random.RandomState) -> 'TelemetryTargetField':
        null_target = TelemetryTargetField.from_null_control(
            self.frame_count,
            self.field_res,
            rng=rng,
            source=f'{self.source}:control-only',
        )
        fields = null_target.fields.copy()
        fields[:, 3] = self.fields[:, 3]
        raw_values = null_target.raw_values.copy()
        raw_values['control_baseline'] = self.raw_values['control_baseline'].to_numpy()
        return TelemetryTargetField(
            fields=fields,
            timestamps=self.timestamps.copy(),
            raw_values=raw_values,
            covariates=self.covariates.copy(),
            source=f'{self.source}:control-only',
        )

    def reg_variance_deviation(self, frame: int, window: int = 50) -> float:
        index = min(max(frame, 0), self.frame_count - 1)
        start = max(0, index - window + 1)
        series = self.raw_values['consciousness_proxy'].iloc[start : index + 1]
        std = float(series.std(ddof=0))
        if std < 1e-12:
            return 0.0
        return float((series.iloc[-1] - series.mean()) / std)

    def sensor_anomaly_ratios(self, frame: int, short_window: int = 5, long_window: int = 50) -> dict[str, float]:
        index = min(max(frame, 0), self.frame_count - 1)
        ratios: dict[str, float] = {}
        for channel in ('em_rf', 'optical_ir'):
            short_start = max(0, index - short_window + 1)
            long_start = max(0, index - long_window + 1)
            short_mean = float(self.raw_values[channel].iloc[short_start : index + 1].abs().mean())
            long_mean = float(self.raw_values[channel].iloc[long_start : index + 1].abs().mean())
            ratios[f'{channel}_short_long'] = short_mean / max(long_mean, 1e-12)
        return ratios

class PerformanceMetrics:
    def __init__(self) -> None:
        self.type_counts: dict[str, int] = defaultdict(int)
        self.env_history: list[float] = []
        self.discovery_rate_history: list[int] = []
        self.coherence_frames: list[int] = []

    def log_discovery(self, agent_type: str) -> None:
        self.type_counts[agent_type] += 1

    def log_frame(self, frame: int, discoveries: int, env_factor: float, is_coherence: bool) -> None:
        self.env_history.append(env_factor)
        self.discovery_rate_history.append(discoveries)
        if is_coherence:
            self.coherence_frames.append(frame)

    @property
    def total_discoveries(self) -> int:
        return int(sum(self.type_counts.values()))


class EnvironmentalModerators:
    def __init__(self) -> None:
        self.coherence_events: list[int] = []
        self.m = 1.0

    def update(self, t: int) -> None:
        self.m = 1.0 + 0.03 * math.sin(t / 31.8) + 0.02 * math.cos(t / 55.7)
        if rng.rand() < 0.02:
            self.coherence_events.append(t)

    def get_modulation(self, t: int) -> float:
        return float(self.m * (1.3 if self.is_coherence_active(t) else 1.0))

    def is_coherence_active(self, t: int) -> bool:
        return any(abs(t - ev) < 15 for ev in self.coherence_events)

    @property
    def discovery_threshold(self) -> float:
        return float(CFG.DISCOVER_THRESH)

    @property
    def confidence_decay(self) -> float:
        return float(CFG.CONFIDENCE_DECAY)

    def snapshot(self) -> dict[str, float]:
        return {'coherence_factor': float(self.m)}


class RealWorldModerator:
    def __init__(self, base_threshold: float, base_decay: float, base_window: int) -> None:
        self.base_threshold = float(base_threshold)
        self.base_decay = float(base_decay)
        self.base_window = int(base_window)
        self.coherence_events: list[int] = []
        self.coherence_window = int(base_window)
        self.m = 1.0
        self.discovery_threshold = float(base_threshold)
        self.confidence_decay = float(base_decay)
        self.current_covariates = dict.fromkeys(COVARIATE_NAMES, 0.0)

    def update(self, t: int, covariates: Mapping[str, float]) -> None:
        self.current_covariates = {name: float(covariates.get(name, 0.0) or 0.0) for name in COVARIATE_NAMES}
        kp_pressure = np.clip(self.current_covariates['kp_index'] / 9.0, 0.0, 1.0)
        lunar_phase = self.current_covariates['lunar_phase'] % 1.0
        lunar_alignment = 0.5 + 0.5 * math.cos(2.0 * math.pi * lunar_phase)
        sidereal = (self.current_covariates['sidereal_time'] % 24.0) / 24.0
        sidereal_alignment = 0.5 + 0.5 * math.sin(2.0 * math.pi * sidereal)
        xray_flux = max(abs(self.current_covariates['xray_flux']), 1e-12)
        flare_pressure = np.clip((math.log10(xray_flux) + 8.0) / 3.0, 0.0, 1.0)
        pressure = (
            0.35 * float(kp_pressure)
            + 0.20 * float(lunar_alignment)
            + 0.15 * float(sidereal_alignment)
            + 0.30 * float(flare_pressure)
        )
        self.m = float(1.0 + 0.55 * pressure)
        self.discovery_threshold = float(max(0.05, self.base_threshold / self.m))
        self.confidence_decay = float(max(0.90, self.base_decay - (self.m - 1.0) * 0.025))
        self.coherence_window = int(round(self.base_window * (1.0 + 0.75 * pressure)))
        if pressure >= 0.35 or not self.coherence_events:
            self.coherence_events.append(t)

    def get_modulation(self, t: int) -> float:
        return float(self.m * (1.15 if self.is_coherence_active(t) else 1.0))

    def is_coherence_active(self, t: int) -> bool:
        return any(abs(t - ev) <= self.coherence_window for ev in self.coherence_events)

    def snapshot(self) -> dict[str, float]:
        values = dict(self.current_covariates)
        values.update(
            {
                'coherence_factor': float(self.m),
                'discovery_threshold': float(self.discovery_threshold),
                'confidence_decay': float(self.confidence_decay),
                'coherence_window': float(self.coherence_window),
            }
        )
        return values


class Agent:
    def __init__(self, aid: int, atype: str) -> None:
        self.id = aid
        self.type = atype
        self.memory: list[Observation] = []
        self.pos = rng.randint(0, CFG.FIELD_RES, size=2)

    def observe(self, obs: Observation) -> None:
        self.memory.append(obs)
        if len(self.memory) > CFG.MEMORY:
            self.memory.pop(0)

    def discover(self, threshold: float | None = None) -> list[dict[str, Any]]:
        threshold = CFG.DISCOVER_THRESH if threshold is None else threshold
        if len(self.memory) < CFG.CORR_WINDOW:
            return []
        data = np.array([o.values for o in self.memory[-CFG.CORR_WINDOW :]], dtype='float64')
        discs: list[dict[str, Any]] = []
        for i in range(CFG.CHANNELS):
            for j in range(i + 1, CFG.CHANNELS):
                r = float(np.corrcoef(data[:, i], data[:, j])[0, 1])
                if math.isnan(r):
                    continue
                if abs(r) > threshold:
                    discs.append({'edge': (f'ch{i}', f'ch{j}'), 'pearson_r': r, 'confidence': abs(r), 'operator_type': self.type})
        return discs

class LongitudinalMetricsRecorder:
    def __init__(self, output_dir: Path = DEFAULT_METRICS_DIR, run_id: str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        self.rows: list[dict[str, Any]] = []

    def log_frame(
        self,
        frame: int,
        timestamp: pd.Timestamp,
        discoveries: Sequence[Mapping[str, Any]],
        moderator_values: Mapping[str, float],
        operator_density: float,
        total_discoveries: int,
        reg_variance_deviation: float,
        sensor_anomaly_ratios: Mapping[str, float],
    ) -> None:
        base_row = {
            'Frame': int(frame),
            'Timestamp': pd.Timestamp(timestamp).isoformat(),
            'Active_Moderator_Values': json.dumps(dict(moderator_values), sort_keys=True),
            'Operator_Density': float(operator_density),
            'Total_Cumulative_Discoveries': int(total_discoveries),
            'REG_Variance_Deviation': float(reg_variance_deviation),
            'Sensor_Anomaly_MultiScale_Ratios': json.dumps(dict(sensor_anomaly_ratios), sort_keys=True),
        }
        if not discoveries:
            self.rows.append(
                {**base_row, 'Channel_A': '', 'Channel_B': '', 'Pearson_R': np.nan, 'Confidence_Score': 0.0, 'Operator_Type': ''}
            )
            return
        for discovery in discoveries:
            channel_a, channel_b = discovery['edge']
            self.rows.append(
                {
                    **base_row,
                    'Channel_A': channel_a,
                    'Channel_B': channel_b,
                    'Pearson_R': float(discovery.get('pearson_r', discovery.get('confidence', 0.0))),
                    'Confidence_Score': float(discovery.get('confidence', 0.0)),
                    'Operator_Type': str(discovery.get('operator_type', '')),
                }
            )

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save_summary(self, metadata: Mapping[str, Any]) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / f'longitudinal_run_{self.run_id}.csv.gz'
        metadata_path = self.output_dir / f'metadata_{self.run_id}.json'
        dataframe = self.dataframe()
        dataframe.to_csv(csv_path, index=False, compression='gzip')
        summary = {
            **dict(metadata),
            'row_count': int(len(dataframe)),
            'discovery_rows': int((dataframe.get('Confidence_Score', pd.Series(dtype=float)) > 0).sum()),
            'csv_path': str(csv_path),
        }
        metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
        return csv_path, metadata_path


def evolve_fields(F: Any, t: int, env_factor: float, env_mod: EnvironmentalModerators) -> Any:
    Fk = fft_rfft(F)
    for ci in range(F.shape[0]):
        Fk[ci] *= xp.exp(
            -xp.asarray([0.002, 0.0015, 0.0018, 0.0012])[ci]
            * (
                xp.fft.fftfreq(CFG.FIELD_RES).reshape(CFG.FIELD_RES, 1) ** 2
                + xp.fft.rfftfreq(CFG.FIELD_RES).reshape(1, -1) ** 2
            )
            * 0.5
            * env_factor
        )
    F = ifft_irfft(Fk, s=(CFG.FIELD_RES, CFG.FIELD_RES))
    F[0] += 0.045 * F[1] * env_factor
    if env_mod.is_coherence_active(t):
        F[3] = 0.7 * F[3] + 0.3 * F[2]
    return 0.92 * F + 0.08 * xp.tanh(F * 4.0)


def build_simulation_state(
    target_field: TelemetryTargetField | None = None,
) -> tuple[list[Agent], Any, dict[str, float], PerformanceMetrics, Any, Any]:
    import networkx as nx

    agents = [Agent(i, CFG.AGENT_TYPES[i % 3]) for i in range(CFG.AGENTS)]
    graph = nx.DiGraph()
    conf_map: dict[str, float] = defaultdict(float)
    metrics = PerformanceMetrics()
    if target_field is None:
        env_mod: Any = EnvironmentalModerators()
        F = xp.asarray(rng.normal(0, 0.02, (CFG.CHANNELS, CFG.FIELD_RES, CFG.FIELD_RES)), dtype=xp.float32)
    else:
        env_mod = RealWorldModerator(base_threshold=CFG.DISCOVER_THRESH, base_decay=CFG.CONFIDENCE_DECAY, base_window=15)
        F = xp.asarray(target_field.field_for_frame(0), dtype=xp.float32)
    return agents, graph, conf_map, metrics, env_mod, F


def calibrate_control_threshold(target_field: TelemetryTargetField, corr_window: int, confidence: float = 0.95) -> float:
    if target_field.frame_count < max(3, corr_window):
        return float(CFG.DISCOVER_THRESH)
    control = target_field.channel_series(3)
    historic_null = np.roll(control, max(1, corr_window // 2))
    correlations: list[float] = []
    for end in range(corr_window, len(control) + 1):
        left = control[end - corr_window : end]
        right = historic_null[end - corr_window : end]
        r = float(np.corrcoef(left, right)[0, 1])
        if not math.isnan(r):
            correlations.append(abs(r))
    if not correlations:
        return float(CFG.DISCOVER_THRESH)
    return float(np.clip(np.quantile(correlations, confidence), 0.05, 0.99))


def _synthetic_reg_deviation(F_cpu: np.ndarray) -> float:
    return float(np.std(F_cpu[2]))


def _synthetic_sensor_ratios(F_cpu: np.ndarray) -> dict[str, float]:
    baseline = max(float(np.mean(np.abs(F_cpu[3]))), 1e-12)
    return {
        'em_rf_short_long': float(np.mean(np.abs(F_cpu[0])) / baseline),
        'optical_ir_short_long': float(np.mean(np.abs(F_cpu[1])) / baseline),
    }

def run_simulation(
    target_field: TelemetryTargetField | None = None,
    preset: str = 'synthetic',
    recorder: LongitudinalMetricsRecorder | None = None,
    calibrated_threshold: float | None = None,
) -> SimulationArtifacts:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.animation import FuncAnimation

    plt.rcParams['animation.embed_limit'] = 200
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3)
    ax_graph = fig.add_subplot(gs[:, 1:])
    ax_field = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[1, 0])

    agents, graph, conf_map, metrics, env_mod, F = build_simulation_state(target_field)
    frames_to_render = CFG.FRAMES if target_field is None else min(CFG.FRAMES, target_field.frame_count)

    def update(frame: int) -> None:
        nonlocal F

        if target_field is None:
            env_mod.update(frame)
            modulation = env_mod.get_modulation(frame)
            threshold = calibrated_threshold or env_mod.discovery_threshold
            confidence_decay = env_mod.confidence_decay
            F = evolve_fields(F, frame, modulation, env_mod)
            F_cpu = cp.asnumpy(F) if on_gpu else F
            timestamp = pd.Timestamp('1970-01-01', tz='UTC') + pd.Timedelta(minutes=frame)
            moderator_values = env_mod.snapshot()
            reg_deviation = _synthetic_reg_deviation(F_cpu)
            sensor_ratios = _synthetic_sensor_ratios(F_cpu)
        else:
            F_cpu = target_field.field_for_frame(frame)
            F = xp.asarray(F_cpu, dtype=xp.float32)
            covariates = target_field.covariates_for_frame(frame)
            env_mod.update(frame, covariates)
            modulation = env_mod.get_modulation(frame)
            threshold = calibrated_threshold or env_mod.discovery_threshold
            confidence_decay = env_mod.confidence_decay
            timestamp = target_field.timestamp_for_frame(frame)
            moderator_values = env_mod.snapshot()
            reg_deviation = target_field.reg_variance_deviation(frame, CFG.CORR_WINDOW)
            sensor_ratios = target_field.sensor_anomaly_ratios(frame)

        frame_discoveries: list[dict[str, Any]] = []
        for agent in agents:
            agent.pos = (agent.pos + rng.randint(-CFG.STEP_SIZE, CFG.STEP_SIZE + 1, 2)) % CFG.FIELD_RES
            values = tuple(float(v) for v in F_cpu[:, agent.pos[0], agent.pos[1]])
            agent.observe(Observation(values=values, env_factor=modulation))
            for discovery in agent.discover(threshold=threshold):
                u, v = discovery['edge']
                key = f'{u}->{v}'
                conf_map[key] = max(conf_map[key], discovery['confidence'])
                graph.add_edge(u, v, weight=conf_map[key])
                metrics.log_discovery(agent.type)
                frame_discoveries.append(discovery)

        for key in list(conf_map.keys()):
            conf_map[key] *= confidence_decay
            if conf_map[key] < 0.05:
                u, v = key.split('->')
                if graph.has_edge(u, v):
                    graph.remove_edge(u, v)
                del conf_map[key]

        metrics.log_frame(frame, len(frame_discoveries), modulation, env_mod.is_coherence_active(frame))
        operator_density = len(agents) / float(CFG.FIELD_RES * CFG.FIELD_RES)
        if recorder is not None:
            recorder.log_frame(
                frame=frame,
                timestamp=timestamp,
                discoveries=frame_discoveries,
                moderator_values=moderator_values,
                operator_density=operator_density,
                total_discoveries=metrics.total_discoveries,
                reg_variance_deviation=reg_deviation,
                sensor_anomaly_ratios=sensor_ratios,
            )

        ax_field.clear()
        ax_field.imshow(F_cpu[0], cmap='magma')
        ax_field.set_title('Channel 0: EM/RF Telemetry')
        ax_field.axis('off')

        ax_graph.clear()
        nx.draw(graph, ax=ax_graph, with_labels=True, node_color='orange', edge_color='cyan')
        ax_graph.set_title('Emergent ATOM Discoveries')

        ax_stats.clear()
        ax_stats.axis('off')
        ax_stats.text(
            0.05,
            0.75,
            (
                f'Total Cumulative Discoveries: {metrics.total_discoveries}\n'
                f'Active Environmental Coherence Factor: {modulation:.3f}\n'
                f'REG Variance Deviation: {reg_deviation:.3f}\n'
                'Sensor Anomaly Multi-scale Ratios:\n'
                f"  EM/RF: {sensor_ratios['em_rf_short_long']:.3f}\n"
                f"  Optical/IR: {sensor_ratios['optical_ir_short_long']:.3f}"
            ),
            fontsize=12,
            va='top',
        )
        ax_stats.set_title(f'Run Stats ({preset})')

    animation = FuncAnimation(fig, update, frames=frames_to_render, interval=50, repeat=False)
    return SimulationArtifacts(
        animation=animation,
        recorder=recorder,
        calibration_threshold=calibrated_threshold,
        metrics=metrics,
    )

def _prepare_target_and_threshold(args: argparse.Namespace) -> tuple[TelemetryTargetField | None, float | None]:
    if args.preset == 'synthetic':
        if args.input_data is not None:
            return TelemetryTargetField.from_csv(args.input_data, CFG.FIELD_RES, rng), None
        return None, None

    if args.input_data is not None:
        target = TelemetryTargetField.from_csv(args.input_data, CFG.FIELD_RES, rng)
    elif args.preset == 'baseline':
        target = TelemetryTargetField.from_null_control(CFG.FRAMES, CFG.FIELD_RES, rng)
    else:
        raise ValueError('--preset empirical requires --input-data PATH')

    if args.frames is None:
        CFG.FRAMES = target.frame_count

    if args.preset == 'baseline':
        threshold = calibrate_control_threshold(target, CFG.CORR_WINDOW, confidence=0.95)
        return target.control_only_view(rng), threshold
    return target, None


def build_run_summary(result: RunResult, metrics: 'PerformanceMetrics') -> dict[str, Any]:
    """Assemble the lightweight per-run summary written beside the animation."""
    return {
        'output_path': str(result.output_path),
        'preset': result.preset,
        'experiment': result.experiment,
        'frames': result.frames,
        'agents': result.agents,
        'field_res': result.field_res,
        'backend': 'GPU' if result.on_gpu else 'CPU',
        'total_discoveries': metrics.total_discoveries,
        'discoveries_by_operator_type': dict(metrics.type_counts),
        'coherence_frames': list(metrics.coherence_frames),
        'coherence_frame_count': len(metrics.coherence_frames),
        'discovery_rate_history': list(metrics.discovery_rate_history),
        'calibration_threshold': result.calibration_threshold,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }


def write_metrics_sidecar(result: RunResult, metrics: 'PerformanceMetrics') -> Path:
    """Write ``<output>.metrics.json`` next to the rendered animation."""
    summary_path = result.output_path.with_suffix('.metrics.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_run_summary(result, metrics)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    return summary_path


def main(argv: Sequence[str] | None = None) -> RunResult:
    args = parse_args(argv)
    apply_runtime_options(args)
    target_field, calibration_threshold = _prepare_target_and_threshold(args)
    recorder = LongitudinalMetricsRecorder() if args.preset in {'baseline', 'empirical'} else None

    artifacts = run_simulation(
        target_field=target_field,
        preset=args.preset,
        recorder=recorder,
        calibrated_threshold=calibration_threshold,
    )
    output_path = save_animation(artifacts.animation, args.output, fps=args.fps)

    metrics_path: Path | None = None
    metadata_path: Path | None = None
    if artifacts.recorder is not None:
        metadata = {
            'preset': args.preset,
            'frames': CFG.FRAMES,
            'agents': CFG.AGENTS,
            'field_res': CFG.FIELD_RES,
            'source': target_field.source if target_field is not None else 'synthetic',
            'calibration_threshold': artifacts.calibration_threshold,
            'channel_schema': list(CHANNEL_NAMES),
            'covariate_schema': list(COVARIATE_NAMES),
        }
        metrics_path, metadata_path = artifacts.recorder.save_summary(metadata)

    if args.show:
        from IPython.display import HTML, display

        display(HTML(output_path.read_text(encoding='utf-8')))

    result = RunResult(
        output_path=output_path,
        frames=CFG.FRAMES,
        agents=CFG.AGENTS,
        field_res=CFG.FIELD_RES,
        on_gpu=on_gpu,
        preset=args.preset,
        experiment=getattr(args, 'experiment', None) or 'balanced',
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        calibration_threshold=artifacts.calibration_threshold,
    )

    # Write the lightweight metrics summary beside the animation. The animation's
    # frame updates ran during save_animation above, so metrics are populated now.
    if artifacts.metrics is not None and not getattr(args, 'no_metrics_sidecar', False):
        result.summary_path = write_metrics_sidecar(result, artifacts.metrics)

    message = (
        'Simulation complete. '
        f'Preset: {result.preset}. Experiment: {result.experiment}. '
        f'Frames: {result.frames}. Agents: {result.agents}. '
        f'Field: {result.field_res}x{result.field_res}. '
        f'Backend: {"GPU" if result.on_gpu else "CPU"}. '
        f'Output: {result.output_path}'
    )
    if result.summary_path is not None:
        message += f'. Summary: {result.summary_path}'
    if result.metrics_path is not None:
        message += f'. Metrics: {result.metrics_path}. Metadata: {result.metadata_path}'
    print(message)
    return result


if __name__ == '__main__':
    main()
