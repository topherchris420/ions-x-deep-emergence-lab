"""IONS-X Deep Emergence Lab: moving agents sense a four-channel field and report associations.

The synthetic field is a known generator, so the lab can ask a falsifiable question: does the
detector respond differently when the injected coupling is removed, and how often does it fire
when nothing was injected at all?
"""

import argparse
import bisect
import hashlib
import json
import math
import platform
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from study_report import (
    HISTOGRAM_BINS,
    discrimination_summary,
    paired_association_summary,
    pooled_histogram,
    render_study_report,
    tail_count,
    threshold_bin_for_rate,
)

__version__ = '0.5.0'

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
    FIELD_RES: int = 128
    CHANNELS: int = 4
    AGENTS: int = 300
    MEMORY: int = 300
    AGENT_TYPES: tuple[str, ...] = ('perceiver', 'forecaster', 'integrator')
    CORR_WINDOW: int = 50
    DISCOVER_THRESH: float = 0.32
    LAG_FRAMES: tuple[int, ...] = (5, 15, 30, 60)
    CONFIDENCE_DECAY: float = 0.995
    GEOMAG_INFLUENCE: bool = True
    LUNAR_CYCLE: bool = True
    COHERENCE_BOOST: bool = True
    FRAMES: int = 500
    SAMPLE_PER_FRAME: int = 8
    SEED: int = 42
    STEP_SIZE: int = 3
    # Synthetic field generator. 'driven' (default since 0.5) renews a spatially correlated field
    # every frame; 'legacy' reproduces the <=0.4 generator bit for bit, including its defects.
    DYNAMICS: str = 'driven'
    FIELD_MEMORY: float = 0.95  # per-frame persistence: correlation time 1 / (1 - memory) frames
    FIELD_CORRELATION_LENGTH: float = 1.0  # pixels; spatial scale of the forcing noise
    COUPLING_STRENGTH: float = 0.08  # injected ch1 -> ch0 per frame, scaled by the moderator factor
    COHERENCE_MIXING: float = 0.25  # injected ch2 -> ch3 per frame, inside coherence windows only


CFG = CFG()
DEFAULT_CONFIG = {key: getattr(CFG, key) for key in type(CFG).__annotations__}
rng = np.random.RandomState(CFG.SEED)


def set_seed(seed: int | None = None) -> None:
    """Set the global simulation random seed deterministically."""
    if seed is not None:
        CFG.SEED = int(seed)
    rng.seed(CFG.SEED)


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
        'LAG_FRAMES': (15, 30, 60, 120),
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
DYNAMICS_MODES = ('driven', 'legacy')
DRIVEN_SETTINGS = ('FIELD_MEMORY', 'FIELD_CORRELATION_LENGTH', 'COUPLING_STRENGTH', 'COHERENCE_MIXING')
UNUSED_SETTINGS = ('LAG_FRAMES', 'SAMPLE_PER_FRAME', 'GEOMAG_INFLUENCE', 'LUNAR_CYCLE', 'COHERENCE_BOOST')
INJECTED_ASSOCIATIONS = {'ch0--ch1': 'continuous', 'ch2--ch3': 'coherence windows only'}
# Per-channel diffusion. The legacy kernel omits (2*pi)^2, so its coefficients attenuate the
# Nyquist frequency by at most 0.07% per frame; the driven kernel uses a true Laplacian.
LEGACY_DIFFUSION = (0.002, 0.0015, 0.0018, 0.0012)
DRIVEN_DIFFUSION = (0.1, 0.075, 0.09, 0.06)
CHANNEL_NAMES = ('em_rf', 'optical_ir', 'consciousness_proxy', 'control_baseline')
CHANNEL_LABELS = {'ch0': 'CH 0\nEM / RF', 'ch1': 'CH 1\nOptical / IR', 'ch2': 'CH 2\nREG proxy', 'ch3': 'CH 3\nReference'}
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
    seed: int = 42
    preset: str = 'synthetic'
    experiment: str = 'balanced'
    metrics_path: Path | None = None
    metadata_path: Path | None = None
    summary_path: Path | None = None
    calibration_threshold: float | None = None
    passport: dict[str, Any] | None = None
    report_path: Path | None = None


@dataclass
class SimulationArtifacts:
    animation: Any
    recorder: Optional['LongitudinalMetricsRecorder'] = None
    calibration_threshold: float | None = None
    metrics: Optional['PerformanceMetrics'] = None
    dashboard: Optional['LiveDashboard'] = None
    engine: Optional['SimulationEngine'] = None


@dataclass
class Observation:
    values: tuple[float, ...]
    env_factor: float


class LiveDashboard:
    """Lightweight real-time metrics streaming panel during animation renders."""

    def __init__(self, total_frames: int, enabled: bool = True, is_notebook: bool = False) -> None:
        self.total_frames = total_frames
        self.enabled = enabled
        self.is_notebook = is_notebook
        self._display_handle: Any = None
        if self.enabled and self.is_notebook:
            try:
                from IPython.display import display

                self._display_handle = display(display_id=True)
            except (ImportError, RuntimeError, AttributeError):
                self._display_handle = None

    def update(
        self,
        frame: int,
        discoveries_this_frame: int,
        total_discoveries: int,
        modulation: float,
        is_coherence: bool,
        reg_deviation: float,
        sensor_ratios: Mapping[str, float],
        last_edge: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        coherence_tag = ' [COHERENCE ACTIVE]' if is_coherence else ''
        edge_str = f' | Last: {last_edge}' if last_edge else ''
        em_val = sensor_ratios.get('em_rf_short_long', 0.0)
        opt_val = sensor_ratios.get('optical_ir_short_long', 0.0)

        text = (
            f'[Live {frame + 1:>{len(str(self.total_frames))}}/{self.total_frames}] '
            f'Discoveries: {total_discoveries} (+{discoveries_this_frame}) | '
            f'Coherence: {modulation:.3f}{coherence_tag} | '
            f'REG Dev: {reg_deviation:+.3f} | '
            f'Sensors (EM/Opt): {em_val:.2f}/{opt_val:.2f}'
            f'{edge_str}'
        )

        if self._display_handle is not None:
            try:
                from IPython.display import HTML

                style = 'font-family:monospace; background:#1e1e1e; color:#eee; padding:8px; border-radius:4px;'
                self._display_handle.update(HTML(f'<pre style="{style}">{text}</pre>'))
                return
            except (ImportError, RuntimeError, AttributeError):
                self._display_handle = None

        import sys

        sys.stdout.write(f'\r{text}')
        sys.stdout.flush()

    def close(self) -> None:
        if not self.enabled:
            return
        if self._display_handle is None:
            import sys

            sys.stdout.write('\n')
            sys.stdout.flush()


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
    parser.add_argument('--seed', type=int, help='Random seed for deterministic simulation and reproducibility.')
    parser.add_argument('--headless', action='store_true', help='Run without rendering; write a JSON experiment report.')
    parser.add_argument('--control-study', type=positive_int, metavar='N',
                        help='Compare coupled and uncoupled synthetic fields over N paired seeds; write JSON.')
    parser.add_argument('--dynamics', choices=DYNAMICS_MODES,
                        help="Synthetic field generator: 'driven' (default) or 'legacy' to reproduce <=0.4 results.")
    parser.add_argument(
        '--live',
        action='store_true',
        help='Stream discoveries, environmental factor, and sensor telemetry live to the console or notebook during rendering.',
    )
    parser.add_argument(
        '--no-metrics-sidecar',
        action='store_true',
        help='Do not write the <output>.metrics.json summary next to the animation.',
    )
    parser.add_argument('--show', action='store_true', help='Also display inline when running in an IPython notebook.')
    args = parser.parse_args(argv)
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error('--seed must be between 0 and 4294967295')
    if args.control_study and (args.preset != 'synthetic' or args.input_data is not None):
        parser.error('--control-study supports synthetic fields only')
    if args.dynamics and (args.preset != 'synthetic' or args.input_data is not None):
        parser.error('--dynamics applies to synthetic fields only')
    if (args.headless or args.control_study) and args.show:
        parser.error('--show requires an animation output')
    if args.headless or args.control_study:
        if args.output == DEFAULT_OUTPUT:
            args.output = Path('outputs/control-study.json' if args.control_study else 'outputs/latest.json')
        if args.output.suffix.lower() != '.json':
            parser.error('headless and control-study output must use a .json suffix')
    elif args.output.suffix.lower() not in {'.html', '.gif'}:
        parser.error('animation output must use a .html or .gif suffix')
    return args


def apply_experiment(name: str | None) -> None:
    """Apply a named experiment bundle onto the global CFG in place."""
    if not name:
        return
    if name not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment {name!r}; choose from {sorted(EXPERIMENTS)}.')
    for key, value in EXPERIMENTS[name].items():
        setattr(CFG, key, value)


def apply_runtime_options(args: argparse.Namespace) -> CFG:
    # Precedence, lowest to highest: experiment bundle -> --seed -> --quick -> explicit flags.
    for key, value in DEFAULT_CONFIG.items():
        setattr(CFG, key, value)
    apply_experiment(getattr(args, 'experiment', None))
    if getattr(args, 'seed', None) is not None:
        CFG.SEED = args.seed
    set_seed(CFG.SEED)
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
    if getattr(args, 'dynamics', None) is not None:
        CFG.DYNAMICS = args.dynamics
    return CFG


def validate_configuration() -> None:
    """Reject settings the synthetic generator cannot run honestly."""
    if CFG.DYNAMICS not in DYNAMICS_MODES:
        raise ValueError(f'Unknown dynamics {CFG.DYNAMICS!r}; choose from {DYNAMICS_MODES}.')
    if not 0.0 <= CFG.FIELD_MEMORY < 1.0:
        raise ValueError('FIELD_MEMORY must lie in [0, 1) so the driven field stays stationary.')
    if CFG.FIELD_CORRELATION_LENGTH < 0:
        raise ValueError('FIELD_CORRELATION_LENGTH cannot be negative.')
    if len(CHANNEL_NAMES) != CFG.CHANNELS:
        raise ValueError(f'The synthetic generator defines exactly {len(CHANNEL_NAMES)} channels.')


def inactive_configuration(synthetic: bool = True) -> list[str]:
    """Configuration names that cannot influence this run."""
    inactive = list(UNUSED_SETTINGS)
    if not synthetic:
        inactive += ['DYNAMICS', *DRIVEN_SETTINGS]
    elif CFG.DYNAMICS == 'legacy':
        inactive += list(DRIVEN_SETTINGS)
    return inactive


def save_animation(animation: Any, output_path: Path, fps: int = DEFAULT_GIF_FPS, dashboard: LiveDashboard | None = None) -> Path:
    """Render the animation to ``output_path``.

    A ``.gif`` suffix writes a shareable clip via Pillow; anything else writes a
    self-contained interactive HTML animation. Both drive the same frame updates,
    so metrics are fully populated afterward either way.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output_path.suffix.lower() == '.gif':
            from matplotlib.animation import PillowWriter

            animation.save(str(output_path), writer=PillowWriter(fps=fps))
        else:
            output_path.write_text(animation.to_jshtml(), encoding='utf-8')
    finally:
        if dashboard is not None:
            dashboard.close()
        if hasattr(animation, '_fig'):
            import matplotlib.pyplot as plt

            plt.close(animation._fig)
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
    """Telemetry projected onto fixed spatial bases.

    Frame ``t`` of channel ``c`` is ``signals[t, c] * bases[c]``. Frames are projected on demand,
    so memory grows with rows plus one grid, not rows times grid.
    """

    signals: np.ndarray  # (frames, channels): standardized channel values
    bases: np.ndarray  # (channels, field_res, field_res): float32 spatial patterns
    timestamps: pd.Series
    raw_values: pd.DataFrame
    covariates: pd.DataFrame
    source: str = 'dataframe'
    quality_report: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._covariate_values = self.covariates[list(COVARIATE_NAMES)].to_numpy(dtype='float64')

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
        quality: dict[str, Any] = {'rows': len(df), 'imputed_sensor_cells': {}, 'defaulted_covariates': [],
                                   'preprocessing': 'offline forward fill then backward fill; full-series standardization'}
        for canonical, aliases in SENSOR_ALIASES.items():
            column = _find_column(df, aliases)
            if column is None or pd.to_numeric(df[column], errors='coerce').notna().sum() == 0:
                raise ValueError(f'Telemetry requires a numeric sensor column for {canonical}.')
            quality['imputed_sensor_cells'][canonical] = int(pd.to_numeric(df[column], errors='coerce').isna().sum())
            raw_values[canonical] = _filled_numeric_series(df, aliases).reset_index(drop=True)
        raw_values['control_baseline'] = rng.normal(0.0, 1.0, len(df))

        covariates = pd.DataFrame(index=range(len(df)))
        for canonical, aliases in COVARIATE_ALIASES.items():
            if _find_column(df, aliases) is None:
                quality['defaulted_covariates'].append(canonical)
            covariates[canonical] = _filled_numeric_series(df, aliases).reset_index(drop=True)
        if not np.isfinite(raw_values.to_numpy()).all() or not np.isfinite(covariates.to_numpy()).all():
            raise ValueError('Telemetry sensor and covariate values must be finite.')

        signals, bases = cls._map_to_grid(raw_values, field_res=field_res, rng=rng)
        return cls(signals=signals, bases=bases, timestamps=timestamps, raw_values=raw_values,
                   covariates=covariates, source=source, quality_report=quality)

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
                'em_rf': rng.normal(size=frame_count),
                'optical_ir': rng.normal(size=frame_count),
                'reg_variance': rng.normal(size=frame_count),
            }
        )
        return cls.from_dataframe(df, field_res=field_res, rng=rng, source=source)

    @staticmethod
    def _map_to_grid(raw_values: pd.DataFrame, field_res: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
        texture = rng.normal(0.0, 0.015, size=(len(CHANNEL_NAMES), field_res, field_res)).astype('float32')
        bases = _spatial_bases(field_res) + texture
        signals = np.column_stack([_normalize_values(raw_values[name]) for name in CHANNEL_NAMES])
        return signals, bases

    @property
    def fields(self) -> np.ndarray:
        """Every frame as one (frames, channels, res, res) array, materialized on request."""
        return (self.signals[:, :, None, None] * self.bases[None]).astype('float32')

    @property
    def frame_count(self) -> int:
        return int(self.signals.shape[0])

    @property
    def field_res(self) -> int:
        return int(self.bases.shape[-1])

    def field_for_frame(self, frame: int) -> np.ndarray:
        index = min(max(frame, 0), self.frame_count - 1)
        return (self.signals[index][:, None, None] * self.bases).astype('float32')

    def timestamp_for_frame(self, frame: int) -> pd.Timestamp:
        index = min(max(frame, 0), self.frame_count - 1)
        return pd.Timestamp(self.timestamps.iloc[index])

    def covariates_for_frame(self, frame: int) -> dict[str, float]:
        index = min(max(frame, 0), self.frame_count - 1)
        return {name: float(value) for name, value in zip(COVARIATE_NAMES, self._covariate_values[index], strict=True)}

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
        signals, bases = null_target.signals.copy(), null_target.bases.copy()
        signals[:, 3], bases[3] = self.signals[:, 3], self.bases[3]
        raw_values = null_target.raw_values.copy()
        raw_values['control_baseline'] = self.raw_values['control_baseline'].to_numpy()
        return TelemetryTargetField(
            signals=signals,
            bases=bases,
            timestamps=self.timestamps.copy(),
            raw_values=raw_values,
            covariates=self.covariates.copy(),
            source=f'{self.source}:control-only',
            quality_report={'generated_control': True, 'source_quality': self.quality_report},
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

def pairwise_correlations(windows: np.ndarray) -> np.ndarray:
    """Pearson correlation matrices for one or many observation windows.

    ``windows`` has shape (..., observations, channels); the result has shape (..., channels,
    channels). Constant channels correlate as exactly 0, never NaN. Agents and the field
    diagnostics share this one implementation.
    """
    windows = np.ascontiguousarray(windows, dtype='float64')
    centered = windows - windows.mean(axis=-2, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=-2))
    denominator = norms[..., :, None] * norms[..., None, :]
    return np.divide(np.swapaxes(centered, -1, -2) @ centered, denominator,
                     out=np.zeros_like(denominator), where=denominator > 1e-24)


def channel_pairs(channels: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(channels) for j in range(i + 1, channels)]


def pair_name(i: int, j: int) -> str:
    return f'ch{i}--ch{j}'


class FieldDiagnostics:
    """What the synthetic field actually contained, measured over every pixel.

    Agents see a few samples; these values are the ground truth they are trying to recover:
    the cross-channel correlation of the whole field each frame, how much of the field survives
    one correlation window, and how smooth it is between neighbouring pixels.
    """

    def __init__(self, window: int) -> None:
        self.window = max(int(window), 1)
        self.pairs: list[tuple[int, int]] = []
        self.pair_correlations: list[list[float]] = []
        self.window_memory: list[float] = []
        self.neighbour_correlation: list[float] = []
        self._anchor: np.ndarray | None = None

    def log(self, frame: int, field: np.ndarray) -> None:
        field = np.asarray(field, dtype='float64')
        self.pairs = channel_pairs(field.shape[0])
        correlations = pairwise_correlations(field.reshape(field.shape[0], -1).T)
        self.pair_correlations.append([float(correlations[i, j]) for i, j in self.pairs])
        if frame % self.window:
            return
        if self._anchor is not None:
            self.window_memory.append(_mean_pattern_correlation(field, self._anchor))
        self.neighbour_correlation.append(0.5 * (_mean_pattern_correlation(field, np.roll(field, 1, axis=1))
                                                 + _mean_pattern_correlation(field, np.roll(field, 1, axis=2))))
        self._anchor = field.copy()

    def summary(self) -> dict[str, Any]:
        history = np.asarray(self.pair_correlations, dtype='float64')

        def mean(values: Sequence[float]) -> float | None:
            return float(np.mean(values)) if len(values) else None

        return {
            'frames': len(self.pair_correlations),
            'pair_correlation_mean': {pair_name(i, j): mean(history[:, k]) for k, (i, j) in enumerate(self.pairs)}
            if history.size else {},
            'window_memory': mean(self.window_memory),
            'neighbour_correlation': mean(self.neighbour_correlation),
        }


def _mean_pattern_correlation(field: np.ndarray, reference: np.ndarray) -> float:
    """Mean over channels of the pixelwise correlation between two field snapshots."""
    channels = field.shape[0]
    stacked = np.concatenate([field.reshape(channels, -1), reference.reshape(channels, -1)]).T
    correlations = pairwise_correlations(stacked)
    return float(np.mean([correlations[c, channels + c] for c in range(channels)]))


class PerformanceMetrics:
    def __init__(self) -> None:
        self.type_counts: dict[str, int] = defaultdict(int)
        self.env_history: list[float] = []
        self.discovery_rate_history: list[int] = []
        self.coherence_frames: list[int] = []
        self.edge_counts: dict[str, int] = defaultdict(int)
        self.evaluated_windows = 0  # agent-windows evaluated per channel pair
        self.abs_r_histogram: np.ndarray | None = None  # (pairs, HISTOGRAM_BINS) counts of |r|
        self.field: FieldDiagnostics | None = None

    def log_discovery(self, agent_type: str, edge: tuple[str, str] | None = None) -> None:
        self.type_counts[agent_type] += 1
        if edge is not None:
            self.edge_counts['--'.join(sorted(edge))] += 1

    def log_discoveries(self, agent_types: Iterable[str], edges: Iterable[str]) -> None:
        """Bulk form of log_discovery; ``edges`` are already-sorted 'chI--chJ' names."""
        for agent_type, count in Counter(agent_types).items():
            self.type_counts[agent_type] += count
        for edge, count in Counter(edges).items():
            self.edge_counts[edge] += count

    def log_evaluation(self, strengths: np.ndarray) -> None:
        """Record every evaluated |r|, detected or not: shape (agents, pairs)."""
        agents, pairs = strengths.shape
        if self.abs_r_histogram is None:
            self.abs_r_histogram = np.zeros((pairs, HISTOGRAM_BINS), dtype=np.int64)
        bins = np.minimum((strengths * HISTOGRAM_BINS).astype(np.int64), HISTOGRAM_BINS - 1)
        flat = (bins + HISTOGRAM_BINS * np.arange(pairs)).ravel()
        self.abs_r_histogram += np.bincount(flat, minlength=pairs * HISTOGRAM_BINS).reshape(pairs, HISTOGRAM_BINS)
        self.evaluated_windows += agents

    def log_frame(self, frame: int, discoveries: int, env_factor: float, is_coherence: bool) -> None:
        self.env_history.append(env_factor)
        self.discovery_rate_history.append(discoveries)
        if is_coherence:
            self.coherence_frames.append(frame)

    @property
    def total_discoveries(self) -> int:
        return int(sum(self.type_counts.values()))


def _event_within(events: Sequence[int], t: int, radius: float, inclusive: bool) -> bool:
    """Whether any event in the sorted ``events`` lies within ``radius`` frames of ``t``.

    Bisection keeps this O(log n); empirical runs can log an event on every frame.
    """
    index = bisect.bisect_left(events, t)
    distances = [abs(t - events[i]) for i in (index - 1, index) if 0 <= i < len(events)]
    if not distances:
        return False
    return min(distances) <= radius if inclusive else min(distances) < radius


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
        return _event_within(self.coherence_events, t, 15, inclusive=False)

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
        self.coherence_window = round(self.base_window * (1.0 + 0.75 * pressure))
        if pressure >= 0.35 or not self.coherence_events:
            self.coherence_events.append(t)

    def get_modulation(self, t: int) -> float:
        return float(self.m * (1.15 if self.is_coherence_active(t) else 1.0))

    def is_coherence_active(self, t: int) -> bool:
        return _event_within(self.coherence_events, t, self.coherence_window, inclusive=True)

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
        correlations = pairwise_correlations([o.values for o in self.memory[-CFG.CORR_WINDOW :]])
        discs: list[dict[str, Any]] = []
        for i, j in channel_pairs(CFG.CHANNELS):
            r = float(np.clip(correlations[i, j], -1.0, 1.0))
            if abs(r) > threshold:
                discs.append({'edge': (f'ch{i}', f'ch{j}'), 'pearson_r': r, 'confidence': abs(r), 'operator_type': self.type})
        return discs


class ObservationBuffer:
    """Ring buffer holding the latest ``capacity`` observations of every agent at once.

    The engine samples, stores and correlates all agents in single array operations instead
    of one Python call per agent per frame.
    """

    def __init__(self, capacity: int, agents: int, channels: int) -> None:
        self.capacity = max(int(capacity), 0)
        self.values = np.zeros((self.capacity, agents, channels), dtype='float64')
        self.env_factors = np.zeros(self.capacity, dtype='float64')
        self.size = 0
        self._next = 0

    def append(self, samples: np.ndarray, env_factor: float) -> None:
        if not self.capacity:
            return
        self.values[self._next] = samples
        self.env_factors[self._next] = env_factor
        self._next = (self._next + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def slots(self, count: int) -> np.ndarray:
        """Buffer rows of the latest ``count`` observations, oldest first."""
        return (self._next - count + np.arange(count)) % self.capacity

    def windows(self, count: int) -> np.ndarray:
        """The latest ``count`` observations per agent: shape (agents, count, channels)."""
        return self.values[self.slots(count)].transpose(1, 0, 2)


class AgentMemoryView(Sequence):
    """Read-only memory of one engine agent, backed by the shared ObservationBuffer.

    The engine records observations for its agents; ``Agent.observe`` is for standalone agents.
    """

    def __init__(self, buffer: ObservationBuffer, index: int) -> None:
        self._buffer = buffer
        self._index = index

    def __len__(self) -> int:
        return self._buffer.size

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, slice):
            return [self[i] for i in range(*item.indices(len(self)))]
        size = len(self)
        position = item + size if item < 0 else item
        if not 0 <= position < size:
            raise IndexError('agent memory index out of range')
        slot = int(self._buffer.slots(size)[position])
        values = tuple(float(v) for v in self._buffer.values[slot, self._index])
        return Observation(values=values, env_factor=float(self._buffer.env_factors[slot]))


class LongitudinalMetricsRecorder:
    def __init__(self, output_dir: Path = DEFAULT_METRICS_DIR, run_id: str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id or f'{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}'
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
            'row_count': len(dataframe),
            'discovery_rows': int((dataframe.get('Confidence_Score', pd.Series(dtype=float)) > 0).sum()),
            'csv_path': str(csv_path),
        }
        metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
        return csv_path, metadata_path


@lru_cache(maxsize=16)
def _frequency_grid(field_res: int) -> Any:
    """Squared spatial frequency, in cycles per pixel, on the rfft2 grid."""
    return xp.fft.fftfreq(field_res).reshape(field_res, 1) ** 2 + xp.fft.rfftfreq(field_res).reshape(1, -1) ** 2


@lru_cache(maxsize=16)
def _forcing_filter(field_res: int, correlation_length: float) -> Any:
    """Gaussian spectral filter: white noise in, unit-variance noise with the requested spatial scale out."""
    scale = -0.5 * (2.0 * math.pi * correlation_length) ** 2
    full_grid = xp.fft.fftfreq(field_res)[:, None] ** 2 + xp.fft.fftfreq(field_res)[None, :] ** 2
    return xp.exp(scale * _frequency_grid(field_res)) / xp.sqrt(xp.mean(xp.exp(scale * full_grid) ** 2))


def correlated_noise(noise: Any) -> Any:
    """Smooth white noise to the configured correlation length, keeping unit variance."""
    res = noise.shape[-1]
    return ifft_irfft(fft_rfft(noise) * _forcing_filter(res, float(CFG.FIELD_CORRELATION_LENGTH)), s=(res, res))


def evolve_fields(F: Any, t: int, env_factor: float, env_mod: EnvironmentalModerators, coupled: bool = True) -> Any:
    """Legacy (<=0.4) synthetic dynamics, kept bit-for-bit reproducible. Do not use for new work.

    Its diffusion kernel omits (2*pi)^2, so it is inert, and its update has gain 1.24 near zero,
    so every pixel grows until it saturates near +-1: within about 60 frames the field is frozen
    salt-and-pepper noise. One coherence window copies channel 2 into channel 3 permanently.
    """
    diffusion = xp.asarray(LEGACY_DIFFUSION)[:, None, None]
    Fk = fft_rfft(F)
    Fk *= xp.exp(-diffusion * _frequency_grid(CFG.FIELD_RES) * 0.5 * env_factor)
    F = ifft_irfft(Fk, s=(CFG.FIELD_RES, CFG.FIELD_RES))
    if coupled:
        F[0] += 0.045 * F[1] * env_factor
    if coupled and env_mod.is_coherence_active(t):
        F[3] = 0.7 * F[3] + 0.3 * F[2]
    return 0.92 * F + 0.08 * xp.tanh(F * 4.0)


def evolve_driven_fields(F: Any, t: int, env_factor: float, env_mod: EnvironmentalModerators, noise: Any,
                         coupled: bool = True) -> Any:
    """Advance the driven-dissipative field by one frame (default since 0.5).

    Every channel relaxes toward zero with persistence FIELD_MEMORY, diffuses through a true
    (2*pi*k)^2 Laplacian scaled by the moderator factor, and is renewed by spatially correlated
    forcing. The field therefore stays stationary, structured and changing. A gentle saturation
    keeps it nonlinear but stable. The paired arms receive identical ``noise``; only the two
    injected terms differ, and coherence mixing fades once its window closes.
    """
    res = F.shape[-1]
    memory = float(CFG.FIELD_MEMORY)
    diffusion = xp.asarray(DRIVEN_DIFFUSION)[:, None, None]
    propagator = memory * xp.exp(-diffusion * (2.0 * math.pi) ** 2 * _frequency_grid(res) * env_factor)
    forcing = math.sqrt(1.0 - memory**2) * fft_rfft(noise) * _forcing_filter(res, float(CFG.FIELD_CORRELATION_LENGTH))
    F = ifft_irfft(fft_rfft(F) * propagator + forcing, s=(res, res))
    if coupled:
        F[0] += CFG.COUPLING_STRENGTH * env_factor * F[1]
        if env_mod.is_coherence_active(t):
            F[3] = (1.0 - CFG.COHERENCE_MIXING) * F[3] + CFG.COHERENCE_MIXING * F[2]
    return (F + 0.1 * (xp.tanh(F) - F)).astype(xp.float32)


def build_simulation_state(
    target_field: TelemetryTargetField | None = None,
) -> tuple[list[Agent], Any, dict[str, float], PerformanceMetrics, Any, Any]:
    import networkx as nx

    agents = [Agent(i, CFG.AGENT_TYPES[i % 3]) for i in range(CFG.AGENTS)]
    graph = nx.Graph()
    graph.add_nodes_from(f'ch{i}' for i in range(CFG.CHANNELS))
    conf_map: dict[str, float] = defaultdict(float)
    metrics = PerformanceMetrics()
    if target_field is None:
        env_mod: Any = EnvironmentalModerators()
        shape = (CFG.CHANNELS, CFG.FIELD_RES, CFG.FIELD_RES)
        if CFG.DYNAMICS == 'legacy':
            F = xp.asarray(rng.normal(0, 0.02, shape), dtype=xp.float32)
        else:  # Start from the forcing's stationary texture, so there is no burn-in transient.
            F = correlated_noise(xp.asarray(rng.normal(0.0, 1.0, shape))).astype(xp.float32)
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

class SimulationEngine:
    """Sequential simulation independent of visualization.

    A duplicate request for the current frame is idempotent. Earlier or skipped
    frames are rejected. Configure CFG before construction and keep it unchanged
    until the run completes; engines currently run sequentially, not concurrently.

    Agents move, sample and correlate as one batch per frame. Random draws happen
    in the same order as the original per-agent loop, so results are unchanged.
    """

    def __init__(self, target_field=None, recorder=None, calibrated_threshold=None,
                 live_dashboard=None, coupled=True):
        if target_field is None:
            validate_configuration()
        self.target_field = target_field
        self.recorder = recorder
        self.calibrated_threshold = calibrated_threshold
        self.live_dashboard = live_dashboard
        self.coupled = coupled
        self.agents, self.graph, self.conf_map, self.metrics, self.env_mod, self.F = build_simulation_state(target_field)
        self.frame_count = CFG.FRAMES if target_field is None else min(CFG.FRAMES, target_field.frame_count)
        self.current_frame = -1
        self.snapshot = None
        # Agents keep their public attributes, but positions and memories live in shared arrays.
        self.positions = np.array([agent.pos for agent in self.agents], dtype=np.int64).reshape(-1, 2)
        self.observations = ObservationBuffer(CFG.MEMORY, len(self.agents), CFG.CHANNELS)
        for index, agent in enumerate(self.agents):
            agent.pos = self.positions[index]
            agent.memory = AgentMemoryView(self.observations, index)
        self._agent_types = [agent.type for agent in self.agents]
        self._pairs = channel_pairs(CFG.CHANNELS)
        self._pair_names = [pair_name(i, j) for i, j in self._pairs]
        self._pair_rows, self._pair_cols = (np.array(axis, dtype=np.intp) for axis in zip(*self._pairs, strict=True))
        if target_field is None:
            self.metrics.field = FieldDiagnostics(CFG.CORR_WINDOW)

    def step(self, frame: int) -> dict[str, Any]:
        if frame == self.current_frame and self.snapshot is not None:
            return self.snapshot
        if frame != self.current_frame + 1 or not 0 <= frame < self.frame_count:
            raise ValueError('Frames must advance sequentially within the configured run.')
        target_field, env_mod, metrics = self.target_field, self.env_mod, self.metrics
        if target_field is None:
            env_mod.update(frame)
            modulation = env_mod.get_modulation(frame)
            if CFG.DYNAMICS == 'legacy':
                F = evolve_fields(self.F, frame, modulation, env_mod, coupled=self.coupled)
            else:
                noise = xp.asarray(rng.normal(0.0, 1.0, self.F.shape))
                F = evolve_driven_fields(self.F, frame, modulation, env_mod, noise, coupled=self.coupled)
            F_cpu = cp.asnumpy(F) if on_gpu else F
            timestamp = pd.Timestamp('1970-01-01', tz='UTC') + pd.Timedelta(minutes=frame)
            reg_deviation = _synthetic_reg_deviation(F_cpu)
            sensor_ratios = _synthetic_sensor_ratios(F_cpu)
            metrics.field.log(frame, F_cpu)
        else:
            F_cpu = target_field.field_for_frame(frame)
            F = xp.asarray(F_cpu, dtype=xp.float32)
            env_mod.update(frame, target_field.covariates_for_frame(frame))
            modulation = env_mod.get_modulation(frame)
            timestamp = target_field.timestamp_for_frame(frame)
            reg_deviation = target_field.reg_variance_deviation(frame, CFG.CORR_WINDOW)
            sensor_ratios = target_field.sensor_anomaly_ratios(frame)
        threshold = env_mod.discovery_threshold if self.calibrated_threshold is None else self.calibrated_threshold
        is_coherence = env_mod.is_coherence_active(frame)

        frame_discoveries, detections, last_edge, shares = self._sense(F_cpu, modulation, threshold)
        self._decay_associations(env_mod.confidence_decay)
        metrics.log_frame(frame, detections, modulation, is_coherence)
        if self.recorder is not None:
            self.recorder.log_frame(
                frame=frame,
                timestamp=timestamp,
                discoveries=frame_discoveries,
                moderator_values=env_mod.snapshot(),
                operator_density=len(self.agents) / float(CFG.FIELD_RES * CFG.FIELD_RES),
                total_discoveries=metrics.total_discoveries,
                reg_variance_deviation=reg_deviation,
                sensor_anomaly_ratios=sensor_ratios,
            )
        if self.live_dashboard is not None:
            self.live_dashboard.update(
                frame=frame,
                discoveries_this_frame=detections,
                total_discoveries=metrics.total_discoveries,
                modulation=modulation,
                is_coherence=is_coherence,
                reg_deviation=reg_deviation,
                sensor_ratios=sensor_ratios,
                last_edge=last_edge,
            )

        self.F = F
        self.current_frame = frame
        self.snapshot = {
            'field': F_cpu, 'modulation': modulation, 'reg_deviation': reg_deviation,
            'sensor_ratios': sensor_ratios, 'coherence': is_coherence,
            'detection_share': None if shares is None else dict(zip(self._pair_names, shares.tolist(), strict=True)),
        }
        return self.snapshot

    def _sense(self, field: np.ndarray, env_factor: float,
               threshold: float) -> tuple[list[dict[str, Any]], int, str | None, np.ndarray | None]:
        """Move every agent, record its sample, and evaluate all six pairs for every full window.

        Returns this frame's discovery records (only when recording), the detection count, the last
        detected pair, and the share of agents detecting each pair (None until windows are full).
        """
        steps = rng.randint(-CFG.STEP_SIZE, CFG.STEP_SIZE + 1, size=self.positions.shape)
        self.positions[:] = (self.positions + steps) % CFG.FIELD_RES
        self.observations.append(field[:, self.positions[:, 0], self.positions[:, 1]].T, env_factor)
        window = CFG.CORR_WINDOW
        if self.observations.size < window or window > self.observations.capacity:
            return [], 0, None, None
        correlations = np.clip(pairwise_correlations(self.observations.windows(window))[:, self._pair_rows, self._pair_cols],
                               -1.0, 1.0)
        strengths = np.abs(correlations)
        self.metrics.log_evaluation(strengths)
        detected = strengths > threshold
        shares = detected.mean(axis=0) if len(self.agents) else np.zeros(len(self._pairs))
        agent_index, pair_index = np.nonzero(detected)  # agent-major, like the original loop
        if not agent_index.size:
            return [], 0, None, shares
        agents, pairs = agent_index.tolist(), pair_index.tolist()
        self.metrics.log_discoveries((self._agent_types[a] for a in agents), (self._pair_names[p] for p in pairs))
        for p in dict.fromkeys(pairs):  # first-detection order keeps graph insertion order unchanged
            i, j = self._pairs[p]
            key = f'ch{i}->ch{j}'
            self.conf_map[key] = max(self.conf_map[key], float(strengths[:, p].max()))
            self.graph.add_edge(f'ch{i}', f'ch{j}', weight=self.conf_map[key])
        discoveries = []
        if self.recorder is not None:
            discoveries = [
                {'edge': (f'ch{self._pairs[p][0]}', f'ch{self._pairs[p][1]}'), 'pearson_r': float(correlations[a, p]),
                 'confidence': float(strengths[a, p]), 'operator_type': self._agent_types[a]}
                for a, p in zip(agents, pairs, strict=True)
            ]
        return discoveries, len(agents), self._pair_names[pairs[-1]], shares

    def _decay_associations(self, confidence_decay: float) -> None:
        for key in list(self.conf_map.keys()):
            self.conf_map[key] *= confidence_decay
            u, v = key.split('->')
            if self.conf_map[key] < 0.05:
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)
                del self.conf_map[key]
            else:
                self.graph[u][v]['weight'] = self.conf_map[key]

    def run(self) -> 'SimulationEngine':
        for frame in range(self.current_frame + 1, self.frame_count):
            self.step(frame)
        return self


def run_simulation(
    target_field: TelemetryTargetField | None = None,
    preset: str = 'synthetic',
    recorder: LongitudinalMetricsRecorder | None = None,
    calibrated_threshold: float | None = None,
    live: bool = False,
    live_dashboard: LiveDashboard | None = None,
    is_notebook: bool = False,
) -> SimulationArtifacts:
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib import gridspec
    from matplotlib.animation import FuncAnimation

    plt.rcParams['animation.embed_limit'] = 200
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8), facecolor='#081b20')
    gs = gridspec.GridSpec(2, 2, width_ratios=(1, 1.5), height_ratios=(1, 0.65),
                          left=0.05, right=0.96, top=0.84, bottom=0.08, hspace=0.32, wspace=0.2)
    ax_graph = fig.add_subplot(gs[:, 1])
    ax_field = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[1, 0])
    fig.text(0.05, 0.94, 'IONS-X / DEEP EMERGENCE LAB', fontsize=20, weight='bold', color='#b9f5e8')
    source = f'{CFG.DYNAMICS} field' if target_field is None else 'telemetry projection'
    fig.text(0.05, 0.89, f'{preset.upper()}  ·  {source}  ·  seed {CFG.SEED}  ·  {CFG.AGENTS} operators  ·  '
             'exploratory associations', fontsize=11, color='#93b5b5')
    fig.text(0.05, 0.025, 'Edge width and label: share of operators whose current window detects the pair  /  '
             'Correlation does not establish causal direction.', fontsize=10, color='#93b5b5')

    frames_to_render = CFG.FRAMES if target_field is None else min(CFG.FRAMES, target_field.frame_count)

    if live_dashboard is None and live:
        live_dashboard = LiveDashboard(total_frames=frames_to_render, enabled=True, is_notebook=is_notebook)

    engine = SimulationEngine(target_field, recorder, calibrated_threshold, live_dashboard)
    graph, metrics = engine.graph, engine.metrics
    positions = nx.circular_layout(graph)
    color_limit = [1e-9]  # only ever widens, so the colour scale never flickers between frames

    def update(frame: int) -> None:

        snapshot = engine.step(frame)
        F_cpu = snapshot['field']
        modulation = snapshot['modulation']
        reg_deviation = snapshot['reg_deviation']
        sensor_ratios = snapshot['sensor_ratios']

        color_limit[0] = max(color_limit[0], float(np.percentile(np.abs(F_cpu[0]), 99.5)))
        ax_field.clear()
        ax_field.imshow(F_cpu[0], cmap='magma', vmin=-color_limit[0], vmax=color_limit[0])
        ax_field.scatter(engine.positions[:, 1], engine.positions[:, 0], s=7, c='#b9f5e8', alpha=0.7, linewidths=0)
        ax_field.set_title(f'CH 0 / EM-RF field + operators  ·  frame {frame + 1}/{frames_to_render}', fontsize=12)
        ax_field.axis('off')

        # The shared graph keeps each pair's strongest recent |r| from any operator; with many operators,
        # chance detections keep every edge alive. Consensus across operators is what separates them.
        ax_graph.clear()
        ax_graph.set_facecolor('#081b20')
        shares = snapshot['detection_share'] or {}
        consensus = nx.Graph()
        consensus.add_nodes_from(graph.nodes)
        consensus.add_edges_from((*name.split('--'), {'share': share}) for name, share in shares.items() if share > 0)
        edges = list(consensus.edges(data='share'))
        nx.draw_networkx_nodes(consensus, positions, ax=ax_graph, node_color='#39c5bb', node_size=2700)
        nx.draw_networkx_labels(consensus, positions, labels=CHANNEL_LABELS, ax=ax_graph, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(consensus, positions, ax=ax_graph, edge_color='#91e1da', node_size=2700,
                               width=[1 + 16 * share for *_, share in edges], alpha=0.9)
        nx.draw_networkx_edge_labels(consensus, positions, ax=ax_graph, font_size=9, font_color='#eaf4f3', label_pos=0.3,
                                     edge_labels={(u, v): f'{share:.0%}' for u, v, share in edges},
                                     bbox={'boxstyle': 'round', 'fc': '#081b20', 'ec': 'none'})
        ax_graph.set_xlim(-1.5, 1.5)
        ax_graph.set_ylim(-1.5, 1.5)
        ax_graph.set_axis_off()
        title = 'filling correlation windows' if snapshot['detection_share'] is None else 'operator consensus this frame'
        ax_graph.set_title(f'ASSOCIATION MAP  /  {title}', fontsize=12)

        ax_stats.clear()
        ax_stats.axis('off')
        ax_stats.text(
            0.05,
            0.75,
            (
                f'Association detections     {metrics.total_discoveries:,}\n'
                f'Unique channel pairs       {len(metrics.edge_counts)} / 6\n'
                f'Moderator factor           {modulation:.3f}\n'
                f"Coherence window           {'active' if snapshot['coherence'] else '-'}\n"
                f'REG diagnostic             {reg_deviation:.3f}\n'
                f"EM / optical diagnostics   {sensor_ratios['em_rf_short_long']:.2f} / "
                f"{sensor_ratios['optical_ir_short_long']:.2f}"
            ),
            fontsize=10,
            fontfamily='monospace',
            linespacing=1.7,
            va='top',
        )
        ax_stats.set_title('OBSERVATION LOG  /  repeated detections are not independent', fontsize=9)

    animation = FuncAnimation(fig, update, init_func=lambda: (), frames=frames_to_render,
                              interval=50, repeat=False, cache_frame_data=False)
    return SimulationArtifacts(
        animation=animation,
        recorder=recorder,
        calibration_threshold=calibrated_threshold,
        metrics=metrics,
        dashboard=live_dashboard,
        engine=engine,
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


def experiment_passport(input_path: Path | None = None, synthetic: bool = True) -> dict[str, Any]:
    """Capture the effective configuration, implementation and input identities."""
    return {
        'version': __version__,
        'configuration': {key: getattr(CFG, key) for key in DEFAULT_CONFIG},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'report_source_sha256': hashlib.sha256(Path(__file__).with_name('study_report.py').read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest() if input_path is not None else None,
        'python': platform.python_version(),
        'dependencies': {name: version(name) for name in ('numpy', 'scipy', 'pandas', 'networkx', 'matplotlib')},
        'backend': 'GPU' if on_gpu else 'CPU',
        'interpretation': 'Exploratory associations; confidence is absolute Pearson r, not a probability or causal evidence.',
        'inactive_configuration': inactive_configuration(synthetic),
    }


# A numeric array as json.dumps(indent=...) lays it out. JSON strings cannot hold raw newlines,
# so requiring them means only formatting, never string content, is ever collapsed.
_INDENTED_NUMBER_ARRAY = re.compile(r'\[\n\s*(-?[0-9][0-9.eE+-]*(?:,\n\s*-?[0-9][0-9.eE+-]*)*)\n\s*\]')


def write_report(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write indented, key-sorted JSON with each numeric array kept on one line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    text = _INDENTED_NUMBER_ARRAY.sub(lambda match: '[' + ', '.join(item.strip() for item in match.group(1).split(',')) + ']', text)
    path.write_text(text, encoding='utf-8')
    return path


def nominal_false_alarm_rate(window: int, threshold: float) -> float | None:
    """P(|r| > threshold) for ``window`` independent samples of two independent Gaussian channels.

    This is the rate a reader would assume from the threshold alone. Under that null,
    r * sqrt(n - 2) / sqrt(1 - r^2) follows Student's t with n - 2 degrees of freedom, so the
    two-sided tail is the regularized incomplete beta I_{1 - threshold^2}((n - 2) / 2, 1 / 2).
    """
    if window < 3:
        return None
    if threshold >= 1:
        return 0.0
    if threshold <= 0:
        return 1.0
    from scipy.special import betainc

    return float(betainc((window - 2) / 2, 0.5, 1.0 - threshold**2))


def effective_samples(false_alarm_rate: float | None, threshold: float) -> float | None:
    """Independent samples per window that would produce ``false_alarm_rate`` at ``threshold``.

    Autocorrelated samples make a window behave like fewer independent ones; this inverts the
    nominal rate to say how many.
    """
    if false_alarm_rate is None or not 0 < false_alarm_rate < 1 or not 0 < threshold < 1:
        return None
    from scipy.special import betainc

    def rate(samples: float) -> float:
        return float(betainc((samples - 2) / 2, 0.5, 1.0 - threshold**2))

    low, high = 2.0 + 1e-9, 1e6
    if rate(high) > false_alarm_rate:
        return None
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if rate(middle) > false_alarm_rate else (low, middle)
    return 0.5 * (low + high)


def calibrate_detector(runs: Sequence[Mapping[str, Any]], pairs: Sequence[str], window: int, threshold: float) -> dict[str, Any]:
    """Compare the detector's nominal false-alarm rate with what the uncoupled arm actually produced.

    Every pair in the uncoupled arm is unrelated by construction, so its detections are false
    alarms. A held-out split then asks which threshold would restore the nominal rate, choosing it
    on even-indexed seeds and judging it only on the odd-indexed seeds it never saw.
    """
    nominal = nominal_false_alarm_rate(window, threshold)
    windows = sum(run['uncoupled']['evaluated_windows'] for run in runs)
    by_pair = {pair: sum(run['uncoupled']['edge_counts'][pair] for run in runs) / windows if windows else None
               for pair in pairs}
    observed = math.fsum(by_pair.values()) / len(pairs) if windows else None
    calibration = {
        'threshold': threshold,
        'correlation_window': window,
        'nominal_false_alarm_rate': nominal,
        'observed_false_alarm_rate': observed,
        'false_alarm_inflation': observed / nominal if observed is not None and nominal else None,
        'effective_samples_per_window': effective_samples(observed, threshold),
        'false_alarm_rate_by_pair': by_pair,
        'held_out': None,
    }
    tuning, evaluation = list(runs[0::2]), list(runs[1::2])
    if not (tuning and evaluation and windows and nominal):
        return calibration
    cut = threshold_bin_for_rate(pooled_histogram(tuning, 'uncoupled', pairs), nominal)
    evaluation_windows = sum(run['coupled']['evaluated_windows'] for run in evaluation)
    calibration['held_out'] = {
        'calibration_seeds': [run['seed'] for run in tuning],
        'evaluation_seeds': [run['seed'] for run in evaluation],
        'threshold_for_nominal_rate': cut / HISTOGRAM_BINS,
        'false_alarm_rate': tail_count(pooled_histogram(evaluation, 'uncoupled', pairs), cut)
        / (len(pairs) * sum(run['uncoupled']['evaluated_windows'] for run in evaluation)),
        'sensitivity_at_default': {pair: sum(run['coupled']['edge_counts'][pair] for run in evaluation) / evaluation_windows
                                   for pair in INJECTED_ASSOCIATIONS},
        'sensitivity': {pair: tail_count(pooled_histogram(evaluation, 'coupled', [pair]), cut) / evaluation_windows
                        for pair in INJECTED_ASSOCIATIONS},
    }
    return calibration


def run_control_study(repeats: int, seed: int) -> dict[str, Any]:
    """Paired synthetic coupling ablation; repetitions, not agents, are the units.

    Both arms replay the same random draws. Only the two explicit cross-channel
    coupling terms differ. Seeds are independent replicates of this generator, so
    seed-level statistics are valid for it; they are not a validated null model for
    empirical telemetry.
    """
    if repeats < 1 or seed < 0 or seed + repeats > 2**32:
        raise ValueError('Paired seeds must fit in the RandomState seed range.')
    if CFG.FRAMES < CFG.CORR_WINDOW:
        raise ValueError('Control studies require at least CORR_WINDOW frames to evaluate associations.')
    validate_configuration()
    saved_seed, saved_state = CFG.SEED, rng.get_state()
    rows = []
    pairs = [pair_name(i, j) for i, j in channel_pairs(CFG.CHANNELS)]
    try:
        for run_seed in range(seed, seed + repeats):
            arms = {}
            for name, coupled in (('coupled', True), ('uncoupled', False)):
                set_seed(run_seed)
                metrics = SimulationEngine(coupled=coupled).run().metrics
                histogram = metrics.abs_r_histogram if metrics.abs_r_histogram is not None else np.zeros((len(pairs), HISTOGRAM_BINS))
                arms[name] = {
                    'total_discoveries': metrics.total_discoveries,
                    'edge_counts': {pair: metrics.edge_counts.get(pair, 0) for pair in pairs},
                    'coherence_frames': list(metrics.coherence_frames),
                    'evaluated_windows': metrics.evaluated_windows,
                    'abs_r_histogram': {pair: [int(count) for count in histogram[k]] for k, pair in enumerate(pairs)},
                    'field_diagnostics': metrics.field.summary(),
                }
            rows.append({'seed': run_seed, **arms,
                         'paired_difference': arms['coupled']['total_discoveries'] - arms['uncoupled']['total_discoveries']})
    finally:
        CFG.SEED = saved_seed
        rng.set_state(saved_state)
    opportunities = CFG.AGENTS * (CFG.FRAMES - CFG.CORR_WINDOW + 1)
    differences = [row['paired_difference'] for row in rows]

    def field_mean(arm: str, key: str, pair: str | None = None) -> float | None:
        values = [row[arm]['field_diagnostics'][key] if pair is None else row[arm]['field_diagnostics'][key][pair]
                  for row in rows]
        values = [value for value in values if value is not None]
        return float(np.mean(values)) if values else None

    return {
        'schema_version': 3,
        'kind': 'paired_synthetic_coupling_ablation',
        'dynamics': CFG.DYNAMICS,
        'passport': experiment_passport(),
        'repetitions': repeats,
        'starting_seed': seed,
        'opportunities_per_pair_per_run': opportunities,
        'injected_associations': dict(INJECTED_ASSOCIATIONS),
        'mean_paired_difference': float(np.mean(differences)),
        'paired_difference_range': [min(differences), max(differences)],
        'association_rates': {
            pair: {arm: float(np.mean([row[arm]['edge_counts'][pair] / opportunities for row in rows]))
                   for arm in ('coupled', 'uncoupled')}
            for pair in pairs
        },
        'paired_association_effects': paired_association_summary(rows, opportunities),
        'detector_calibration': calibrate_detector(rows, pairs, CFG.CORR_WINDOW, float(CFG.DISCOVER_THRESH)),
        'discrimination': discrimination_summary(rows),
        'field_ground_truth': {
            'pair_correlation_mean': {pair: {arm: field_mean(arm, 'pair_correlation_mean', pair) for arm in ('coupled', 'uncoupled')}
                                      for pair in pairs},
            'window_memory': {arm: field_mean(arm, 'window_memory') for arm in ('coupled', 'uncoupled')},
            'neighbour_correlation': {arm: field_mean(arm, 'neighbour_correlation') for arm in ('coupled', 'uncoupled')},
        },
        'runs': rows,
        'limitations': [
            ('Seeds, not agents or windows, are the independent units. Sign-flip p-values are exact over seeds and '
             'Holm-adjusted across the six pairs; they describe this generator and configuration only.'),
            'Agents and overlapping windows observing one field are dependent; their counts are never used as sample sizes.',
            'The nominal false-alarm rate assumes independent samples. The uncoupled arm measures the real rate.',
            'The uncoupled arm preserves field evolution, forcing noise, moderators, agent paths and all random draws.',
            'This measures detector sensitivity to injected coupling, not causal discovery on real-world data.',
            'Operator types are labels for one shared detector; lagged detection is not implemented.',
        ],
    }


def build_run_summary(result: RunResult, metrics: 'PerformanceMetrics') -> dict[str, Any]:
    """Assemble the lightweight per-run summary written beside the animation."""
    config = (result.passport or {}).get('configuration', {})
    window = config.get('CORR_WINDOW', CFG.CORR_WINDOW)
    channels = config.get('CHANNELS', CFG.CHANNELS)
    eligible = metrics.evaluated_windows
    return {
        'schema_version': 4,
        'evaluation': {
            'status': 'evaluated' if eligible else 'insufficient_observations',
            'correlation_window': window,
            'opportunities_per_pair': eligible,
            'association_rates': {
                pair_name(i, j): metrics.edge_counts.get(pair_name(i, j), 0) / eligible if eligible else None
                for i, j in channel_pairs(channels)
            },
        },
        'field_diagnostics': metrics.field.summary() if metrics.field is not None else None,
        'output_path': str(result.output_path),
        'preset': result.preset,
        'experiment': result.experiment,
        'frames': result.frames,
        'agents': result.agents,
        'field_res': result.field_res,
        'seed': result.seed,
        'backend': 'GPU' if result.on_gpu else 'CPU',
        'total_discoveries': metrics.total_discoveries,
        'unique_associations': len(metrics.edge_counts),
        'association_counts': dict(metrics.edge_counts),
        'frames_processed': len(metrics.discovery_rate_history),
        'passport': result.passport,
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
    write_report(summary_path, summary)
    return summary_path


def main(argv: Sequence[str] | None = None) -> RunResult:
    args = parse_args(argv)
    apply_runtime_options(args)
    if args.control_study:
        report = run_control_study(args.control_study, CFG.SEED)
        write_report(args.output, report)
        report_path = render_study_report(report, args.output.with_suffix('.html'))
        print(f'Paired control study complete: {args.control_study} seeds. Report: {args.output}. Readable report: {report_path}')
        return RunResult(output_path=args.output, frames=CFG.FRAMES, agents=CFG.AGENTS,
                         field_res=CFG.FIELD_RES, on_gpu=on_gpu, seed=CFG.SEED,
                         experiment=args.experiment or ('quick' if args.quick else 'balanced'),
                         summary_path=args.output, passport=report['passport'], report_path=report_path)
    target_field, calibration_threshold = _prepare_target_and_threshold(args)
    if target_field is not None:
        CFG.FRAMES = min(CFG.FRAMES, target_field.frame_count)
    effective_preset = 'empirical' if args.input_data is not None and args.preset == 'synthetic' else args.preset
    passport = experiment_passport(args.input_data, synthetic=target_field is None)
    if target_field is not None:
        passport['telemetry_quality'] = target_field.quality_report
    recorder = LongitudinalMetricsRecorder(output_dir=args.output.parent) if effective_preset in {'baseline', 'empirical'} else None

    if args.headless:
        dashboard = LiveDashboard(CFG.FRAMES, enabled=args.live)
        try:
            engine = SimulationEngine(target_field, recorder, calibration_threshold, dashboard).run()
        finally:
            dashboard.close()
        artifacts = SimulationArtifacts(animation=None, recorder=recorder, calibration_threshold=calibration_threshold,
                                        metrics=engine.metrics, engine=engine)
        output_path = args.output
    else:
        artifacts = run_simulation(
            target_field=target_field,
            preset=effective_preset,
            recorder=recorder,
            calibrated_threshold=calibration_threshold,
            live=getattr(args, 'live', False),
            is_notebook=getattr(args, 'show', False),
        )
        output_path = save_animation(artifacts.animation, args.output, fps=args.fps, dashboard=artifacts.dashboard)

    metrics_path: Path | None = None
    metadata_path: Path | None = None
    if artifacts.recorder is not None:
        metadata = {
            'preset': effective_preset,
            'passport': passport,
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
        from IPython.display import HTML, Image, display

        display(Image(filename=str(output_path)) if output_path.suffix.lower() == '.gif'
                else HTML(output_path.read_text(encoding='utf-8')))

    result = RunResult(
        output_path=output_path,
        frames=CFG.FRAMES,
        agents=CFG.AGENTS,
        field_res=CFG.FIELD_RES,
        on_gpu=on_gpu,
        seed=CFG.SEED,
        preset=effective_preset,
        experiment=getattr(args, 'experiment', None) or ('quick' if args.quick else 'balanced'),
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        calibration_threshold=artifacts.calibration_threshold,
        passport=passport,
    )

    # Write the lightweight metrics summary beside the animation. The animation's
    # frame updates ran during save_animation above, so metrics are populated now.
    if args.headless:
        result.summary_path = write_report(output_path, build_run_summary(result, artifacts.metrics))
    elif artifacts.metrics is not None and not getattr(args, 'no_metrics_sidecar', False):
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


def cli(argv: Sequence[str] | None = None) -> None:
    """Console script entrypoint for the ions-x command."""
    main(argv)


if __name__ == '__main__':
    cli()
