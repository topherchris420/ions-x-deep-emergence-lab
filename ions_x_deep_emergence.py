import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

# GPU acceleration with fallback
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


@dataclass
class RunResult:
    output_path: Path
    frames: int
    agents: int
    field_res: int
    on_gpu: bool


def positive_int(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError('value must be a positive integer')
    return value


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description='Run the IONS-X Deep Emergence Lab simulation and save an HTML animation.'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use a smaller, faster configuration for first runs and demos.',
    )
    parser.add_argument('--frames', type=positive_int, help='Number of animation frames to render.')
    parser.add_argument('--agents', type=positive_int, help='Number of autonomous agents to simulate.')
    parser.add_argument('--field-res', type=positive_int, dest='field_res', help='2D field resolution.')
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help='HTML output path for the rendered animation.',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Also display the animation inline when running in an IPython notebook.',
    )
    return parser.parse_args(argv)


def apply_runtime_options(args):
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


def save_animation_html(animation, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(animation.to_jshtml(), encoding='utf-8')
    return output_path


class PerformanceMetrics:
    def __init__(self):
        self.type_counts = defaultdict(int)
        self.env_history = []
        self.discovery_rate_history = []
        self.coherence_frames = []

    def log_discovery(self, agent_type):
        self.type_counts[agent_type] += 1

    def log_frame(self, frame, discoveries, env_factor, is_coherence):
        self.env_history.append(env_factor)
        self.discovery_rate_history.append(discoveries)
        if is_coherence:
            self.coherence_frames.append(frame)


class EnvironmentalModerators:
    def __init__(self):
        self.coherence_events = []
        self.m = 1.0

    def update(self, t):
        self.m = 1.0 + 0.03 * math.sin(t / 31.8) + 0.02 * math.cos(t / 55.7)
        if rng.rand() < 0.02:
            self.coherence_events.append(t)

    def get_modulation(self, t):
        return float(self.m * (1.3 if self.is_coherence_active(t) else 1.0))

    def is_coherence_active(self, t):
        return any(abs(t - ev) < 15 for ev in self.coherence_events)


def evolve_fields(F, t, env_factor, env_mod):
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


@dataclass
class Observation:
    values: Tuple[float, ...]
    env_factor: float


class Agent:
    def __init__(self, aid, atype):
        self.id = aid
        self.type = atype
        self.memory = []
        self.pos = rng.randint(0, CFG.FIELD_RES, size=2)

    def observe(self, obs):
        self.memory.append(obs)
        if len(self.memory) > CFG.MEMORY:
            self.memory.pop(0)

    def discover(self):
        if len(self.memory) < CFG.CORR_WINDOW:
            return []

        data = np.array([o.values for o in self.memory[-CFG.CORR_WINDOW :]])
        discs = []
        for i in range(CFG.CHANNELS):
            for j in range(i + 1, CFG.CHANNELS):
                r = np.corrcoef(data[:, i], data[:, j])[0, 1]
                if abs(r) > CFG.DISCOVER_THRESH:
                    discs.append({'edge': (f'ch{i}', f'ch{j}'), 'confidence': abs(r)})
        return discs


def build_simulation_state():
    import networkx as nx

    agents = [Agent(i, CFG.AGENT_TYPES[i % 3]) for i in range(CFG.AGENTS)]
    graph = nx.DiGraph()
    conf_map = defaultdict(float)
    metrics = PerformanceMetrics()
    env_mod = EnvironmentalModerators()
    F = xp.asarray(
        rng.normal(0, 0.02, (CFG.CHANNELS, CFG.FIELD_RES, CFG.FIELD_RES)),
        dtype=xp.float32,
    )
    return agents, graph, conf_map, metrics, env_mod, F


def run_simulation():
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

    agents, graph, conf_map, metrics, env_mod, F = build_simulation_state()

    def update(frame):
        nonlocal F

        env_mod.update(frame)
        modulation = env_mod.get_modulation(frame)
        F = evolve_fields(F, frame, modulation, env_mod)
        F_cpu = cp.asnumpy(F) if on_gpu else F

        count = 0
        for agent in agents:
            agent.pos = (agent.pos + rng.randint(-3, 4, 2)) % CFG.FIELD_RES
            agent.observe(Observation(tuple(F_cpu[:, agent.pos[0], agent.pos[1]]), modulation))
            for discovery in agent.discover():
                u, v = discovery['edge']
                key = f'{u}->{v}'
                conf_map[key] = max(conf_map[key], discovery['confidence'])
                graph.add_edge(u, v, weight=conf_map[key])
                metrics.log_discovery(agent.type)
                count += 1

        for key in list(conf_map.keys()):
            conf_map[key] *= CFG.CONFIDENCE_DECAY
            if conf_map[key] < 0.05:
                u, v = key.split('->')
                graph.remove_edge(u, v)
                del conf_map[key]

        metrics.log_frame(frame, count, modulation, env_mod.is_coherence_active(frame))
        ax_field.clear()
        ax_field.imshow(F_cpu[0], cmap='magma')
        ax_field.axis('off')

        ax_graph.clear()
        nx.draw(graph, ax=ax_graph, with_labels=True, node_color='orange', edge_color='cyan')

        ax_stats.clear()
        ax_stats.text(
            0.1,
            0.5,
            f"Discoveries: {sum(metrics.type_counts.values())}\nM-Factor: {modulation:.3f}",
            fontsize=12,
        )

    animation = FuncAnimation(fig, update, frames=CFG.FRAMES, interval=50, repeat=False)
    return animation


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    apply_runtime_options(args)
    animation = run_simulation()
    output_path = save_animation_html(animation, args.output)

    if args.show:
        from IPython.display import HTML, display

        display(HTML(output_path.read_text(encoding='utf-8')))

    result = RunResult(
        output_path=output_path,
        frames=CFG.FRAMES,
        agents=CFG.AGENTS,
        field_res=CFG.FIELD_RES,
        on_gpu=on_gpu,
    )
    print(
        'Simulation complete. '
        f'Frames: {result.frames}. Agents: {result.agents}. '
        f'Field: {result.field_res}x{result.field_res}. '
        f'Backend: {"GPU" if result.on_gpu else "CPU"}. '
        f'Output: {result.output_path}'
    )
    return result


if __name__ == '__main__':
    main()
