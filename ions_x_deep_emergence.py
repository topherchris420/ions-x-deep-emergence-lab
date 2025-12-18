import matplotlib.pyplot as plt
plt.rcParams['animation.embed_limit'] = 200
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from IPython.display import HTML, display
import numpy as np
import math
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict
from scipy import stats
import json

# GPU acceleration with fallback
try:
    import cupy as cp
    xp = cp
    fft_rfft = cp.fft.rfft2
    ifft_irfft = cp.fft.irfft2
    on_gpu = True
except ImportError:
    xp = np
    fft_rfft = np.fft.rfft2
    ifft_irfft = np.fft.irfft2
    on_gpu = False

# ============================================================================
# CONFIGURATION - TUNED FOR SENSITIVITY
# ============================================================================
class CFG:
    FIELD_RES = 128
    CHANNELS = 4
    AGENTS = 300
    MEMORY = 300
    AGENT_TYPES = ['perceiver', 'forecaster', 'integrator']
    CORR_WINDOW = 50       # REDUCED: Allows discovery to start at frame 50
    DISCOVER_THRESH = 0.32 # INCREASED SENSITIVITY
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

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================
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
        if is_coherence: self.coherence_frames.append(frame)

metrics = PerformanceMetrics()

# ============================================================================
# FIELD SYSTEM & MODERATORS
# ============================================================================
class EnvironmentalModerators:
    def __init__(self): self.coherence_events = []
    def update(self, t):
        self.m = 1.0 + 0.03*math.sin(t/31.8) + 0.02*math.cos(t/55.7)
        if rng.rand() < 0.02: self.coherence_events.append(t)
    def get_modulation(self, t): return float(self.m * (1.3 if self.is_coherence_active(t) else 1.0))
    def is_coherence_active(self, t): return any(abs(t - ev) < 15 for ev in self.coherence_events)

env_mod = EnvironmentalModerators()

def evolve_fields(F, t, env_factor, env_mod):
    Fk = fft_rfft(F)
    for ci in range(F.shape[0]):
        Fk[ci] *= xp.exp(-xp.asarray([0.002, 0.0015, 0.0018, 0.0012])[ci] * (xp.fft.fftfreq(CFG.FIELD_RES).reshape(CFG.FIELD_RES, 1)**2 + xp.fft.rfftfreq(CFG.FIELD_RES).reshape(1, -1)**2) * 0.5 * env_factor)
    F = ifft_irfft(Fk, s=(CFG.FIELD_RES, CFG.FIELD_RES))
    F[0] += 0.045 * F[1] * env_factor # STRONGER COUPLING
    if env_mod.is_coherence_active(t): F[3] = 0.7 * F[3] + 0.3 * F[2]
    return 0.92 * F + 0.08 * xp.tanh(F * 4.0)

# ============================================================================
# OPERATORS & GRAPH
# ============================================================================
@dataclass
class Observation:
    values: Tuple[float, ...]; env_factor: float

class Agent:
    def __init__(self, aid, atype):
        self.id = aid; self.type = atype; self.memory = []; self.pos = rng.randint(0, CFG.FIELD_RES, size=2)
    def observe(self, obs):
        self.memory.append(obs)
        if len(self.memory) > CFG.MEMORY: self.memory.pop(0)
    def discover(self):
        if len(self.memory) < CFG.CORR_WINDOW: return []
        data = np.array([o.values for o in self.memory[-CFG.CORR_WINDOW:]])
        discs = []
        for i in range(CFG.CHANNELS):
            for j in range(i+1, CFG.CHANNELS):
                r = np.corrcoef(data[:,i], data[:,j])[0,1]
                if abs(r) > CFG.DISCOVER_THRESH: discs.append({'edge': (f"ch{i}", f"ch{j}"), 'confidence': abs(r)})
        return discs

agents = [Agent(i, CFG.AGENT_TYPES[i % 3]) for i in range(CFG.AGENTS)]
graph = nx.DiGraph(); conf_map = defaultdict(float)

# ============================================================================
# RUN SIMULATION
# ============================================================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(2, 3)
ax_graph = fig.add_subplot(gs[:, 1:])
ax_field = fig.add_subplot(gs[0, 0])
ax_stats = fig.add_subplot(gs[1, 0])
F = xp.asarray(rng.normal(0, 0.02, (CFG.CHANNELS, CFG.FIELD_RES, CFG.FIELD_RES)), dtype=xp.float32)

def update(frame):
    global F
    env_mod.update(frame); m = env_mod.get_modulation(frame)
    F = evolve_fields(F, frame, m, env_mod); F_cpu = (cp.asnumpy(F) if on_gpu else F)
    count = 0
    for a in agents:
        a.pos = (a.pos + rng.randint(-3, 4, 2)) % CFG.FIELD_RES
        a.observe(Observation(tuple(F_cpu[:, a.pos[0], a.pos[1]]), m))
        for d in a.discover():
            u, v = d['edge']; key = f"{u}->{v}"
            conf_map[key] = max(conf_map[key], d['confidence']); graph.add_edge(u, v, weight=conf_map[key])
            metrics.log_discovery(a.type); count += 1
    for k in list(conf_map.keys()):
        conf_map[k] *= CFG.CONFIDENCE_DECAY
        if conf_map[k] < 0.05:
            u, v = k.split('->'); graph.remove_edge(u, v); del conf_map[k]
    metrics.log_frame(frame, count, m, env_mod.is_coherence_active(frame))
    ax_field.clear(); ax_field.imshow(F_cpu[0], cmap='magma'); ax_field.axis('off')
    ax_graph.clear(); nx.draw(graph, ax=ax_graph, with_labels=True, node_color='orange', edge_color='cyan')
    ax_stats.clear(); ax_stats.text(0.1, 0.5, f"Discoveries: {sum(metrics.type_counts.values())}\nM-Factor: {m:.3f}", fontsize=12)


anim = FuncAnimation(fig, update, frames=CFG.FRAMES, interval=50, repeat=False)
display(HTML(anim.to_jshtml()))
