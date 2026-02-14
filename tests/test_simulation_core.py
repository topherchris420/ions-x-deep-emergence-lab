import importlib.util
from pathlib import Path


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
