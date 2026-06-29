import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'ions_x_deep_emergence.py'
MODULE_SPEC = importlib.util.spec_from_file_location('ions_x_deep_emergence_cli', MODULE_PATH)


def load_module():
    module = importlib.util.module_from_spec(MODULE_SPEC)
    MODULE_SPEC.loader.exec_module(module)
    return module


def snapshot_cfg(sim):
    return {
        'FIELD_RES': sim.CFG.FIELD_RES,
        'AGENTS': sim.CFG.AGENTS,
        'FRAMES': sim.CFG.FRAMES,
        'SAMPLE_PER_FRAME': sim.CFG.SAMPLE_PER_FRAME,
    }


def restore_cfg(sim, values):
    for key, value in values.items():
        setattr(sim.CFG, key, value)


def test_parse_args_accepts_quick_mode_and_output_path():
    sim = load_module()

    args = sim.parse_args([
        '--quick',
        '--frames', '12',
        '--agents', '7',
        '--field-res', '32',
        '--output', 'outputs/demo.html',
    ])

    assert args.quick is True
    assert args.frames == 12
    assert args.agents == 7
    assert args.field_res == 32
    assert args.output == Path('outputs/demo.html')


def test_quick_mode_and_overrides_update_runtime_config():
    sim = load_module()
    original = snapshot_cfg(sim)
    try:
        args = sim.parse_args(['--quick', '--frames', '12', '--agents', '7', '--field-res', '32'])

        sim.apply_runtime_options(args)

        assert sim.CFG.FRAMES == 12
        assert sim.CFG.AGENTS == 7
        assert sim.CFG.FIELD_RES == 32
        assert sim.CFG.SAMPLE_PER_FRAME <= original['SAMPLE_PER_FRAME']
    finally:
        restore_cfg(sim, original)


def test_main_writes_html_output_without_running_full_simulation(tmp_path):
    sim = load_module()
    original = snapshot_cfg(sim)
    output_path = tmp_path / 'demo.html'
    calls = []

    class FakeAnimation:
        def to_jshtml(self):
            return '<section>IONS-X demo animation</section>'

    def fake_run_simulation():
        calls.append((sim.CFG.FRAMES, sim.CFG.AGENTS, sim.CFG.FIELD_RES))
        return FakeAnimation()

    try:
        sim.run_simulation = fake_run_simulation

        result = sim.main([
            '--quick',
            '--frames', '2',
            '--agents', '3',
            '--field-res', '16',
            '--output', str(output_path),
        ])

        assert calls == [(2, 3, 16)]
        assert output_path.read_text(encoding='utf-8') == '<section>IONS-X demo animation</section>'
        assert result.output_path == output_path
        assert result.frames == 2
        assert result.agents == 3
        assert result.field_res == 16
    finally:
        restore_cfg(sim, original)