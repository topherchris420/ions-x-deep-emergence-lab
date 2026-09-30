"""Evidence accounting and portable study report regressions."""

import json

import pytest

import ions_x_deep_emergence as lab
from study_report import paired_association_summary, render_study_report


def test_paired_effects_preserve_negative_zero_and_positive_results():
    runs = [
        {'coupled': {'edge_counts': {'ch0--ch1': c}}, 'uncoupled': {'edge_counts': {'ch0--ch1': u}}}
        for c, u in [(8, 2), (3, 3), (1, 5)]
    ]
    effect = paired_association_summary(runs, 10)['ch0--ch1']
    assert effect['per_seed_rate_differences'] == [0.6, 0, -0.4]
    assert effect['mean_rate_difference'] == pytest.approx(1 / 15)
    assert effect['rate_difference_range'] == [-0.4, 0.6]
    assert [effect[key] for key in ('positive_seeds', 'zero_seeds', 'negative_seeds')] == [1, 1, 1]


@pytest.mark.parametrize('frames,expected', [(49, 0), (50, 2), (51, 4)])
def test_evaluation_window_boundary(tmp_path, frames, expected):
    result = lab.main(['--headless', '--frames', str(frames), '--agents', '2', '--field-res', '4',
                       '--output', str(tmp_path / 'run.json')])
    evaluation = json.loads(result.summary_path.read_text())['evaluation']
    assert evaluation['opportunities_per_pair'] == expected
    assert evaluation['status'] == ('evaluated' if expected else 'insufficient_observations')
    assert len(evaluation['association_rates']) == 6
    if not expected:
        assert all(value is None for value in evaluation['association_rates'].values())
    else:
        assert all(0 <= value <= 1 for value in evaluation['association_rates'].values())


def test_study_cli_writes_html_and_consistent_seed_effects(tmp_path):
    result = lab.main(['--control-study', '2', '--frames', '50', '--agents', '2', '--field-res', '4',
                       '--output', str(tmp_path / 'study.json')])
    report = json.loads(result.summary_path.read_text())
    assert result.report_path == tmp_path / 'study.html'
    assert report['paired_association_effects'] == paired_association_summary(report['runs'], 2)
    html = result.report_path.read_text()
    assert 'Every seed, every pair' in html
    assert 'ch0--ch1' in html
    assert '<script' not in html
    # Input strings can never become executable markup in the portable report.
    report['limitations'].append('<script>alert("bad")</script>')
    render_study_report(report, result.report_path)
    assert '<script>' not in result.report_path.read_text()
    assert '&lt;script&gt;' in result.report_path.read_text()
