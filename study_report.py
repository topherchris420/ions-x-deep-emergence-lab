"""Dependency-free statistics and offline presentation of paired experiment evidence.

Seeds are the independent units of a paired study. Agents and overlapping windows observing
one field are not, so nothing here turns their counts into a sample size.
"""

import random
from collections.abc import Sequence
from html import escape
from math import fsum, sqrt
from pathlib import Path
from typing import Any

HISTOGRAM_BINS = 100  # |r| bins of width 0.01, so every two-decimal threshold is a bin edge
EXACT_SIGN_FLIP_LIMIT = 18  # enumerate all 2^n sign assignments up to this many seeds
MONTE_CARLO_DRAWS = 20_000
ARMS = ('coupled', 'uncoupled')


def tail_count(histogram: Sequence[int], start_bin: int) -> int:
    """Windows with |r| >= start_bin / HISTOGRAM_BINS."""
    return int(sum(histogram[max(start_bin, 0):]))


def threshold_bin_for_rate(histogram: Sequence[int], rate: float | None) -> int | None:
    """Lowest bin edge whose tail holds at most ``rate`` of all windows."""
    total = sum(histogram)
    if not total or rate is None:
        return None
    tail = 0
    for k in range(len(histogram) - 1, -1, -1):
        if (tail + histogram[k]) / total > rate:
            return k + 1
        tail += histogram[k]
    return 0


def pooled_histogram(runs: Sequence[dict[str, Any]], arm: str, pairs: Sequence[str]) -> list[int]:
    """Sum the |r| histograms of ``arm`` over runs and pairs."""
    pooled = [0] * HISTOGRAM_BINS
    for run in runs:
        for pair in pairs:
            for k, count in enumerate(run[arm]['abs_r_histogram'][pair]):
                pooled[k] += count
    return pooled


def auc_from_histograms(positive: Sequence[int], negative: Sequence[int]) -> float | None:
    """P(a positive-arm |r| exceeds a negative-arm |r|); ties within one bin count one half."""
    positive_total, negative_total = sum(positive), sum(negative)
    if not positive_total or not negative_total:
        return None
    below, wins = 0, 0.0
    for p, n in zip(positive, negative, strict=True):
        wins += p * (below + 0.5 * n)
        below += n
    return wins / (positive_total * negative_total)


def sign_flip_p_value(values: Sequence[float], seed: int = 0) -> tuple[float, str]:
    """Two-sided p-value for 'the paired differences are symmetric about zero'.

    Each seed contributes one paired difference. Under the null hypothesis its sign is as likely
    negative as positive, so every sign assignment is equally likely. Up to EXACT_SIGN_FLIP_LIMIT
    seeds all assignments are enumerated; beyond that a seeded Monte Carlo estimate that counts
    the observed assignment is returned.
    """
    values = [float(v) for v in values]
    observed = abs(fsum(values))
    tolerance = 1e-9 * max(1.0, sum(abs(v) for v in values))
    if len(values) <= EXACT_SIGN_FLIP_LIMIT:
        sums = [0.0]
        for value in values:
            sums = [s + value for s in sums] + [s - value for s in sums]
        return sum(abs(s) >= observed - tolerance for s in sums) / len(sums), 'exact'
    draws = random.Random(seed)
    hits = sum(
        abs(sum(v if draws.random() < 0.5 else -v for v in values)) >= observed - tolerance
        for _ in range(MONTE_CARLO_DRAWS)
    )
    return (hits + 1) / (MONTE_CARLO_DRAWS + 1), 'monte_carlo'


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjustment: family-wise error control across all tested pairs."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def paired_association_summary(runs: list[dict[str, Any]], opportunities: int) -> dict[str, Any]:
    """Seed-level effects per pair, with uncertainty taken from seeds and never from windows."""
    if not runs or opportunities <= 0:
        raise ValueError('Paired summaries require runs and positive detection opportunities.')
    pairs = sorted(runs[0]['coupled']['edge_counts'])
    result = {}
    for pair in pairs:
        differences = [
            (run['coupled']['edge_counts'][pair] - run['uncoupled']['edge_counts'][pair]) / opportunities
            for run in runs
        ]
        n = len(differences)
        mean = fsum(differences) / n  # correctly rounded on every Python version, unlike sum()
        spread = sqrt(fsum((d - mean) ** 2 for d in differences) / (n - 1)) if n > 1 else None
        p_value, method = sign_flip_p_value(differences)
        result[pair] = {
            'mean_rate_difference': mean,
            'rate_difference_range': [min(differences), max(differences)],
            'standard_error': spread / sqrt(n) if spread is not None else None,
            'sign_flip_p': p_value,
            'sign_flip_method': method,
            'positive_seeds': sum(value > 0 for value in differences),
            'zero_seeds': sum(value == 0 for value in differences),
            'negative_seeds': sum(value < 0 for value in differences),
            'per_seed_rate_differences': differences,
        }
    for pair, adjusted in zip(pairs, holm_adjust([result[pair]['sign_flip_p'] for pair in pairs]), strict=True):
        result[pair]['holm_p'] = adjusted
    return result


def discrimination_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Threshold-free separation of the two arms for each pair (area under the ROC curve)."""
    if not runs or 'abs_r_histogram' not in runs[0]['coupled']:
        return {}
    result = {}
    for pair in sorted(runs[0]['coupled']['abs_r_histogram']):
        per_seed = [auc_from_histograms(run['coupled']['abs_r_histogram'][pair], run['uncoupled']['abs_r_histogram'][pair])
                    for run in runs]
        observed = [value for value in per_seed if value is not None]
        result[pair] = {
            'auc': auc_from_histograms(pooled_histogram(runs, 'coupled', [pair]), pooled_histogram(runs, 'uncoupled', [pair])),
            'per_seed_auc': per_seed,
            'auc_range': [min(observed), max(observed)] if observed else None,
        }
    return result


# --------------------------------------------------------------------------- presentation


def _pct(value: float | None, digits: int = 2) -> str:
    return '—' if value is None else f'{100 * value:.{digits}f}%'


def _p(value: float | None) -> str:
    if value is None:
        return '—'
    return '&lt;0.001' if value < 0.001 else f'{value:.3f}'


def _num(value: float | None, digits: int = 2) -> str:
    return '—' if value is None else f'{value:+.{digits}f}'


def _nice_step(span: float, target: int = 6) -> float:
    for step in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100):
        if span / step <= target:
            return step
    return 100.0


def _effects_plot(effects: dict[str, Any], labels: dict[str, str], seeds: Sequence[int]) -> str:
    """Every seed's coupled-minus-uncoupled rate difference, one row per pair, in percentage points."""
    pairs = list(effects)
    values = [100 * v for pair in pairs for v in effects[pair]['per_seed_rate_differences']]
    low, high = min([0.0, *values]), max([0.0, *values])
    span = max(high - low, 1.0)
    low, high = low - 0.06 * span, high + 0.06 * span
    left, right, row, top = 170, 748, 40, 34
    height = top + row * len(pairs) + 12

    def x(value: float) -> float:
        return left + (value - low) / (high - low) * (right - left)

    step = _nice_step(high - low)
    ticks = [k * step for k in range(int(low // step) + 1, int(high // step) + 1)]
    parts = [f'<svg class="figure" viewBox="0 0 760 {height}" role="img" aria-labelledby="effects-title">',
             '<title id="effects-title">Per-seed detection-rate differences for every channel pair</title>']
    for tick in ticks:
        parts.append(f'<line class="grid" x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top - 8}" y2="{height - 10}"/>'
                     f'<text class="axis" x="{x(tick):.1f}" y="{top - 14}" text-anchor="middle">{tick:+g}</text>')
    parts.append(f'<line class="zero" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top - 8}" y2="{height - 10}"/>'
                 f'<text class="axis" x="0" y="{top - 14}">coupled − uncoupled, pp</text>')
    for index, pair in enumerate(pairs):
        center = top + row * index + row / 2
        label = labels.get(pair, 'no injected coupling')
        parts.append(f'<text class="label" x="0" y="{center - 2:.1f}">{escape(pair)}</text>'
                     f'<text class="sublabel" x="0" y="{center + 13:.1f}">{escape(label)}</text>')
        for k, value in enumerate(effects[pair]['per_seed_rate_differences']):
            seed = seeds[k] if k < len(seeds) else k
            parts.append(f'<circle class="dot" cx="{x(100 * value):.1f}" cy="{center + ((k % 5) - 2) * 3:.1f}" r="4.5">'
                         f'<title>seed {seed}: {100 * value:+.2f} pp</title></circle>')
        mean = 100 * effects[pair]['mean_rate_difference']
        parts.append(f'<line class="mean" x1="{x(mean):.1f}" x2="{x(mean):.1f}" y1="{center - 13:.1f}" y2="{center + 13:.1f}">'
                     f'<title>mean {mean:+.2f} pp</title></line>')
    parts.append('</svg>')
    return ''.join(parts)


def _roc_panel(pair: str, label: str, positive: Sequence[int], negative: Sequence[int], threshold: float,
               auc: float | None) -> str:
    size, pad = 220, 34
    plot = size - pad - 12
    pos_total, neg_total = sum(positive) or 1, sum(negative) or 1
    points = [(tail_count(negative, k) / neg_total, tail_count(positive, k) / pos_total)
              for k in range(HISTOGRAM_BINS, -1, -1)]

    def xy(fpr: float, tpr: float) -> str:
        return f'{pad + fpr * plot:.1f},{12 + (1 - tpr) * plot:.1f}'

    operating_bin = round(threshold * HISTOGRAM_BINS)
    fpr, tpr = tail_count(negative, operating_bin) / neg_total, tail_count(positive, operating_bin) / pos_total
    auc_text = '—' if auc is None else f'{auc:.3f}'
    return (f'<figure class="roc"><svg viewBox="0 0 {size} {size}" role="img" aria-label="ROC curve for {escape(pair)}">'
            f'<rect class="frame" x="{pad}" y="12" width="{plot}" height="{plot}"/>'
            f'<line class="chance" x1="{pad}" y1="{12 + plot}" x2="{pad + plot}" y2="12"/>'
            f'<polyline class="curve" points="{" ".join(xy(f, t) for f, t in points)}"/>'
            f'<circle class="operating" cx="{pad + fpr * plot:.1f}" cy="{12 + (1 - tpr) * plot:.1f}" r="5">'
            f'<title>|r| &gt; {threshold:g}: coupled {100 * tpr:.1f}%, uncoupled {100 * fpr:.1f}%</title></circle>'
            f'<text class="axis" x="{pad + plot / 2}" y="{size - 4}" text-anchor="middle">uncoupled detection rate</text>'
            f'<text class="axis" transform="translate(11 {12 + plot / 2}) rotate(-90)" text-anchor="middle">coupled rate</text>'
            f'<text class="auc" x="{pad + plot - 6}" y="{12 + plot - 8}" text-anchor="end">AUC {auc_text}</text>'
            f'</svg><figcaption>{escape(pair)}<small>{escape(label)}</small></figcaption></figure>')


def _calibration_section(calibration: dict[str, Any], labels: dict[str, str]) -> str:
    inflation = calibration.get('false_alarm_inflation')
    n_eff = calibration.get('effective_samples_per_window')
    window = calibration['correlation_window']
    lines = [
        '<h2>Is the detector calibrated?</h2>',
        f'<p>With {window} independent samples, two unrelated channels exceed |r| &gt; {calibration["threshold"]:g} '
        f'in <strong>{_pct(calibration.get("nominal_false_alarm_rate"))}</strong> of windows. That is the textbook '
        f'false-alarm rate. Every pair in the uncoupled arm is unrelated by construction, and there the detector fired in '
        f'<strong>{_pct(calibration.get("observed_false_alarm_rate"))}</strong> of windows'
        + (f', <strong>{inflation:.1f}×</strong> the nominal rate.' if inflation else '.') + '</p>',
    ]
    if n_eff:
        lines.append(f'<p class="note">An agent drifting through a smooth, slowly changing field takes samples that repeat '
                     f'one another. The detector behaves as if each {window}-sample window held about '
                     f'<strong>{n_eff:.0f} independent samples</strong>. Its nominal significance should not be read literally.</p>')
    held_out = calibration.get('held_out')
    if held_out:
        seeds_a = ', '.join(str(s) for s in held_out['calibration_seeds'])
        seeds_b = ', '.join(str(s) for s in held_out['evaluation_seeds'])
        rows = ''.join(
            f'<tr><th scope="row">{escape(pair)}<small>{escape(labels.get(pair, ""))}</small></th>'
            f'<td>{_pct(held_out["sensitivity_at_default"].get(pair))}</td><td>{_pct(held_out["sensitivity"].get(pair))}</td></tr>'
            for pair in held_out['sensitivity']
        )
        threshold = held_out.get('threshold_for_nominal_rate')
        threshold_text = '—' if threshold is None else f'{threshold:.2f}'
        lines.append(
            f'<h3>Held-out recalibration</h3><p>The uncoupled arms of seeds {escape(seeds_a)} set the threshold that restores '
            f'the nominal rate: |r| ≥ <strong>{threshold_text}</strong>. Seeds {escape(seeds_b)} were not used to choose it. '
            f'On those seeds that threshold gives a false-alarm rate of <strong>{_pct(held_out.get("false_alarm_rate"))}</strong>, '
            'and detects the injected couplings as follows:</p>'
            '<div class="scroll"><table><thead><tr><th scope="col">Injected pair</th>'
            f'<th scope="col">Sensitivity at |r| &gt; {calibration["threshold"]:g}</th>'
            f'<th scope="col">Sensitivity at |r| ≥ {threshold_text}</th></tr></thead><tbody>{rows}</tbody></table></div>')
    by_pair = calibration.get('false_alarm_rate_by_pair') or {}
    if by_pair:
        cells = ''.join(f'<tr><th scope="row">{escape(pair)}</th><td>{_pct(rate)}</td></tr>' for pair, rate in by_pair.items())
        lines.append('<details><summary>Uncoupled-arm false-alarm rate by pair</summary><div class="scroll"><table>'
                     f'<tbody>{cells}</tbody></table></div></details>')
    return ''.join(lines)


def _field_section(truth: dict[str, Any], labels: dict[str, str]) -> str:
    rows = ''.join(
        f'<tr><th scope="row">{escape(pair)}<small>{escape(labels.get(pair, "no injected coupling"))}</small></th>'
        f'<td>{_num(values.get("coupled"))}</td><td>{_num(values.get("uncoupled"))}</td></tr>'
        for pair, values in truth['pair_correlation_mean'].items()
    )
    memory, neighbour = truth.get('window_memory', {}), truth.get('neighbour_correlation', {})
    return (
        '<h2>What the field actually contained</h2><p>Agents see a few samples. These values use every pixel: the '
        'mean cross-channel correlation of the whole field, the field\'s correlation with itself one detection window '
        'earlier, and the correlation between neighbouring pixels. A healthy generator changes within a window '
        '(memory near 0) and has spatial structure (neighbour correlation well above 0).</p>'
        '<div class="scroll"><table><thead><tr><th scope="col">Field quantity</th><th scope="col">Coupled</th>'
        f'<th scope="col">Uncoupled</th></tr></thead><tbody>{rows}'
        f'<tr><th scope="row">Memory across one window<small>1 = frozen field</small></th>'
        f'<td>{_num(memory.get("coupled"))}</td><td>{_num(memory.get("uncoupled"))}</td></tr>'
        f'<tr><th scope="row">Neighbouring-pixel correlation<small>0 = no spatial structure</small></th>'
        f'<td>{_num(neighbour.get("coupled"))}</td><td>{_num(neighbour.get("uncoupled"))}</td></tr>'
        '</tbody></table></div>'
    )


def render_study_report(report: dict[str, Any], output_path: Path) -> Path:
    """Render a portable HTML companion; no network, scripts, or external assets."""
    opportunities = report['opportunities_per_pair_per_run']
    effects = paired_association_summary(report['runs'], opportunities)
    labels = report['injected_associations']
    seeds = [run['seed'] for run in report['runs']]
    has_statistics = len(report['runs']) > 1
    rows = []
    for pair, effect in effects.items():
        rates = report['association_rates'][pair]
        low, high = effect['rate_difference_range']
        statistics = (f'<td>{_pct(effect["standard_error"])}</td><td>{_p(effect["sign_flip_p"])}</td>'
                      f'<td>{_p(effect["holm_p"])}</td>') if has_statistics else ''
        rows.append(
            f'<tr><th scope="row">{escape(pair)}<small>{escape(labels.get(pair, "no direct injected coupling"))}</small></th>'
            f'<td>{rates["coupled"]:.2%}</td><td>{rates["uncoupled"]:.2%}</td>'
            f'<td>{100 * effect["mean_rate_difference"]:+.2f} pp</td>'
            f'<td>{100 * low:+.2f} to {100 * high:+.2f} pp</td>'
            f'<td>{effect["positive_seeds"]} / {effect["zero_seeds"]} / {effect["negative_seeds"]}</td>{statistics}</tr>'
        )
    statistics_heading = ('<th scope="col">SE</th><th scope="col">Sign-flip p</th><th scope="col">Holm p</th>'
                          if has_statistics else '')
    pairs = list(effects)
    details = []
    for index, run in enumerate(report['runs']):
        cells = ''.join(f'<td>{100 * effects[pair]["per_seed_rate_differences"][index]:+.2f}</td>' for pair in pairs)
        details.append(f'<tr><th scope="row">{run["seed"]}</th>{cells}</tr>')
    passport = report['passport']
    config = passport['configuration']
    dynamics = report.get('dynamics', config.get('DYNAMICS', 'legacy'))
    threshold = float(config.get('DISCOVER_THRESH', 0.32))

    calibration = report.get('detector_calibration')
    calibration_html = _calibration_section(calibration, labels) if calibration else ''
    discrimination = report.get('discrimination') or {}
    roc_html = ''
    if discrimination and 'abs_r_histogram' in report['runs'][0]['coupled']:
        panels = ''.join(
            _roc_panel(pair, labels.get(pair, 'no injected coupling'), _pooled(report, 'coupled', pair),
                       _pooled(report, 'uncoupled', pair), threshold, discrimination[pair]['auc'])
            for pair in pairs if pair in discrimination
        )
        roc_html = ('<h2>Threshold-free discrimination</h2><p>Each curve sweeps the threshold from 1 to 0, plotting '
                    'how often the coupled arm crosses it against how often the uncoupled arm does. The dot is the configured '
                    'threshold. AUC is the chance that a coupled window shows a stronger |r| than an uncoupled one: '
                    '0.5 means no separation.</p>'
                    f'<div class="rocs">{panels}</div>')
    truth = report.get('field_ground_truth')
    field_html = _field_section(truth, labels) if truth else ''
    nominal = (calibration or {}).get('nominal_false_alarm_rate')
    observed = (calibration or {}).get('observed_false_alarm_rate')
    calibration_stat = (f'<div class="stat"><strong>{_pct(nominal, 1)} → {_pct(observed, 1)}</strong>'
                        'nominal → observed false alarms</div>') if calibration else ''

    settings = ''.join(f'<tr><th scope="row">{escape(str(k))}</th><td>{escape(str(v))}</td></tr>' for k, v in config.items())
    limitations = ''.join(f'<li>{escape(item)}</li>' for item in report['limitations'])
    headings = ''.join(f'<th scope="col">{escape(pair)}</th>' for pair in pairs)
    source = escape(passport['source_sha256'])
    environment = escape(str(passport['dependencies']))
    version = escape(str(passport.get('version', 'unrecorded')))
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IONS-X · Paired experiment report</title><style>
:root{{color-scheme:dark;--bg:#081b20;--panel:#123039;--text:#eaf4f3;--muted:#adc5c5;--accent:#89e1d6;
--line:#315059;--coupled:#39c5bb;--mean:#f2c14e}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);
color:var(--text);font:16px/1.65 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:48px clamp(16px,4vw,24px)}}
h1{{font-size:clamp(2rem,5vw,3.7rem);line-height:1.12;max-width:850px}}h2{{margin-top:48px}}
.eyebrow{{color:#79d4ca;letter-spacing:.14em;text-transform:uppercase;font-size:.8rem}}
.lead{{max-width:800px;color:#bfd4d4}}.stats{{display:flex;flex-wrap:wrap;gap:16px;margin:30px 0}}
.stat{{background:var(--panel);padding:18px 24px;border-radius:12px;flex:1;min-width:180px}}
.stat strong{{display:block;font-size:1.8rem;color:var(--accent)}}.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:13px;
border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}
thead th{{color:var(--accent);font-size:.85rem}}small{{display:block;color:var(--muted);font-weight:400}}
.note{{border-left:3px solid #79d4ca;padding:12px 20px;background:#102b32}}code{{overflow-wrap:anywhere}}
summary{{cursor:pointer;color:var(--accent)}}footer{{margin-top:40px;color:var(--muted);font-size:.85rem}}
svg text{{fill:var(--muted);font:12px system-ui,sans-serif}}svg .label{{fill:var(--text);font-weight:600;font-size:13px}}
.figure{{width:100%;min-width:620px;max-width:900px;height:auto;margin:12px 0}}.grid{{stroke:var(--line)}}
.zero{{stroke:var(--accent);stroke-width:1.5}}.dot{{fill:var(--coupled);fill-opacity:.75}}
.mean{{stroke:var(--mean);stroke-width:3}}.rocs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:18px}}
.roc{{margin:0;background:var(--panel);border-radius:12px;padding:10px}}.roc svg{{width:100%;height:auto}}
.roc figcaption{{font-weight:600;padding:0 8px 6px}}.roc svg text{{font-size:10px}}.frame{{fill:none;stroke:var(--line)}}
.chance{{stroke:var(--muted);stroke-dasharray:4 4}}.curve{{fill:none;stroke:var(--coupled);stroke-width:2.5}}
.operating{{fill:var(--mean)}}svg .auc{{fill:var(--text);font-weight:600}}
@media print{{:root{{color-scheme:light;--bg:#fff;--panel:#eee;--text:#000;--muted:#333;--accent:#07554e;
--line:#bbb;--coupled:#0b7a71;--mean:#a66b00}}.lead{{color:#333}}.note{{background:#eee}}.eyebrow{{color:#07554e}}}}
</style></head><body><main>
<p class="eyebrow">Vers3Dynamics / IONS-X Deep Emergence Lab · {escape(str(dynamics))} field</p>
<h1>What changes when the coupling disappears?</h1>
<p class="lead">A paired synthetic experiment. Each seed replays the same initial field, forcing noise, moving agents,
and moderator schedule. The uncoupled arm removes the two explicit cross-channel coupling terms and nothing else.</p>
<div class="stats"><div class="stat"><strong>{report['repetitions']}</strong>paired seeds</div>
<div class="stat"><strong>{config['FRAMES']}</strong>frames per arm</div>
<div class="stat"><strong>{opportunities:,}</strong>eligible agent-windows per pair / arm</div>{calibration_stat}</div>
<h2>Association response</h2><p>Mean rates across seeds. Differences are coupled minus uncoupled in percentage points (pp).
The range is the observed seed variation, not a confidence interval. Uncertainty comes from seeds alone: SE is the
standard error of the mean seed difference, and the sign-flip test is exact over seeds, Holm-adjusted across all six pairs.</p>
<div class="scroll"><table><thead><tr><th scope="col">Pair / injected relationship</th>
<th scope="col">Coupled</th><th scope="col">Uncoupled</th><th scope="col">Difference</th>
<th scope="col">Seed range</th><th scope="col">Seeds + / = / −</th>{statistics_heading}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<div class="scroll">{_effects_plot(effects, labels, seeds)}</div>
<p class="note">A higher detection rate measures response to this synthetic intervention.
It does not establish causality in empirical data. Pairs without direct coupling may still change
indirectly. Agents and overlapping windows are not independent experimental replicates.</p>
{calibration_html}{roc_html}{field_html}
<h2>Every seed, every pair</h2><p>Rate differences in percentage points. Negative and zero results are retained.</p>
<div class="scroll"><table><thead><tr><th scope="col">Seed</th>{headings}</tr></thead>
<tbody>{''.join(details)}</tbody></table></div>
<h2>Interpretation boundaries</h2><ul>{limitations}</ul>
<details><summary>Effective configuration and provenance</summary><div class="scroll"><table>
<tbody>{settings}</tbody></table></div><p>Lab version: <code>{version}</code>. Engine SHA-256: <code>{source}</code></p>
<p>Dependencies: <code>{environment}</code></p></details>
<footer>Generated from the adjacent JSON report. No external resources or tracking. The JSON is the
machine-readable record; this page is its descriptive companion.</footer></main></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    return output_path


def _pooled(report: dict[str, Any], arm: str, pair: str) -> list[int]:
    return pooled_histogram(report['runs'], arm, [pair])
