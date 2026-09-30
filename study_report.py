"""Offline, dependency-free presentation of paired experiment evidence."""

from html import escape
from pathlib import Path
from typing import Any


def paired_association_summary(runs: list[dict[str, Any]], opportunities: int) -> dict[str, Any]:
    """Describe seed-level effects without treating overlapping windows as replicates."""
    if not runs or opportunities <= 0:
        raise ValueError('Paired summaries require runs and positive detection opportunities.')
    pairs = sorted(runs[0]['coupled']['edge_counts'])
    result = {}
    for pair in pairs:
        differences = [
            (run['coupled']['edge_counts'][pair] - run['uncoupled']['edge_counts'][pair]) / opportunities
            for run in runs
        ]
        result[pair] = {
            'mean_rate_difference': sum(differences) / len(differences),
            'rate_difference_range': [min(differences), max(differences)],
            'positive_seeds': sum(value > 0 for value in differences),
            'zero_seeds': sum(value == 0 for value in differences),
            'negative_seeds': sum(value < 0 for value in differences),
            'per_seed_rate_differences': differences,
        }
    return result


def render_study_report(report: dict[str, Any], output_path: Path) -> Path:
    """Render a portable HTML companion; no network, scripts, or external assets."""
    opportunities = report['opportunities_per_pair_per_run']
    effects = paired_association_summary(report['runs'], opportunities)
    rows = []
    details = []
    for pair, effect in effects.items():
        rates = report['association_rates'][pair]
        label = report['injected_associations'].get(pair, 'no direct injected coupling')
        low, high = effect['rate_difference_range']
        rows.append(
            f'<tr><th scope="row">{escape(pair)}<small>{escape(label)}</small></th>'
            f'<td>{rates["coupled"]:.2%}</td><td>{rates["uncoupled"]:.2%}</td>'
            f'<td>{100 * effect["mean_rate_difference"]:+.2f} pp</td>'
            f'<td>{100 * low:+.2f} to {100 * high:+.2f} pp</td>'
            f'<td>{effect["positive_seeds"]} / {effect["zero_seeds"]} / {effect["negative_seeds"]}</td></tr>'
        )
    pairs = list(effects)
    for index, run in enumerate(report['runs']):
        cells = ''.join(f'<td>{100 * effects[pair]["per_seed_rate_differences"][index]:+.2f}</td>' for pair in pairs)
        details.append(f'<tr><th scope="row">{run["seed"]}</th>{cells}</tr>')
    config = report['passport']['configuration']
    settings = ''.join(f'<tr><th scope="row">{escape(str(k))}</th><td>{escape(str(v))}</td></tr>'
                       for k, v in config.items())
    limitations = ''.join(f'<li>{escape(item)}</li>' for item in report['limitations'])
    headings = ''.join(f'<th scope="col">{escape(pair)}</th>' for pair in pairs)
    source = escape(report['passport']['source_sha256'])
    environment = escape(str(report['passport']['dependencies']))
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IONS-X · Paired experiment report</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#081b20;color:#eaf4f3;
font:16px/1.65 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:48px 24px}}
h1{{font-size:clamp(2rem,5vw,3.7rem);line-height:1.12;max-width:850px}}h2{{margin-top:40px}}
.eyebrow{{color:#79d4ca;letter-spacing:.14em;text-transform:uppercase;font-size:.8rem}}
.lead{{max-width:800px;color:#bfd4d4}}.stats{{display:flex;flex-wrap:wrap;gap:16px;margin:30px 0}}
.stat{{background:#123039;padding:18px 24px;border-radius:12px;flex:1;min-width:180px}}
.stat strong{{display:block;font-size:1.8rem;color:#89e1d6}}.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:13px;
border-bottom:1px solid #315059;text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}
thead th{{color:#89e1d6;font-size:.85rem}}small{{display:block;color:#adc5c5;font-weight:400}}
.note{{border-left:3px solid #79d4ca;padding:12px 20px;background:#102b32}}code{{overflow-wrap:anywhere}}
summary{{cursor:pointer;color:#89e1d6}}footer{{margin-top:40px;color:#adc5c5;font-size:.85rem}}
@media print{{:root{{color-scheme:light}}body{{background:white;color:black}}.lead,small,footer{{color:#333}}
.stat,.note{{background:#eee}}.stat strong,thead th,summary,.eyebrow{{color:#07554e}}}}
</style></head><body><main>
<p class="eyebrow">Vers3Dynamics / IONS-X Deep Emergence Lab</p>
<h1>What changes when the coupling disappears?</h1>
<p class="lead">A paired synthetic experiment. Each seed replays the same initial field, moving agents,
and moderator schedule. The uncoupled arm removes the two explicit cross-channel coupling terms.</p>
<div class="stats"><div class="stat"><strong>{report['repetitions']}</strong>paired seeds</div>
<div class="stat"><strong>{config['FRAMES']}</strong>frames per arm</div>
<div class="stat"><strong>{opportunities:,}</strong>eligible agent-windows per pair / arm</div></div>
<h2>Association response</h2><p>Mean rates across seeds. Differences are coupled minus uncoupled in
percentage points (pp). The range describes observed seed variation, not a confidence interval.</p>
<div class="scroll"><table><thead><tr><th scope="col">Pair / injected relationship</th>
<th scope="col">Coupled</th><th scope="col">Uncoupled</th><th scope="col">Difference</th>
<th scope="col">Seed range</th><th scope="col">Seeds + / = / −</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="note">A higher detection rate measures response to this synthetic intervention.
It does not establish causality in empirical data. Pairs without direct coupling may still change
indirectly. Agents and overlapping windows are not independent experimental replicates.</p>
<h2>Every seed, every pair</h2><p>Rate differences in percentage points. Negative and zero results are retained.</p>
<div class="scroll"><table><thead><tr><th scope="col">Seed</th>{headings}</tr></thead>
<tbody>{''.join(details)}</tbody></table></div>
<h2>Interpretation boundaries</h2><ul>{limitations}</ul>
<details><summary>Effective configuration and provenance</summary><div class="scroll"><table>
<tbody>{settings}</tbody></table></div><p>Engine SHA-256: <code>{source}</code></p>
<p>Dependencies: <code>{environment}</code></p></details>
<footer>Generated from the adjacent JSON report. No external resources or tracking. The JSON is the
machine-readable record; this page is its descriptive companion.</footer></main></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    return output_path
