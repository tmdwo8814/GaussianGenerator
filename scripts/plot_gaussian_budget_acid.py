#!/usr/bin/env python3
"""Plot RE10K -> ACID Gaussian control directly from saved evaluation results.

Default input: outputs/gaussian_control/re10k_to_acid/summary.json
Default outputs: outputs/gaussian_control/figures/re10k_to_acid/
Paths are relative to this repository, independent of the working directory.

Uses overall LightGaussian-style, NanoGS-style and Redistribute results at
10%, 5%, 1%, plus the shared 100% reference. Offsets and latency ratios are
computed from the JSON. No PDF, Downloads folder, CUDA or inference is needed.
Dependency: matplotlib. pypdf and pymupdf are no longer used.

Offset = Ours - Merging. Improvement is green, degradation red, exact zero gray.
For LPIPS a negative offset means improvement. Small nonzero offsets are colored.

Examples:
    python scripts/plot_gaussian_budget_acid.py
    python scripts/plot_gaussian_budget_acid.py --input outputs/gaussian_control/re10k_to_acid
    python scripts/plot_gaussian_budget_acid.py --percentages 75 50 25 10 5 1
"""

import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'outputs/gaussian_control/re10k_to_acid/summary.json'
DEFAULT_OUTPUT = ROOT / 'outputs/gaussian_control/figures/re10k_to_acid'
GREEN, RED, NEUTRAL = '#008000', '#d62728', '#555555'
METHODS = ('lightgaussian', 'merge', 'redistribute')
METRICS = ('psnr', 'ssim', 'lpips')
COLORS = dict(lightgaussian='#adb5bf', merge='#4d6bff', redistribute='#e65c00')
LABELS = {'lightgaussian': 'Pruning (LightGaussian-style)',
          'merge': 'Merging (NanoGS-style)', 'redistribute': 'Redistribute (Ours)'}
MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def offset_color(metric, delta):
    improvement = -delta if metric.lower() == 'lpips' else delta
    return GREEN if improvement > 0 else RED if improvement < 0 else NEUTRAL


def offset_label(metric, delta):
    places = 2 if metric == 'psnr' else 3
    if delta != 0 and delta.quantize(Decimal(1).scaleb(-places)) == 0:
        text = f'{delta:+.2E}'
    else:
        text = f'{delta:+.{places}f}' if delta else f'{delta:.{places}f}'
    return text + (' dB' if metric == 'psnr' else '')


def count_label(count):
    return f'{count / 1000:.2g}k' if count >= 1000 else f'{count:g}'


def load_results(source, percentages):
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        source /= 'summary.json'
    if not source.is_file():
        raise FileNotFoundError(f'ACID results not found: {source}\n'
                                'Use --input /path/to/re10k_to_acid/summary.json')
    if source.suffix.lower() != '.json':
        raise ValueError('Input must be the evaluation summary.json, not a PDF')
    raw = read_json(source)
    if not isinstance(raw, list):
        raise ValueError('Expected the JSON array written by the ACID evaluation summary')
    rows = {}
    for row in raw:
        if row.get('overlap') != 'overall':
            continue
        key = row['method'], int(row['percent'])
        if key in rows:
            raise ValueError(f'Duplicate overall condition: {key}')
        rows[key] = row
    required = [(m, p) for m in METHODS for p in percentages]
    missing = [key for key in required if key not in rows]
    if missing:
        raise ValueError(f'Missing conditions: {missing}. Use combined re10k_to_acid/summary.json, '
                         'not controls/summary.json')
    references = [rows[key] for key in [('final', 100), ('redistribute', 100), ('merge', 100),
                  ('lightgaussian', 100), ('prune', 100)] if key in rows]
    if not references:
        raise ValueError('Shared 100% reference is absent from summary.json')
    baseline = references[0]
    chosen = [rows[key] for key in required] + references
    for row in chosen:
        if int(row['scenes']) < 1 or int(row['scenes']) != row['scenes']:
            raise ValueError('Each condition needs a positive integer scene count')
        for key in (*METRICS, 'postprocess_ms', 'gaussians'):
            if not math.isfinite(float(row[key])):
                raise ValueError(f'Non-finite {key}: {row["method"]}/{row["percent"]}')
        if row['gaussians'] <= 0:
            raise ValueError('Gaussian counts must be positive')
        if row['percent'] != 100 and row['postprocess_ms'] <= 0:
            raise ValueError('Control latency must be positive for a logarithmic axis')
    if len({r['scenes'] for r in chosen}) != 1:
        raise ValueError('Scene counts differ; regenerate a combined summary on the same completed scenes')
    for reference in references[1:]:
        if any(reference[k] != baseline[k] for k in (*METRICS, 'gaussians')):
            raise ValueError('100% reference values differ between methods')
    for p in percentages:
        counts = [rows[m, p]['gaussians'] for m in METHODS]
        if not all(math.isclose(n, counts[0], rel_tol=1e-9, abs_tol=1e-6) for n in counts):
            raise ValueError(f'Gaussian counts differ between methods at {p}%')
    status_path = source.parent / 'summary_status.json'
    status = read_json(status_path) if status_path.exists() else None
    if status and status['common_scenes'] != baseline['scenes']:
        raise ValueError('Summary/status scene counts differ; rerun the report stage and retry')
    return source, rows, baseline, status


def plot_results(source, output_dir, percentages=(10, 5, 1), dpi=300, pdf_only=False):
    percentages = list(percentages)
    if (not percentages or len(set(percentages)) != len(percentages)
            or any(type(p) is not int or not 1 <= p < 100 for p in percentages)):
        raise ValueError('Use distinct integer percentages from 1 to 99; 100% is the reference')
    if dpi <= 0:
        raise ValueError('DPI must be positive')
    source, rows, baseline, status = load_results(source, percentages)
    output_dir = Path(output_dir).expanduser().resolve()
    stem = output_dir / 'fig_budget_latency_re10k_acid'
    annotations = []
    speed = {p: rows['merge', p]['postprocess_ms'] / rows['redistribute', p]['postprocess_ms']
             for p in percentages}
    low, high = min(speed.values()), max(speed.values())
    if low > 1:
        speed_label = f'{low:.0f}\N{EN DASH}{high:.0f}\N{MULTIPLICATION SIGN} faster'
    elif high < 1:
        speed_label = f'{1/high:.1f}\N{EN DASH}{1/low:.1f}\N{MULTIPLICATION SIGN} slower'
    else:
        speed_label = f'Merging / Ours: {low:.1f}\N{EN DASH}{high:.1f}\N{MULTIPLICATION SIGN}'
    latencies = [rows[m, p]['postprocess_ms'] for m in METHODS for p in percentages]
    xmin, xmax = min(latencies) * .6, max(latencies) * 12
    titles = ('PSNR (dB) \N{UPWARDS ARROW}', 'SSIM \N{UPWARDS ARROW}', 'LPIPS \N{DOWNWARDS ARROW}')
    style = {'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 10,
             'axes.titlesize': 12, 'axes.labelsize': 11, 'pdf.fonttype': 42,
             'ps.fonttype': 42, 'svg.fonttype': 'none'}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6))
        try:
            fig.subplots_adjust(left=.055, right=.985, bottom=.17, top=.77, wspace=.34)
            for ax, metric, title in zip(axes, METRICS, titles):
                ax.set_xscale('log')
                ax.set_xlim(xmin, xmax)
                ax.xaxis.set_major_locator(LogLocator(base=10))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_major_formatter(FuncFormatter(
                    lambda x, pos: f'{x/1000:g}k' if x >= 1000 else f'{x:g}'))
                ax.set_xlabel('control latency (ms, log)')
                ax.set_title(title)
                ax.grid(True, color='#e6e6e6', linewidth=.8)
                ax.set_axisbelow(True)
                reference = baseline[metric]
                ax.axhline(reference, color='#222222', linestyle='--', linewidth=1.)
                ax.annotate('Full (100%)', xy=(.015, reference), xycoords=('axes fraction', 'data'),
                            xytext=(0, -3), textcoords='offset points', va='top', fontsize=8.5)
                for i, p in enumerate(percentages):
                    marker = MARKERS[i % len(MARKERS)]
                    merging, ours = rows['merge', p], rows['redistribute', p]
                    ax.annotate('', xy=(ours['postprocess_ms'], ours[metric]),
                                xytext=(merging['postprocess_ms'], merging[metric]),
                                arrowprops={'arrowstyle': '->', 'color': '#89929c',
                                            'linestyle': '--', 'lw': 1.1, 'shrinkA': 5, 'shrinkB': 6}, zorder=2)
                    for method in METHODS:
                        row = rows[method, p]
                        ax.scatter(row['postprocess_ms'], row[metric], s=92 if method == 'redistribute' else 65,
                                   marker=marker, color=COLORS[method], zorder=4,
                                   edgecolors='#222222' if method == 'redistribute' else 'white',
                                   linewidths=1.3 if method == 'redistribute' else .8)
                    delta = Decimal(str(ours[metric])) - Decimal(str(merging[metric]))
                    label, color = offset_label(metric, delta), offset_color(metric, delta)
                    ax.annotate(label, xy=(merging['postprocess_ms'], merging[metric]),
                                xytext=(7, 0), textcoords='offset points', ha='left', va='center',
                                fontsize=8.7, color=color, zorder=5)
                    annotations.append({'metric': metric, 'percent': p, 'ours_minus_merging': str(delta),
                                        'label': label, 'color': color})
                values = [rows[m, p][metric] for m in METHODS for p in percentages] + [reference]
                bottom, top = min(values), max(values)
                pad = max((top - bottom) * .08, .015 if metric != 'psnr' else .5)
                if metric == 'lpips':
                    ax.set_ylim(top + pad, max(0, bottom - pad))
                elif metric == 'ssim':
                    ax.set_ylim(min(0, bottom - pad), min(1, top + pad))
                else:
                    ax.set_ylim(bottom - pad, top + pad)
                ax.text(.98, .025, '\N{GREEK CAPITAL LETTER DELTA}: Ours \N{MINUS SIGN} Merging',
                        transform=ax.transAxes, ha='right', fontsize=8, color='#666666', style='italic')
            axes[0].annotate('', xy=(.29, .8), xytext=(.64, .8), xycoords='axes fraction',
                             arrowprops={'arrowstyle': '-|>', 'lw': 1.8, 'color': COLORS['redistribute']})
            axes[0].text(.465, .835, speed_label, transform=axes[0].transAxes, ha='center',
                         fontsize=9.5, weight='bold', color=COLORS['redistribute'])
            method_handles = [Line2D([], [], marker='o', linestyle='', markersize=8,
                                     color=COLORS[m], markeredgecolor='#222222' if m == 'redistribute' else COLORS[m],
                                     label=LABELS[m]) for m in METHODS]
            budget_handles = [Line2D([], [], marker=MARKERS[i % len(MARKERS)], linestyle='', color='#555555',
                                     markersize=7, label=f'{p}% ({count_label(rows["redistribute", p]["gaussians"])})')
                              for i, p in enumerate(percentages)]
            fig.legend(handles=method_handles, loc='upper center', bbox_to_anchor=(.5, 1.0),
                       ncol=3, frameon=False, fontsize=9.5, columnspacing=1.6, handletextpad=.4)
            fig.legend(handles=budget_handles, loc='upper center', bbox_to_anchor=(.5, .925),
                       ncol=min(len(budget_handles), 6), frameon=False, fontsize=9.3, columnspacing=1.8)
            output_dir.mkdir(parents=True, exist_ok=True)
            formats = ['pdf'] if pdf_only else ['pdf', 'png', 'svg']
            outputs = []
            for extension in formats:
                destination = stem.with_suffix('.' + extension)
                options = {'metadata': {'Title': 'RE10K to ACID Gaussian budget adaptation'}} if extension == 'pdf' else {}
                fig.savefig(destination, format=extension, dpi=dpi, bbox_inches='tight', pad_inches=.04, **options)
                outputs.append(destination)
        finally:
            plt.close(fig)
    metadata = {'source_summary': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'evaluation': 'RE10K -> ACID', 'overlap': 'overall', 'scenes': baseline['scenes'],
                'evaluation_status': status, 'percentages': percentages,
                'comparison': 'Ours minus Merging', 'higher_is_better': dict(psnr=True, ssim=True, lpips=False),
                'color_rule': 'performance improvement green; degradation red; only exactly zero neutral',
                'green': GREEN, 'red': RED, 'neutral': NEUTRAL, 'annotations': annotations,
                'merging_over_ours_latency_ratio': speed, 'speed_label': speed_label,
                'baseline': baseline, 'plotted_rows': [rows[m, p] for m in METHODS for p in percentages],
                'outputs': list(map(str, outputs))}
    (output_dir / 'figure_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False),
                                                   encoding='utf-8')
    return outputs, metadata


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--input', '--summary', type=Path, default=DEFAULT_INPUT,
                     help='Combined ACID summary.json, or its parent directory')
    cli.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    cli.add_argument('--percentages', type=int, nargs='+', default=[10, 5, 1])
    cli.add_argument('--dpi', type=int, default=300)
    cli.add_argument('--pdf-only', action='store_true')
    args = cli.parse_args(argv)
    outputs, metadata = plot_results(args.input, args.output_dir, args.percentages, args.dpi, args.pdf_only)
    print(f"[Plot] source={metadata['source_summary']}; same completed scenes={metadata['scenes']}")
    status = metadata['evaluation_status']
    if status and not status.get('complete'):
        print(f"[Plot] Partial common cohort: {status['common_scenes']}/{status['expected_scenes']} scenes")
    for item in metadata['annotations']:
        print(f"{item['metric']:5s} {item['percent']:3d}% {item['label']:12s} -> {item['color']}")
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()