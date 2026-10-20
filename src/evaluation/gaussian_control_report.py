"""Scene-weighted metrics for configured Gaussian retention conditions."""

import json
from math import isfinite
from pathlib import Path


FRACTIONS = (.75, .50, .25)
CONDITIONS = ('final_100',) + tuple(
    f'{method}_{round(fraction * 100):03d}'
    for fraction in FRACTIONS for method in ('prune', 'redistribute')
)
GROUPS = ('overall', 'small', 'medium', 'large')
METRICS = ('psnr', 'ssim', 'lpips')


def retention_fractions(values=None) -> tuple[float, ...]:
    fractions = tuple(FRACTIONS if values is None else values)
    if (not fractions or any(isinstance(value, bool) or not isinstance(value, (int, float))
                            or not isfinite(value) or not 0 < value < 1 for value in fractions)):
        raise ValueError('Retention fractions must be a nonempty list of numbers strictly between 0 and 1')
    suffixes = [round(value * 100) for value in fractions]
    if any(abs(value * 100 - suffix) > 1e-8 for value, suffix in zip(fractions, suffixes)):
        raise ValueError('Use whole percentage retention fractions, for example [0.05,0.10]')
    if len(set(suffixes)) != len(fractions):
        raise ValueError('Retention fractions must be unique')
    return tuple(float(value) for value in fractions)


class GaussianControlReport:
    def __init__(self, directory: Path, metadata: dict, *, fractions=None, reuse_baseline=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.fractions = retention_fractions(fractions)
        self.reuse_baseline = reuse_baseline
        reduced = tuple(f'{method}_{round(fraction * 100):03d}'
                        for fraction in self.fractions for method in ('prune', 'redistribute'))
        self.conditions = reduced if reuse_baseline else ('final_100',) + reduced
        self.counts = {name: dict.fromkeys(GROUPS, 0) for name in ('final_100',) + reduced}
        self.sums = {name: {group: {} for group in GROUPS} for name in self.counts}
        self.samples = self.directory / 'per_scene.jsonl'
        if self.samples.exists():
            raise FileExistsError(f'Refusing to mix evaluations in {self.directory}; use a new output directory')
        self.samples.touch()
        (self.directory / 'run_status.txt').write_text('RUNNING\n', encoding='utf-8')
        (self.directory / 'metadata.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8',
        )

    def add_scene(self, identity: dict, overlap: str, conditions: dict, *, reference_final=None):
        if set(conditions) != set(self.conditions):
            raise ValueError('Every scene must supply exactly the configured conditions')
        values_to_add = dict(conditions)
        if self.reuse_baseline:
            if reference_final is None or any(key not in reference_final for key in METRICS):
                raise ValueError('Reused baseline requires per-scene Final PSNR, SSIM and LPIPS')
            values_to_add['final_100'] = reference_final
        for name, values in values_to_add.items():
            groups = ('overall', overlap) if overlap in GROUPS[1:] else ('overall',)
            for group in groups:
                self.counts[name][group] += 1
                sums = self.sums[name][group]
                for key, value in values.items():
                    sums[key] = sums.get(key, 0.) + value
        with self.samples.open('a', encoding='utf-8') as stream:
            record = {**identity, 'overlap_group': overlap, 'conditions': conditions}
            if self.reuse_baseline:
                record['reference_final_100'] = reference_final
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    def average(self, name: str, group: str = 'overall') -> dict:
        count = self.counts[name][group]
        return {key: value / count for key, value in self.sums[name][group].items()} if count else {}

    def condition_text(self, name: str) -> str:
        limit = self.metadata.get('max_scenes', 0)
        scope = f'first {limit} scenes (limited run)' if limit else 'full evaluation loader'
        lines = [f'Condition: {name}', f'Protocol: {self.metadata["camera_protocol"]}',
                 f'Checkpoint: {self.metadata["checkpoint"]}',
                 ('Baseline: reused per-scene Final from ' + self.metadata.get('reference_run', '')
                  if self.reuse_baseline else 'Baseline: Final evaluated in this run'),
                 f'Scope: {scope}',
                 'Aggregation: average target views within each scene, then equal weight per scene.',
                 'PSNR / SSIM: higher is better. LPIPS: lower is better.', '',
                 'Group       Scenes       PSNR       SSIM      LPIPS     dPSNR      dSSIM     dLPIPS']
        for group in GROUPS:
            count = self.counts[name][group]
            if not count:
                lines.append(f'{group:<12} {0:6d}         NA         NA         NA')
                continue
            values, baseline = self.average(name, group), self.average('final_100', group)
            columns = [values[key] for key in METRICS]
            columns += [values[key] - baseline[key] for key in METRICS]
            lines.append(f'{group:<12} {count:6d} ' + ' '.join(f'{x:10.6f}' for x in columns))
        lines += ['', 'Diagnostics and timings (arithmetic means of scene-level statistics):']
        for key, value in self.average(name).items():
            if key not in METRICS:
                lines.append(f'{key}: {value:.9g}')
        lines += ['', 'budget_error is expected for pruning, and must be <= 1e-5 relative for redistribution.',
                  'covariance_radius = sqrt(trace(covariance)) in the original scene coordinate system.',
                  'Opacity/radius quantiles are computed per scene and then averaged.',
                  'shared_encoder_ms includes backbone, DPT and the original full Final decoding.',
                  'additional_decode_ms includes masking, fallback search/transfers, pooling and new attributes.',
                  'render_ms measures a synchronized render of all targets; pose alignment is timed separately.',
                  'Timings include initial kernel/allocator warmup; inspect per_scene.jsonl for individual scenes.']
        return '\n'.join(lines) + '\n'

    def finish(self):
        if not self.counts['final_100']['overall']:
            raise RuntimeError('No evaluation scenes were processed; no final performance is available')
        texts = []
        for name in self.conditions:
            content = self.condition_text(name)
            (self.directory / f'{name}.txt').write_text(content, encoding='utf-8')
            texts.append(content)
        lines = ['Redistribution minus pruning (signed difference; negative LPIPS is an improvement):',
                 'Retention Group       Scenes      dPSNR      dSSIM     dLPIPS']
        for fraction in self.fractions:
            suffix = f'{round(fraction * 100):03d}'
            for group in GROUPS:
                prune = self.average(f'prune_{suffix}', group)
                regen = self.average(f'redistribute_{suffix}', group)
                count = self.counts[f'prune_{suffix}'][group]
                if not count:
                    lines.append(f'{fraction:7.0%}   {group:<12} {0:6d}         NA         NA         NA')
                    continue
                differences = [regen[key] - prune[key] for key in METRICS]
                lines.append(f'{fraction:7.0%}   {group:<12} {count:6d} ' +
                             ' '.join(f'{x:10.6f}' for x in differences))
        comparison = '\n'.join(lines) + '\n'
        (self.directory / 'redistribution_minus_pruning.txt').write_text(comparison, encoding='utf-8')
        summary = '\n'.join(texts) + '\n' + comparison
        (self.directory / 'summary.txt').write_text(summary, encoding='utf-8')
        (self.directory / 'run_status.txt').write_text('COMPLETE\n', encoding='utf-8')
        return summary
