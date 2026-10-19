"""Scene-weighted metrics and human-readable reports for the seven conditions."""

import json
from pathlib import Path


FRACTIONS = (.75, .50, .25)
CONDITIONS = ('final_100',) + tuple(
    f'{method}_{round(fraction * 100):03d}'
    for fraction in FRACTIONS for method in ('prune', 'redistribute')
)
GROUPS = ('overall', 'small', 'medium', 'large')
METRICS = ('psnr', 'ssim', 'lpips')


class GaussianControlReport:
    def __init__(self, directory: Path, metadata: dict):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.counts = {name: dict.fromkeys(GROUPS, 0) for name in CONDITIONS}
        self.sums = {name: {group: {} for group in GROUPS} for name in CONDITIONS}
        self.samples = self.directory / 'per_scene.jsonl'
        if self.samples.exists():
            raise FileExistsError(f'Refusing to mix evaluations in {self.directory}; use a new output directory')
        self.samples.touch()
        (self.directory / 'run_status.txt').write_text('RUNNING\n', encoding='utf-8')
        (self.directory / 'metadata.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8',
        )

    def add_scene(self, identity: dict, overlap: str, conditions: dict):
        if set(conditions) != set(CONDITIONS):
            raise ValueError('Every scene must supply all seven conditions')
        for name, values in conditions.items():
            groups = ('overall', overlap) if overlap in GROUPS[1:] else ('overall',)
            for group in groups:
                self.counts[name][group] += 1
                sums = self.sums[name][group]
                for key, value in values.items():
                    sums[key] = sums.get(key, 0.) + value
        with self.samples.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({**identity, 'overlap_group': overlap,
                                     'conditions': conditions}, ensure_ascii=False) + '\n')

    def average(self, name: str, group: str = 'overall') -> dict:
        count = self.counts[name][group]
        return {key: value / count for key, value in self.sums[name][group].items()} if count else {}

    def condition_text(self, name: str) -> str:
        limit = self.metadata.get('max_scenes', 0)
        scope = f'first {limit} scenes (limited run)' if limit else 'full evaluation loader'
        lines = [f'Condition: {name}', f'Protocol: {self.metadata["camera_protocol"]}',
                 f'Checkpoint: {self.metadata["checkpoint"]}',
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
        for name in CONDITIONS:
            content = self.condition_text(name)
            (self.directory / f'{name}.txt').write_text(content, encoding='utf-8')
            texts.append(content)
        lines = ['Redistribution minus pruning (signed difference; negative LPIPS is an improvement):',
                 'Retention Group       Scenes      dPSNR      dSSIM     dLPIPS']
        for fraction in FRACTIONS:
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
