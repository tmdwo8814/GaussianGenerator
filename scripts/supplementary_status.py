"""Print M1-M7 progress and export their measured checkpoint scores."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.misc.supplementary_results import EVAL_STEPS, batch_status, export_results, read_metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch_dir', type=Path)
    parser.add_argument('--write-csv', action='store_true')
    parser.add_argument('--require-complete', choices=[f'M{i}' for i in range(1, 8)])
    args = parser.parse_args(argv)
    if not args.batch_dir.is_dir():
        parser.error(f'Batch directory does not exist: {args.batch_dir}')
    print('Model  GPU  State         Train step   Evaluated checkpoints')
    for row in batch_status(args.batch_dir):
        evaluated = ', '.join(str(step) for step in row['evaluated_steps']) or '-'
        print(f'{row["variant"]:5}  {row["gpu"]:3}  {row["state"]:12}  {row["train_step"]:5}/10000  {evaluated}')
    if args.write_csv:
        print(f'Results CSV: {export_results(args.batch_dir)}')
    if args.require_complete:
        directory = args.batch_dir / args.require_complete
        if any(read_metrics(directory, step) is None for step in EVAL_STEPS):
            parser.error(f'{args.require_complete}: both 5k and 10k score files are required')


if __name__ == '__main__':
    main()
