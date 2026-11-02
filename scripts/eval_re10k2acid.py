"""Evaluate RE10K-trained Final++ and NoPoSplat on the ACID test benchmark.

Run from the repository root: python -m scripts.eval_re10k2acid --gpus 0 1
"""

from scripts.eval_re10k2dl3dv import main as evaluate_cross_domain


def main(argv=None) -> None:
    evaluate_cross_domain(argv, training_dataset="re10k", evaluation_dataset="acid")


if __name__ == "__main__":
    main()
