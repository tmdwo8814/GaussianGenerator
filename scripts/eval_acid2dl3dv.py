"""Evaluate ACID-trained Final++ and NoPoSplat on the shared DL3DV protocol.

Run from the repository root: python -m scripts.eval_acid2dl3dv --gpus 0 1
"""

from scripts.eval_re10k2dl3dv import main as evaluate_dl3dv


def main(argv=None) -> None:
    evaluate_dl3dv(argv, training_dataset="acid")


if __name__ == "__main__":
    main()
