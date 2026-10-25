import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from exp3.plot_scale_opacity import main


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        checkpoint = root / "model.ckpt"
        checkpoint.touch()
        index = root / "index.json"
        index.write_text(json.dumps({"scene": {
            "context": [0, 10], "target": [5], "overlap": 0.4,
        }}), encoding="utf-8")
        self.output = root / "output"
        self.arguments = ["--original-checkpoint", str(checkpoint),
                          "--ours-checkpoint", str(checkpoint), "--data-root", str(root),
                          "--index", str(index), "--output-dir", str(self.output)]

    def extract_fixture(self, args, name, scenes, directory):
        row = {"decoder": name, "scene": "scene", "overlap": "medium",
               "N_total": 10, "N_active": 8, "N_selected": 8, "median_log_s_max": -2.0}
        with (directory / "per_scene.csv").open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=row).writerow(row)
        np.savez_compressed(directory / "samples" / f"{name}.npz",
                            log_s_max=np.linspace(-3, -1, 8), opacity=np.linspace(0.1, 0.9, 8),
                            scene_index=np.zeros(8, dtype=int), gaussian_index=np.arange(8))
        return {"N_total": 10, "N_active": 8, "N_selected": 8,
                "checkpoint_step": 140000, "scenes": 1}

    def fail_after_original(self):
        def extract(args, name, scenes, directory):
            counts = self.extract_fixture(args, name, scenes, directory)
            if name == "ours":
                raise RuntimeError("ours interrupted")
            return counts
        with patch("exp3.plot_scale_opacity.extract", side_effect=extract):
            with self.assertRaisesRegex(RuntimeError, "ours interrupted"):
                main(self.arguments)
        # Metadata produced before --resume existed must also be readable.
        path = self.output / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["arguments"].pop("resume")
        path.write_text(json.dumps(metadata), encoding="utf-8")

    def test_resume_keeps_original_and_retries_only_ours_without_duplicate_rows(self):
        self.fail_after_original()
        sample = self.output / "samples/original.npz"
        original_bytes = sample.read_bytes()
        with patch("exp3.plot_scale_opacity.extract", side_effect=self.extract_fixture) as extract:
            main([*self.arguments, "--resume"])
        self.assertEqual([call.args[1] for call in extract.call_args_list], ["ours"])
        self.assertEqual(sample.read_bytes(), original_bytes)
        with (self.output / "per_scene.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([row["decoder"] for row in rows], ["original", "ours"])
        self.assertEqual((self.output / "status.txt").read_text(encoding="utf-8").strip(), "COMPLETE")
        self.assertTrue((self.output / "figs/figC_scale_opacity.png").exists())

    def test_resume_rejects_changed_seed_before_extracting_or_mutating_cache(self):
        self.fail_after_original()
        before = (self.output / "per_scene.csv").read_bytes()
        with patch("exp3.plot_scale_opacity.extract") as extract:
            with self.assertRaisesRegex(ValueError, "changed seed"):
                main([*self.arguments, "--resume", "--seed", "1"])
        extract.assert_not_called()
        self.assertEqual((self.output / "per_scene.csv").read_bytes(), before)

    def test_resume_rejects_missing_completed_sample(self):
        self.fail_after_original()
        (self.output / "samples/original.npz").unlink()
        with patch("exp3.plot_scale_opacity.extract") as extract:
            with self.assertRaises(FileNotFoundError):
                main([*self.arguments, "--resume"])
        extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
