import json
from pathlib import Path
import unittest
from scripts.run_experiments import command_plan, ROOT
from scripts.package_release import files_for_release


class OrchestrationTests(unittest.TestCase):
    def test_recorded_training_plan(self):
        config = json.loads((ROOT / "configs/production.json").read_text())
        plan = command_plan(config, ROOT / "runs/test", "all", list(range(5)))
        self.assertEqual(len(plan), 22)  # 5 SSL + 5 main + 9 ablation + aggregate + input build + comparators
        for label, cmd, _ in plan:
            self.assertFalse(any(x.startswith("--max-") for x in cmd))
            if label.startswith("main_all"):
                self.assertEqual(cmd[cmd.index("--batch-size") + 1], "16")
                self.assertEqual(cmd[cmd.index("--accumulate") + 1], "1")
                self.assertEqual(cmd[cmd.index("--epochs") + 1], "35")
            if label.startswith(("A0_", "A1_", "A2_", "A3_", "signal_ablation_")):
                self.assertEqual(cmd[cmd.index("--test-fold") + 1], "0")
                self.assertEqual(cmd[cmd.index("--epochs") + 1], "10")
        a0 = next(cmd for label, cmd, _ in plan if label.startswith("A0_"))
        self.assertIn("--disable-calibration", a0); self.assertNotIn("--ssl-checkpoint", a0)

    def test_archive_selection_excludes_datasets_and_preserves_weights(self):
        files = list(files_for_release())
        self.assertEqual(sum(p.suffix == ".pt" for p in files), 39)
        self.assertFalse(any(p.suffix in {".edf", ".xml", ".npz", ".h5", ".npy"} for p in files))
        self.assertFalse(any("results_smoke" in p.parts or "pretrained_smoke" in p.parts for p in files))


if __name__ == "__main__": unittest.main()
