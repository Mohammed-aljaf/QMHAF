import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from deep_learning.model import ModelConfig, QMHAFNet, multitask_loss
from comparison_2024plus.train_comparators import NAMES, build_model
from scripts.predict import waveform_batch


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def batch(self):
        torch.manual_seed(42)
        batch = dict(resp=torch.randn(2, 23, 3, 300), low=torch.randn(2, 23, 5, 30),
            resp_quality=torch.ones(2, 23, 3), low_quality=torch.ones(2, 23, 2),
            epoch_mask=torch.ones(2, 23, dtype=torch.bool), target_valid=torch.ones(2, 23, dtype=torch.bool),
            sleep_target=torch.ones(2, 23), event_target=torch.ones(2, 23),
            target_class=torch.tensor([0, 2]), target_ahi=torch.tensor([3., 20.]))
        batch["epoch_mask"][1, -3:] = False
        return batch

    def test_qmhaf_gradient_with_padding_and_multitask_targets(self):
        model = QMHAFNet(ModelConfig(dimension=32, heads=4, local_layers=1, night_layers=1))
        batch = self.batch(); out = model(batch); loss, _ = multitask_loss(out, batch); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients); self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        torch.testing.assert_close(out["class_probabilities"].sum(1), torch.ones(2))

    def test_annotations_do_not_change_inference(self):
        model = QMHAFNet(ModelConfig(dimension=32, heads=4, local_layers=1, night_layers=1)).eval()
        batch = self.batch()
        waveform_only = {k: batch[k] for k in ("resp", "low", "resp_quality", "low_quality", "epoch_mask")}
        with torch.inference_mode():
            a = model(batch); b = model(waveform_only)
        torch.testing.assert_close(a["class_probabilities"], b["class_probabilities"], atol=0, rtol=0)
        torch.testing.assert_close(a["predicted_ahi"], b["predicted_ahi"], atol=0, rtol=0)

    def test_all_comparator_families_have_four_finite_outputs(self):
        for name in NAMES:
            with self.subTest(architecture=name), torch.inference_mode():
                result = build_model(name, 16).eval()(torch.randn(2, 256, 16))
                self.assertEqual(result.shape, (2, 4)); self.assertTrue(torch.isfinite(result).all())

    def test_waveform_loader_needs_no_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "synthetic.npz"
            # Synthetic signals are created only in a temporary directory.
            np.savez(p, resp_q10=np.zeros((3, 600), dtype=np.int16), resp_quant_scale=16.,
                resp_valid_1hz=np.ones((3, 60), dtype=bool), spo2_1hz=np.full(60, 95.),
                spo2_valid_1hz=np.ones(60, dtype=bool), hr_1hz=np.full(60, 70.), hr_valid_1hz=np.ones(60, dtype=bool))
            b = waveform_batch(p)
            self.assertEqual(b["resp"].shape, (1, 2, 3, 300)); self.assertEqual(b["low"].shape, (1, 2, 5, 30))
            self.assertEqual(set(b), {"resp", "low", "resp_quality", "low_quality", "epoch_mask"})


if __name__ == "__main__": unittest.main()
