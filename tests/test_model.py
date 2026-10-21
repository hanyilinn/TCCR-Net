"""Synthetic CPU checks; no EEG datasets or pretrained weights are read."""

import unittest

import torch

from tccrnet import TCCRNet


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_both_modes_shapes_finite_and_stateless(self):
        for spiking in (False, True):
            with self.subTest(spiking=spiking):
                torch.manual_seed(7)
                model = TCCRNet(spiking=spiking).eval()
                x = torch.randn(2, 512)
                with torch.inference_mode():
                    y, stats = model(x)
                    y_with_channel, _ = model(x.unsqueeze(1))
                self.assertEqual(tuple(y.shape), (2, 512))
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(all(torch.isfinite(v).all() for v in stats.values()))
                torch.testing.assert_close(y, y_with_channel, rtol=0, atol=0)
                self.assertEqual(sum(p.numel() for p in model.parameters()),
                                 3135302 if spiking else 3135270)
                restored = TCCRNet(**model.config()).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                with torch.inference_mode():
                    restored_y, _ = restored(x)
                torch.testing.assert_close(y, restored_y, rtol=0, atol=0)

    def test_backward_in_both_modes(self):
        for spiking in (False, True):
            with self.subTest(spiking=spiking):
                torch.manual_seed(11)
                model = TCCRNet(spiking=spiking).train()
                x = torch.randn(2, 1, 512, requires_grad=True)
                y, _ = model(x)
                loss = (y - torch.randn_like(y)).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(x.grad).all())
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(torch.count_nonzero(g) for g in gradients))

    def test_input_contract(self):
        model = TCCRNet().eval()
        for shape in ((1, 2, 512), (1, 511), (512,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(*shape))
        for config in ({"signal_length": 513}, {"stage_count": 4},
                       {"decoder_mode": "global"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                TCCRNet(**config)


if __name__ == "__main__":
    unittest.main()
