import unittest

import torch

from model import GPT, GPTConfig
from train import batch_from


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = GPT(GPTConfig(vocab_size=32, context=8, width=16, heads=4))

    def test_shapes_and_cross_entropy(self):
        ids = torch.randint(32, (2, 5))
        targets = torch.randint(32, (2, 5))
        logits, loss = self.model(ids, targets)
        self.assertEqual(logits.shape, (2, 5, 32))
        expected = -logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).mean()
        torch.testing.assert_close(loss, expected)

    def test_future_tokens_do_not_change_prefix_logits(self):
        first, _ = self.model(torch.tensor([[1, 2, 3, 4, 5]]))
        second, _ = self.model(torch.tensor([[1, 2, 3, 9, 8]]))
        torch.testing.assert_close(first[:, :3], second[:, :3])

    def test_backward_and_weight_sharing(self):
        ids = torch.tensor([[1, 2, 3]])
        _, loss = self.model(ids, torch.tensor([[2, 3, 4]]))
        loss.backward()
        self.assertIs(self.model.tokens.weight, self.model.head.weight)
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.model.blocks[0].attention.qkv.weight.grad.abs().sum().item(), 0)

    def test_generation_rolls_context(self):
        prefix = torch.arange(10).unsqueeze(0)
        logits, _ = self.model(prefix[:, -8:])
        output = self.model.generate(prefix, 4, top_k=1)
        self.assertEqual(output.shape, (1, 14))
        torch.testing.assert_close(output[:, :10], prefix)
        torch.testing.assert_close(output[:, 10], logits[:, -1].argmax(-1))

    def test_targets_are_shifted(self):
        x, y = batch_from(torch.arange(20), 3, 5, torch.Generator().manual_seed(2), "cpu")
        self.assertEqual(x.shape, (3, 5))
        torch.testing.assert_close(x + 1, y)

    def test_invalid_config_and_context(self):
        with self.assertRaises(ValueError):
            GPTConfig(width=17, heads=4)
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 9), dtype=torch.long))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()

