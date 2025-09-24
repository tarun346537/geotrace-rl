import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from geotrace.data import generate, label_records, write_jsonl


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install .[train] for PyTorch tests")
class TrainingTests(unittest.TestCase):
    def test_interrupted_training_matches_uninterrupted_weights(self) -> None:
        import torch

        from geotrace.train_prm import train

        local = Path(__file__).resolve().parents[1] / "outputs" / "test-temp"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as directory:
            directory = Path(directory)
            source = directory / "labels.jsonl"
            write_jsonl(source, label_records(generate(10), samples=4))

            def args(
                output: str, max_steps: int = 0, resume: Path | None = None
            ) -> SimpleNamespace:
                return SimpleNamespace(
                    input=source,
                    output=directory / output,
                    epochs=2,
                    hidden=12,
                    batch_size=16,
                    lr=0.003,
                    seed=17,
                    max_steps=max_steps,
                    resume=resume,
                    wandb_project=None,
                    run_id=None,
                    synthetic=True,
                )

            with contextlib.redirect_stdout(io.StringIO()):
                train(args("full"))
                train(args("resumed", max_steps=3))
                train(args("resumed", resume=directory / "resumed" / "checkpoint.pt"))
            full = torch.load(directory / "full" / "checkpoint.pt", weights_only=True)
            resumed = torch.load(directory / "resumed" / "checkpoint.pt", weights_only=True)
            self.assertEqual(full["step"], resumed["step"])
            for key, value in full["model"].items():
                torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
            torch.testing.assert_close(full["torch_rng"], resumed["torch_rng"], rtol=0, atol=0)
            self.assertEqual(full["numpy_rng"], resumed["numpy_rng"])


if __name__ == "__main__":
    unittest.main()
