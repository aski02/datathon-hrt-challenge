from __future__ import annotations

import importlib
import unittest
from pathlib import Path

STRATEGY_DIR = Path(__file__).resolve().parents[1] / "pipeline" / "strategies"
UTILITY_MODULES = {"__init__.py", "model_risk_utils.py"}


class TestStrategyCatalog(unittest.TestCase):
    def test_strategy_modules_import_and_expose_build_strategy(self) -> None:
        strategy_files = sorted(path for path in STRATEGY_DIR.glob("*.py") if path.name not in UTILITY_MODULES)

        self.assertGreater(len(strategy_files), 1)

        for path in strategy_files:
            with self.subTest(strategy_file=path.name):
                module = importlib.import_module(f"pipeline.strategies.{path.stem}")
                build_strategy = getattr(module, "build_strategy", None)
                self.assertTrue(callable(build_strategy))
                strategy = build_strategy()
                self.assertTrue(callable(getattr(strategy, "predict", None)))
                self.assertIsInstance(getattr(strategy, "name", ""), str)


if __name__ == "__main__":
    unittest.main()
