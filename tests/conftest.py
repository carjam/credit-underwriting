from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    random.seed(42)
    np.random.seed(42)
