"""Phase 1 smoke tests.

These do not test any ML logic yet (there isn't any) -- they only
verify that the project skeleton itself is sound: the package
imports, the config loads and contains the frozen hyperparameters,
seeding is deterministic, and logging writes to disk. If these pass,
Phase 1 is validated and Phase 2 can build on top of it.
"""

import numpy as np
import torch

from fedpda_ids.utils.config import load_config
from fedpda_ids.utils.logging_utils import setup_logging
from fedpda_ids.utils.seed import set_seed


def test_config_loads_and_has_frozen_values():
    config = load_config()

    assert config["sequence"]["window_size"] == 10
    assert config["sequence"]["stride"] == 5
    assert config["model"]["latent_dim"] == 32
    assert config["model"]["batch_size"] == 64
    assert config["model"]["optimizer"]["learning_rate"] == 0.001
    assert config["federated"]["num_rounds"] == 100
    assert config["federated"]["local_epochs"] == 2
    assert config["federated"]["client_participation_fraction"] == 0.2
    assert config["federated"]["cicids2017"]["num_clients"] == 40
    assert config["federated"]["nbaiot"]["num_clients"] == 45
    assert config["drift"]["adwin_delta"] == 0.002
    assert config["privacy"]["delta"] == 1.0e-5


def test_set_seed_is_deterministic():
    set_seed(42)
    a_np = np.random.rand(5)
    a_torch = torch.rand(5)

    set_seed(42)
    b_np = np.random.rand(5)
    b_torch = torch.rand(5)

    assert np.allclose(a_np, b_np)
    assert torch.allclose(a_torch, b_torch)


def test_logging_writes_to_file(tmp_path):
    logger = setup_logging(log_dir=tmp_path, log_to_file=True, run_name="test")
    logger.info("phase 1 smoke test log line")

    log_files = list(tmp_path.glob("test_*.log"))
    assert len(log_files) == 1
    assert "phase 1 smoke test log line" in log_files[0].read_text(encoding="utf-8")
