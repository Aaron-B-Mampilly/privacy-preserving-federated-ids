"""CLI entry point for Phase 2 Part A: CICIDS2017 preprocessing.

Usage:
    python scripts/preprocess_cicids2017.py
    python scripts/preprocess_cicids2017.py --config configs/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedpda_ids.data.cicids2017 import preprocess_cicids2017
from fedpda_ids.utils.config import load_config
from fedpda_ids.utils.logging_utils import setup_logging
from fedpda_ids.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess CICIDS2017 (Phase 2 Part A)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"],
        level=config["logging"]["level"],
        log_to_file=config["logging"]["log_to_file"],
        run_name="preprocess_cicids2017",
    )

    data_cfg = config["data"]
    cic_cfg = data_cfg["cicids2017"]

    logger.info("Starting CICIDS2017 preprocessing")
    metadata = preprocess_cicids2017(
        raw_dir=cic_cfg["raw_dir"],
        processed_dir=cic_cfg["processed_dir"],
        file_to_day=cic_cfg["files"],
        zero_day_holdout_labels=cic_cfg["zero_day_holdout_labels"],
        train_fraction=data_cfg["train_fraction"],
        val_fraction=data_cfg["val_fraction"],
        log1p_skew_threshold=data_cfg["log1p_skew_threshold"],
        dirichlet_alpha_values=config["federated"]["cicids2017"]["dirichlet_alpha_values"],
        num_clients=config["federated"]["cicids2017"]["num_clients"],
        seed=config["project"]["seed"],
    )

    logger.info("Done. Summary:")
    logger.info(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
