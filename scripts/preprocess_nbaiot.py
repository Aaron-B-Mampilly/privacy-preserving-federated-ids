"""CLI entry point for Phase 2 Part C: N-BaIoT preprocessing.

Usage:
    python scripts/preprocess_nbaiot.py
    python scripts/preprocess_nbaiot.py --config configs/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedpda_ids.data.nbaiot import preprocess_nbaiot
from fedpda_ids.utils.config import load_config
from fedpda_ids.utils.logging_utils import setup_logging
from fedpda_ids.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess N-BaIoT (Phase 2 Part C)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"],
        level=config["logging"]["level"],
        log_to_file=config["logging"]["log_to_file"],
        run_name="preprocess_nbaiot",
    )

    data_cfg = config["data"]
    nbaiot_cfg = data_cfg["nbaiot"]

    logger.info("Starting N-BaIoT preprocessing")
    metadata = preprocess_nbaiot(
        raw_dir=nbaiot_cfg["raw_dir"],
        processed_dir=nbaiot_cfg["processed_dir"],
        devices=nbaiot_cfg["devices"],
        zero_day_holdout_labels=nbaiot_cfg["zero_day_holdout_labels"],
        train_fraction=data_cfg["train_fraction"],
        val_fraction=data_cfg["val_fraction"],
        log1p_skew_threshold=data_cfg["log1p_skew_threshold"],
        shards_per_device=config["federated"]["nbaiot"]["shards_per_device"],
        seed=config["project"]["seed"],
    )

    logger.info("Done. Summary:")
    logger.info(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
