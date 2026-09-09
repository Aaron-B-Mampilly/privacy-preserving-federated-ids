"""CLI entry point for Phase 3: CICIDS2017 sequence construction.

Builds sequences separately for each Dirichlet alpha value, since
each alpha is a distinct client-assignment scheme (a row's client_id
column differs per alpha) -- see Phase 3 Part A/B for why grouping
must include the active client-id column, not just the host.

Usage:
    python scripts/build_sequences_cicids2017.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from fedpda_ids.data.sequences import build_sequences  # noqa: E402
from fedpda_ids.utils.config import load_config  # noqa: E402
from fedpda_ids.utils.logging_utils import setup_logging  # noqa: E402
from fedpda_ids.utils.seed import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CICIDS2017 sequences (Phase 3)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"],
        level=config["logging"]["level"],
        log_to_file=config["logging"]["log_to_file"],
        run_name="build_sequences_cicids2017",
    )

    data_cfg = config["data"]
    cic_cfg = data_cfg["cicids2017"]
    seq_cfg = config["sequence"]
    alpha_values = config["federated"]["cicids2017"]["dirichlet_alpha_values"]

    processed_dir = Path(cic_cfg["processed_dir"])
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    feature_cols = metadata["feature_columns"]

    logger.info("Loading full_processed.parquet")
    needed_cols = [
        *feature_cols,
        *cic_cfg["sequence_group_columns"],
        cic_cfg["sequence_time_column"],
        "Label",
        "temporal_split",
        *metadata["client_id_columns"],
    ]
    df = pd.read_parquet(processed_dir / "full_processed.parquet", columns=list(dict.fromkeys(needed_cols)))

    for alpha in alpha_values:
        client_id_col = f"client_id_alpha{alpha}"
        output_dir = processed_dir / "sequences" / f"alpha_{alpha}"
        logger.info("Building sequences for alpha=%s -> %s", alpha, output_dir)

        summary = build_sequences(
            df,
            feature_cols=feature_cols,
            base_group_cols=cic_cfg["sequence_group_columns"],
            time_col=cic_cfg["sequence_time_column"],
            label_col="Label",
            client_id_col=client_id_col,
            window_size=seq_cfg["window_size"],
            stride=seq_cfg["stride"],
            output_dir=output_dir,
        )
        logger.info("alpha=%s summary:\n%s", alpha, json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
