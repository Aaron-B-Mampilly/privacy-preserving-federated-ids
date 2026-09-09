"""CLI entry point for Phase 3: N-BaIoT sequence construction.

Builds sequences for the two client schemes:
  - "main_45": group by (device, Label, shard_id, temporal_split);
    client_id_45 is already a deterministic function of device+shard_id,
    so no extra grouping column is needed to respect client boundaries.
  - "ablation_9": group by (device, Label, temporal_split) -- shard_id
    is dropped from the group key, since in this scheme a "client" is
    the whole device and shard boundaries don't apply (this is what
    makes the 9-client ablation have fewer, larger clients).

Usage:
    python scripts/build_sequences_nbaiot.py                  # both schemes
    python scripts/build_sequences_nbaiot.py --schemes main_45 # one only
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fedpda_ids.data.sequences import build_sequences  # noqa: E402
from fedpda_ids.utils.config import load_config  # noqa: E402
from fedpda_ids.utils.logging_utils import setup_logging  # noqa: E402
from fedpda_ids.utils.seed import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build N-BaIoT sequences (Phase 3)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--schemes", nargs="+", default=["main_45", "ablation_9"], choices=["main_45", "ablation_9"]
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    logger = setup_logging(
        log_dir=config["logging"]["log_dir"],
        level=config["logging"]["level"],
        log_to_file=config["logging"]["log_to_file"],
        run_name="build_sequences_nbaiot",
    )

    data_cfg = config["data"]
    nbaiot_cfg = data_cfg["nbaiot"]
    seq_cfg = config["sequence"]

    processed_dir = Path(nbaiot_cfg["processed_dir"])
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    feature_cols = metadata["feature_columns"]

    schemes = {
        "main_45": {
            "base_group_cols": ["device", "Label", "shard_id"],
            "client_id_col": "client_id_45",
        },
        "ablation_9": {
            "base_group_cols": ["device", "Label"],
            "client_id_col": "client_id_9",
        },
    }

    needed_cols = [
        *feature_cols,
        "device", "Label", "shard_id", "row_order",
        "temporal_split", "client_id_45", "client_id_9",
    ]
    logger.info("Loading full_processed.parquet")
    df = pd.read_parquet(processed_dir / "full_processed.parquet", columns=list(dict.fromkeys(needed_cols)))

    for scheme_name in args.schemes:
        scheme = schemes[scheme_name]
        output_dir = processed_dir / "sequences" / scheme_name
        logger.info("Building sequences for scheme=%s -> %s", scheme_name, output_dir)

        summary = build_sequences(
            df,
            feature_cols=feature_cols,
            base_group_cols=scheme["base_group_cols"],
            time_col="row_order",
            label_col="Label",
            client_id_col=scheme["client_id_col"],
            window_size=seq_cfg["window_size"],
            stride=seq_cfg["stride"],
            output_dir=output_dir,
            # float16: halves disk usage (~6.5GB -> ~3.25GB per scheme).
            # This machine's C: drive hit 96%+ full during Phase 3 --
            # see sequences.py's build_sequences docstring for why this
            # is safe (values are already MinMax-scaled to ~[0,1]).
            storage_dtype=np.float16,
        )
        logger.info("%s summary:\n%s", scheme_name, json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
