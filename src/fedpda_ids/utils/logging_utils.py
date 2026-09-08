"""Project-wide logging setup.

Every phase (data prep, training, FL rounds, drift detection, ...)
should log through a logger created here rather than print(), so that
runs leave a persistent, timestamped record in logs/ alongside the
console output. This is part of the reproducibility requirement: a
result should always be traceable back to the run that produced it.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_dir: str | Path = "logs",
    level: str = "INFO",
    log_to_file: bool = True,
    run_name: str | None = None,
) -> logging.Logger:
    logger = logging.getLogger("fedpda_ids")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_to_file:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or "run"
        log_file = log_dir / f"{run_name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "fedpda_ids") -> logging.Logger:
    return logging.getLogger(name)
