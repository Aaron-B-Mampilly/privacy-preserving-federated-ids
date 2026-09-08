"""Shared fixtures for CICIDS2017 pipeline tests.

Generates a small synthetic dataset shaped like the real 8 CICIDS2017
weekday CSVs (same column names, same known quirks: mixed timestamp
formats, a mangled attack-label encoding, a duplicate "Fwd Header
Length" column, and an injected Inf value) and runs the real
preprocessing pipeline on it once per test session. This lets the
whole suite run in seconds without touching the ~1GB real dataset,
while still exercising the exact code path used on real data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedpda_ids.data.cicids2017 import preprocess_cicids2017

FEATURE_COLS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Fwd PSH Flags",
    "SYN Flag Count",
    "ACK Flag Count",
    "Average Packet Size",
    "Init_Win_bytes_forward",
    "Active Mean",
]

# day index -> (filename, weekday, labels present that day)
DAY_FILES = {
    "Monday-WorkingHours.pcap_ISCX.csv": ("Monday", ["BENIGN"]),
    "Tuesday-WorkingHours.pcap_ISCX.csv": ("Tuesday", ["BENIGN", "FTP-Patator", "SSH-Patator"]),
    "Wednesday-workingHours.pcap_ISCX.csv": ("Wednesday", ["BENIGN", "DoS Hulk", "Heartbleed"]),
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": (
        "Thursday",
        ["BENIGN", "Web Attack \x96 Brute Force", "Web Attack \x96 XSS"],
    ),
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": (
        "Thursday",
        ["BENIGN", "Infiltration"],
    ),
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": ("Friday", ["BENIGN", "Bot"]),
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": ("Friday", ["BENIGN", "PortScan"]),
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": ("Friday", ["BENIGN", "DDoS"]),
}

HOSTS = [f"192.168.10.{i}" for i in range(1, 9)]


def _make_synthetic_day_file(rng: np.random.Generator, day_idx: int, labels: list[str], n_rows: int = 400) -> pd.DataFrame:
    base_time = pd.Timestamp("2017-07-03") + pd.Timedelta(days=day_idx)
    rows = []
    for i in range(n_rows):
        host = rng.choice(HOSTS)
        if len(labels) > 1:
            probs = [0.7] + [0.3 / (len(labels) - 1)] * (len(labels) - 1)
            label = rng.choice(labels, p=probs)
        else:
            label = labels[0]
        ts = base_time + pd.Timedelta(minutes=i)
        if day_idx == 0:
            # Monday-style: zero-padded, with seconds -- e.g. "03/07/2017 08:55:58"
            timestamp_str = ts.strftime("%d/%m/%Y %H:%M:%S")
        else:
            # Tue-Fri style: no zero-padding, no seconds -- e.g. "7/7/2017 3:30"
            # (built manually -- %-d/%-H are glibc-only strftime extensions
            # and are not portable to Windows Python).
            timestamp_str = f"{ts.day}/{ts.month}/{ts.year} {ts.hour}:{ts.minute:02d}"
        row = {
            "Flow ID": f"{host}-{i}",
            "Source IP": host,
            "Source Port": rng.integers(1024, 65535),
            "Destination IP": f"8.8.8.{rng.integers(1, 255)}",
            "Destination Port": int(rng.choice([80, 443, 22, 21, 8080, 60000])),
            "Timestamp": timestamp_str,
            "Label": label,
        }
        for col in FEATURE_COLS:
            row[col] = float(rng.exponential(scale=100))
        if i == 5:
            row["Flow Bytes/s"] = np.inf
        rows.append(row)
    df = pd.DataFrame(rows)
    df["Fwd Header Length.1"] = df["Total Fwd Packets"]
    df["Fwd Header Length"] = df["Total Fwd Packets"]
    return df


@pytest.fixture(scope="session")
def synthetic_cicids_raw_dir(tmp_path_factory) -> Path:
    raw_dir = tmp_path_factory.mktemp("cicids_raw")
    rng = np.random.default_rng(0)
    for day_idx, (filename, (_, labels)) in enumerate(DAY_FILES.items()):
        df = _make_synthetic_day_file(rng, day_idx, labels)
        df.to_csv(raw_dir / filename, index=False, encoding="latin1")
    return raw_dir


@pytest.fixture(scope="session")
def processed_cicids(tmp_path_factory, synthetic_cicids_raw_dir) -> tuple[pd.DataFrame, dict]:
    """Runs the real preprocess_cicids2017() pipeline on the synthetic
    data once per test session and returns (dataframe, metadata)."""
    processed_dir = tmp_path_factory.mktemp("cicids_processed")
    file_to_day = {fname: day for fname, (day, _) in DAY_FILES.items()}

    metadata = preprocess_cicids2017(
        raw_dir=synthetic_cicids_raw_dir,
        processed_dir=processed_dir,
        file_to_day=file_to_day,
        zero_day_holdout_labels=["Infiltration", "Web Attack-Brute Force"],
        train_fraction=0.70,
        val_fraction=0.15,
        log1p_skew_threshold=1.0,
        dirichlet_alpha_values=[5, 0.5, 0.1],
        num_clients=40,
        seed=42,
    )
    df = pd.read_parquet(processed_dir / "full_processed.parquet")
    return df, metadata
