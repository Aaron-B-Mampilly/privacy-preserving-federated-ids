"""Shared fixtures for CICIDS2017 and N-BaIoT pipeline tests.

Generates small synthetic datasets shaped like the real raw files for
each federation (same column names and known quirks) and runs the
real preprocessing pipeline on them once per test session. This lets
the whole suite run in seconds without touching the real ~1GB
CICIDS2017 / ~7M-row N-BaIoT datasets, while still exercising the
exact code path used on real data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedpda_ids.data.cicids2017 import preprocess_cicids2017
from fedpda_ids.data.nbaiot import preprocess_nbaiot

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


# ---------------------------------------------------------------------
# N-BaIoT
# ---------------------------------------------------------------------

NBAIOT_FEATURE_COLS = [
    "MI_dir_L5_weight",
    "MI_dir_L5_mean",
    "MI_dir_L5_variance",
    "H_L5_weight",
    "H_L5_mean",
    "HH_L5_weight",
    "HH_L5_mean",
    "HH_L5_std",
    "HH_jit_L5_weight",
    "HH_jit_L5_variance",
    "HpHp_L5_weight",
    "HpHp_L5_mean",
]

# device -> has_mirai (Ennio_Doorbell/Samsung_SNH_1011_N_Webcam in the real
# dataset have no Mirai file -- Device_B mirrors that here)
NBAIOT_DEVICES = {
    "Device_A": True,
    "Device_B": False,
    "Device_C": True,
}


def _make_nbaiot_label_csv(rng: np.random.Generator, n_rows: int, huge_value_col: str | None = None) -> pd.DataFrame:
    data = {col: rng.exponential(scale=100, size=n_rows).astype(np.float32) for col in NBAIOT_FEATURE_COLS}
    if huge_value_col is not None:
        # Mimics the real HH_jit_*_variance columns, whose huge magnitude
        # (~1e17) overflows float32 in skew()'s cubing step -- regression
        # coverage for the select_log1p_columns float64 fix.
        data[huge_value_col] = rng.exponential(scale=1e15, size=n_rows).astype(np.float32)
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def synthetic_nbaiot_raw_dir(tmp_path_factory) -> Path:
    raw_dir = tmp_path_factory.mktemp("nbaiot_raw")
    rng = np.random.default_rng(1)

    for device, has_mirai in NBAIOT_DEVICES.items():
        device_dir = raw_dir / device
        device_dir.mkdir()

        _make_nbaiot_label_csv(rng, 200).to_csv(device_dir / "benign_traffic.csv", index=False)

        gafgyt_dir = device_dir / "gafgyt_attacks"
        gafgyt_dir.mkdir()
        for attack in ["combo", "junk", "scan", "tcp", "udp"]:
            _make_nbaiot_label_csv(rng, 150, huge_value_col="HH_jit_L5_variance").to_csv(
                gafgyt_dir / f"{attack}.csv", index=False
            )

        if has_mirai:
            mirai_dir = device_dir / "mirai_attacks"
            mirai_dir.mkdir()
            for attack in ["ack", "scan", "syn", "udp", "udpplain"]:
                _make_nbaiot_label_csv(rng, 150, huge_value_col="HH_jit_L5_variance").to_csv(
                    mirai_dir / f"{attack}.csv", index=False
                )

    return raw_dir


@pytest.fixture(scope="session")
def processed_nbaiot(tmp_path_factory, synthetic_nbaiot_raw_dir) -> tuple[pd.DataFrame, dict]:
    """Runs the real preprocess_nbaiot() pipeline on synthetic data once
    per test session and returns (dataframe, metadata)."""
    processed_dir = tmp_path_factory.mktemp("nbaiot_processed")

    metadata = preprocess_nbaiot(
        raw_dir=synthetic_nbaiot_raw_dir,
        processed_dir=processed_dir,
        devices=list(NBAIOT_DEVICES.keys()),
        zero_day_holdout_labels=["Mirai-Udpplain"],
        train_fraction=0.70,
        val_fraction=0.15,
        log1p_skew_threshold=1.0,
        shards_per_device=5,
        seed=42,
    )
    df = pd.read_parquet(processed_dir / "full_processed.parquet")
    return df, metadata
