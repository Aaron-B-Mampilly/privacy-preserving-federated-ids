"""Intrusion detection: real inference on real checkpoints.

Two honest input paths, no third:
  * a PREPARED SAMPLE from the held-out test split (real data, real
    forward pass, and the sample's true label is returned alongside the
    prediction so a demo can be checked rather than taken on trust), and
  * an UPLOADED CSV matching the frozen preprocessed contract
    (window_size rows x num_features columns).

There is deliberately no hand-entry form: a real input is a 10x70
(CICIDS2017) or 10x115 (N-BaIoT) preprocessed float matrix, and a form
that let a user type 700 numbers would be theatre rather than a feature.
"""

from __future__ import annotations

import csv
import io
import uuid
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from api.config import resolve_scope
from api.registry import registry
from api.repository import repository
from api.routers.health import scope_info
from api.schemas import (
    PredictionResponse,
    PredictRequest,
    Provenance,
    SampleItem,
    SamplesResponse,
)

router = APIRouter(tags=["detection"])

# Bounded history: this is a demo surface, not an audit log.
_history: deque = deque(maxlen=100)

BENIGN_LABELS = {"BENIGN", "benign", "Normal"}


@router.get("/samples", response_model=SamplesResponse)
def samples(
    dataset: str = Query(...),
    scope: str | None = Query(None),
    label: str | None = Query(None, description="filter to one class"),
    client_id: int | None = Query(None),
    split: str = Query("test", description="test | zero_day_holdout"),
    limit: int = Query(40, ge=1, le=200),
) -> SamplesResponse:
    """Prepared real samples the user can run inference on.

    Only metadata is returned -- index, class, client, split. The feature
    values themselves stay server-side and are looked up at predict time.
    """
    try:
        resolved = resolve_scope(dataset, scope)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    meta_path = resolved.sequence_dir / "metadata.parquet"
    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Sequence artifacts not built for {resolved.label}.",
        )

    metadata = pd.read_parquet(
        meta_path,
        columns=["temporal_split", "sequence_label", "sequence_index", resolved.client_id_col],
    )
    subset = metadata[metadata["temporal_split"] == split]
    if split != "zero_day_holdout":
        subset = subset[subset[resolved.client_id_col] != -1]
    if client_id is not None:
        subset = subset[subset[resolved.client_id_col] == client_id]

    labels_available = sorted(subset["sequence_label"].unique().tolist())
    if label:
        subset = subset[subset["sequence_label"] == label]

    total = int(len(subset))
    if total == 0:
        return SamplesResponse(
            scope=scope_info(dataset, scope), total=0, samples=[], labels_available=labels_available
        )

    # Spread the picks across the split so a demo isn't always the same
    # contiguous block of near-identical windows.
    step = max(1, total // limit)
    picked = subset.iloc[::step].head(limit)

    return SamplesResponse(
        scope=scope_info(dataset, scope),
        total=total,
        samples=[
            SampleItem(
                sequence_index=int(row["sequence_index"]),
                label=str(row["sequence_label"]),
                client_id=int(row[resolved.client_id_col]),
                split=split,
            )
            for row in picked.to_dict("records")
        ],
        labels_available=labels_available,
    )


def _threat_level(label: str, confidence: float, is_new_class: bool) -> str:
    if is_new_class:
        return "high"
    if label in BENIGN_LABELS:
        return "benign"
    if confidence >= 0.9:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _run_inference(
    resolved,
    window: np.ndarray,
    mechanism: str,
    client_id: int | None,
    source: str,
    sequence_index: int | None,
    true_label: str | None,
) -> PredictionResponse:
    expected = (registry.config["sequence"]["window_size"], resolved.num_features)
    if window.shape != expected:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected a {expected[0]}x{expected[1]} window for {resolved.label}, "
                f"got {window.shape[0]}x{window.shape[1]}."
            ),
        )

    try:
        loaded = registry.get_model(resolved, mechanism, "best", client_id=client_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    with torch.no_grad():
        tensor = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
        reconstruction, logits, latent = loaded.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        recon_mse = float(((reconstruction - tensor) ** 2).mean())

    order = torch.argsort(probs, descending=True)
    top_k = [
        {"label": loaded.index_to_label[int(i)], "probability": float(probs[int(i)])}
        for i in order[:5]
    ]
    top_index = int(order[0])
    predicted_label = loaded.index_to_label[top_index]
    confidence = float(probs[top_index])

    # Zero-day / NEW CLASS via the prototype mechanism. Unavailable is
    # reported as such rather than silently defaulting to "known".
    is_new_class = False
    distance = threshold = None
    prototype_label = None
    try:
        proto_pred, distance, threshold = registry.classify_new_class(resolved, latent[0].numpy())
        is_new_class = proto_pred == -1
        prototype_label = None if is_new_class else loaded.index_to_label.get(proto_pred)
    except Exception:  # noqa: BLE001 - prototypes are optional enrichment
        pass

    response = PredictionResponse(
        prediction_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc).isoformat(),
        scope=scope_info(resolved.dataset, resolved.scope),
        model_run=loaded.run_name,
        mechanism=mechanism,
        client_id=client_id,
        source=source,
        sequence_index=sequence_index,
        true_label=true_label,
        predicted_label=predicted_label,
        confidence=confidence,
        threat_level=_threat_level(predicted_label, confidence, is_new_class),
        is_attack=predicted_label not in BENIGN_LABELS,
        is_new_class=is_new_class,
        new_class_distance=distance,
        new_class_threshold=threshold,
        prototype_label=prototype_label,
        reconstruction_mse=recon_mse,
        top_k=top_k,
        provenance=Provenance.LIVE,
    )
    _history.appendleft(response)
    return response


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest) -> PredictionResponse:
    try:
        resolved = resolve_scope(request.dataset, request.scope)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if request.sequence_index is None and request.window is None:
        raise HTTPException(
            status_code=422,
            detail="Provide either sequence_index (a prepared sample) or window (a raw matrix).",
        )

    true_label = None
    client_id = request.client_id

    if request.sequence_index is not None:
        try:
            window = registry.sample_window(resolved, request.sequence_index)
        except (IndexError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        metadata = pd.read_parquet(
            resolved.sequence_dir / "metadata.parquet",
            columns=["sequence_index", "sequence_label", resolved.client_id_col],
        )
        row = metadata[metadata["sequence_index"] == request.sequence_index]
        if not row.empty:
            true_label = str(row.iloc[0]["sequence_label"])
            if client_id is None:
                found = int(row.iloc[0][resolved.client_id_col])
                # -1 marks the shared zero-day holdout, which belongs to no client.
                client_id = found if found >= 0 else None
        source = "prepared_sample"
    else:
        window = np.asarray(request.window, dtype=np.float32)
        source = "raw_window"

    # Personalized inference needs a client head; fall back to a client
    # that has one rather than failing when the sample has no owner.
    if request.mechanism == "personalized" and client_id is None:
        run = repository.load_mechanism(resolved, "personalized")
        evaluated = [
            int(cid) for cid, m in (run or {}).get("per_client_test_metrics", {}).items()
            if isinstance(m, dict) and "status" not in m
        ]
        if evaluated:
            client_id = evaluated[0]

    return _run_inference(
        resolved, window, request.mechanism, client_id, source, request.sequence_index, true_label
    )


@router.post("/predict/upload", response_model=PredictionResponse)
async def predict_upload(
    file: UploadFile = File(...),
    dataset: str = Form(...),
    scope: str | None = Form(None),
    mechanism: str = Form("personalized"),
    client_id: int | None = Form(None),
) -> PredictionResponse:
    """CSV upload: window_size rows x num_features columns, already
    preprocessed to match the frozen pipeline (an optional header row is
    accepted and skipped)."""
    try:
        resolved = resolve_scope(dataset, scope)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Please upload a .csv file.")

    raw = await file.read()
    if len(raw) > 2_000_000:
        raise HTTPException(status_code=413, detail="CSV too large (limit 2 MB).")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="CSV must be UTF-8 encoded.") from exc

    rows: list[list[float]] = []
    for line_no, fields in enumerate(csv.reader(io.StringIO(text))):
        if not fields or all(not f.strip() for f in fields):
            continue
        try:
            rows.append([float(f) for f in fields])
        except ValueError as exc:
            if line_no == 0:
                continue  # header row
            raise HTTPException(
                status_code=422,
                detail=f"Row {line_no + 1} contains a non-numeric value. Expected preprocessed floats.",
            ) from exc

    if not rows:
        raise HTTPException(status_code=422, detail="No numeric rows found in the CSV.")

    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise HTTPException(
            status_code=422,
            detail=f"Inconsistent row widths in CSV: {sorted(widths)}. Every row needs the same columns.",
        )

    window = np.asarray(rows, dtype=np.float32)
    expected = (registry.config["sequence"]["window_size"], resolved.num_features)
    if window.shape != expected:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{resolved.label} expects {expected[0]} rows x {expected[1]} features; "
                f"this file is {window.shape[0]} x {window.shape[1]}. "
                "The CSV must already be preprocessed by the project's own pipeline."
            ),
        )

    return _run_inference(resolved, window, mechanism, client_id, "uploaded_csv", None, None)


@router.get("/predictions", response_model=list[PredictionResponse])
def predictions(limit: int = Query(25, ge=1, le=100)) -> list[PredictionResponse]:
    """Recent predictions made through this API process (in-memory)."""
    return list(_history)[:limit]
