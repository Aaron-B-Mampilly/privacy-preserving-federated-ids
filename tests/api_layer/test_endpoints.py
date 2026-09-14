"""API contract tests.

These run against the project's REAL results directory, because the
contract that matters is "the API serves the same numbers the thesis
tables do" -- a mocked repository could not catch a drift between them.
Tests that need an artifact which may not be present on a given machine
skip rather than fail, so a fresh clone without checkpoints still gets a
meaningful signal from the rest of the suite.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.repository import repository

client = TestClient(app)


@pytest.fixture(scope="module")
def tables():
    data = repository.load_consolidated_tables()
    if data is None:
        pytest.skip("consolidated E1-E6/T7 tables not generated on this machine")
    return data


# ---------------------------------------------------------------- system


def test_health_reports_inventory():
    body = client.get("/api/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["results_available"] >= 0
    assert set(body["datasets"]) == {"cicids2017", "nbaiot"}
    # Grafana stays the observability layer -- the API links to it, never replaces it.
    assert body["grafana_url"].endswith(":3000")


def test_datasets_are_never_merged():
    body = client.get("/api/datasets").json()
    ids = [d["id"] for d in body]
    assert ids == ["cicids2017", "nbaiot"]
    # There must be no combined/"all" pseudo-dataset: merging the two
    # federations is a standing research-integrity rule for this project.
    assert not any(d["id"] in {"all", "combined", "merged"} for d in body)
    assert all(d["scopes"] for d in body)


def test_unknown_dataset_is_404_not_a_silent_fallback():
    response = client.get("/api/dashboard?dataset=not_a_dataset")
    assert response.status_code == 404
    assert "not_a_dataset" in response.json()["detail"]


# ---------------------------------------------------------------- results


def test_results_index_lists_all_seven_experiments(tables):
    body = client.get("/api/results").json()
    assert [e["id"] for e in body] == ["E1", "E2", "E3", "E4", "E5", "E6", "T7"]
    assert all(e["available"] for e in body)


def test_experiment_payload_matches_the_consolidated_tables_exactly(tables):
    """The API must not reshape or round a published number."""
    body = client.get("/api/results/E1").json()
    assert body["data"] == tables["E1_core_comparison"]
    assert body["interpretation"]


def test_unknown_experiment_is_rejected():
    response = client.get("/api/results/E42")
    assert response.status_code == 404
    assert "valid" in response.json()["detail"]


def test_pending_cells_are_preserved_not_zero_filled(tables):
    """A cell the spec never measured must stay an explicit pending
    marker end-to-end -- silently rendering it as 0 would fabricate a
    result."""
    body = client.get("/api/results/E6").json()
    gi = body["data"]["cicids2017_alpha5"]["gradient_inversion"]["ours_dp"]
    # Ours+DP diverged in every run, so the mean is null and the
    # divergence is reported separately rather than averaged in.
    assert gi["reconstruction_mse"] is None
    assert gi["diverged"] is True
    assert gi["num_diverged_runs"] == gi["num_total_runs"]


# -------------------------------------------------------------- dashboard


def test_dashboard_kpis_match_the_source_run():
    body = client.get("/api/dashboard?dataset=cicids2017&scope=5").json()
    assert body["source_run"] == "cicids2017_5_personalized"

    run = repository.load("cicids2017_5_personalized")
    expected = run["per_client_summary"]["macro_f1_mean"]
    assert body["kpis"]["macro_f1"]["mean"] == pytest.approx(expected, rel=1e-9)

    # Every KPI carries provenance so the UI can label stored vs derived.
    assert all("provenance" in kpi for kpi in body["kpis"].values())


def test_dashboard_series_are_round_indexed():
    body = client.get("/api/dashboard?dataset=cicids2017&scope=5").json()
    macro = next(s for s in body["series"] if s["key"] == "macro_f1")
    assert len(macro["points"]) > 1
    rounds = [p["round"] for p in macro["points"]]
    assert rounds == sorted(rounds)


# --------------------------------------------------------------------- fl


def test_fl_runs_only_lists_mechanisms_that_actually_completed():
    body = client.get("/api/fl/runs?dataset=cicids2017&scope=5").json()
    mechanisms = {r["mechanism"] for r in body}
    assert {"fedavg", "personalized"} <= mechanisms
    for run in body:
        assert repository.exists(run["run_name"])


def test_fl_run_detail_exposes_no_raw_client_data():
    body = client.get("/api/fl/runs/cicids2017_5_personalized").json()
    assert body["clients"]
    allowed = {"client_id", "status", "participated", "num_samples", "accuracy", "macro_f1"}
    for entry in body["clients"]:
        assert set(entry) <= allowed


def test_unknown_run_is_404():
    assert client.get("/api/fl/runs/does_not_exist").status_code == 404


# ---------------------------------------------------------------- privacy


def test_privacy_utility_curve_covers_the_full_epsilon_sweep():
    body = client.get("/api/privacy/utility?dataset=cicids2017&scope=0.5").json()
    assert [p["epsilon"] for p in body["curve"]] == ["inf", "8", "3", "1", "0.5"]


def test_rare_recall_zero_under_dp_is_reported_as_measured_zero():
    """0.000 here is a real finding (DP eliminates rare-class recall),
    so it must arrive as a measured 0.0, not as a missing value."""
    body = client.get("/api/privacy/utility?dataset=cicids2017&scope=0.5").json()
    eps8 = next(p for p in body["curve"] if p["epsilon"] == "8")
    assert eps8["rare_class_recall"]["mean"] == 0.0
    assert eps8["rare_class_recall"]["n"] > 0


def test_attack_scope_limitation_is_stated_not_hidden():
    """E6 was only run for CICIDS2017 alpha=5; other scopes must say so
    rather than presenting another scope's attack numbers."""
    body = client.get("/api/privacy?dataset=nbaiot").json()
    assert body["attacks"]["available"] is False
    assert "alpha=5" in body["attacks"]["reason"]


# ------------------------------------------------------------------ drift


def test_drift_status_reports_the_matched_subset_comparison():
    """The matched subset is what makes the retrain comparison valid --
    without it the arms describe different client sets."""
    body = client.get("/api/drift/status?dataset=nbaiot").json()
    matched = body["recovery"]["matched_subset"]
    assert matched["num_clients_matched"] == 8
    assert matched["num_clients_total"] == 45
    assert matched["f1_after_triggered_retraining"]["mean"] > matched["f1_after_drift_no_adaptation"]["mean"]


def test_drift_events_form_an_ordered_timeline():
    body = client.get("/api/drift/events?dataset=cicids2017&scope=5").json()
    stages = [e["stage"] for e in body["events"]]
    assert stages[0] == "normal"
    assert "drift_detected" in stages


def test_retrain_rejects_an_unknown_scope():
    response = client.post("/api/drift/retrain", json={"dataset": "cicids2017", "scope": "9.9"})
    assert response.status_code == 404


def test_retrain_job_lookup_is_404_for_unknown_id():
    assert client.get("/api/drift/retrain/deadbeef").status_code == 404


# -------------------------------------------------------------- detection


def test_samples_returns_metadata_only():
    response = client.get("/api/samples?dataset=cicids2017&scope=5&limit=5")
    if response.status_code == 404:
        pytest.skip("sequence artifacts not built on this machine")
    body = response.json()
    assert body["total"] > 0
    for sample in body["samples"]:
        # Feature values must never leave the server in the sample list.
        assert set(sample) == {"sequence_index", "label", "client_id", "split"}


def test_predict_on_a_prepared_sample_returns_a_real_forward_pass():
    listing = client.get("/api/samples?dataset=cicids2017&scope=5&limit=1")
    if listing.status_code == 404:
        pytest.skip("sequence artifacts not built on this machine")
    sample = listing.json()["samples"][0]

    response = client.post("/api/predict", json={
        "dataset": "cicids2017", "scope": "5",
        "sequence_index": sample["sequence_index"], "client_id": sample["client_id"],
    })
    if response.status_code == 404:
        pytest.skip("checkpoints not available on this machine")

    body = response.json()
    assert body["provenance"] == "live"
    assert body["true_label"] == sample["label"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert sum(t["probability"] for t in body["top_k"]) <= 1.0 + 1e-6
    assert body["reconstruction_mse"] >= 0.0


def test_predict_rejects_a_wrongly_shaped_window():
    response = client.post("/api/predict", json={
        "dataset": "cicids2017", "scope": "5",
        "window": [[0.0] * 70] * 3,  # 3 rows, needs 10
    })
    assert response.status_code == 422
    assert "10x70" in response.json()["detail"]


def test_predict_requires_an_input():
    response = client.post("/api/predict", json={"dataset": "cicids2017", "scope": "5"})
    assert response.status_code == 422


def test_error_responses_never_leak_a_traceback():
    for response in (
        client.get("/api/dashboard?dataset=bogus"),
        client.get("/api/results/E42"),
        client.get("/api/fl/runs/nope"),
    ):
        assert response.status_code in (404, 422)
        body = response.text.lower()
        assert "traceback" not in body
        assert "file \"" not in body
