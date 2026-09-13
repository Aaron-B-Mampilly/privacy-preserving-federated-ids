"""Phase 13: Prometheus metrics exporter tests -- verifies recorded
values land in the global registry correctly, using prometheus_client's
own recommended test pattern (REGISTRY.get_sample_value), and that the
HTTP server is never started implicitly by anything but
start_metrics_server()."""

from prometheus_client import REGISTRY

from fedpda_ids.monitoring.metrics_exporter import (
    _server_started,
    record_dp_clip_norm,
    record_round,
)


def test_record_round_sets_round_and_loss_gauges():
    record_round("test_run_a", 5, val_loss=0.1234)
    assert REGISTRY.get_sample_value("fedpda_fl_round", {"run_name": "test_run_a"}) == 5.0
    assert REGISTRY.get_sample_value("fedpda_centralized_val_loss", {"run_name": "test_run_a"}) == 0.1234


def test_record_round_without_val_loss_only_sets_round():
    record_round("test_run_b", 3)
    assert REGISTRY.get_sample_value("fedpda_fl_round", {"run_name": "test_run_b"}) == 3.0
    # never set for this run -- get_sample_value returns None for an unset label combination
    assert REGISTRY.get_sample_value("fedpda_centralized_val_loss", {"run_name": "test_run_b"}) is None


def test_record_round_updates_overwrite_previous_value():
    record_round("test_run_c", 1, val_loss=1.0)
    record_round("test_run_c", 2, val_loss=0.5)
    assert REGISTRY.get_sample_value("fedpda_fl_round", {"run_name": "test_run_c"}) == 2.0
    assert REGISTRY.get_sample_value("fedpda_centralized_val_loss", {"run_name": "test_run_c"}) == 0.5


def test_record_dp_clip_norm():
    record_dp_clip_norm("test_run_dp", 7.25)
    assert REGISTRY.get_sample_value("fedpda_dp_clip_norm", {"run_name": "test_run_dp"}) == 7.25


def test_different_run_names_are_independent():
    record_round("run_x", 10, val_loss=0.9)
    record_round("run_y", 20, val_loss=0.1)
    assert REGISTRY.get_sample_value("fedpda_fl_round", {"run_name": "run_x"}) == 10.0
    assert REGISTRY.get_sample_value("fedpda_fl_round", {"run_name": "run_y"}) == 20.0


def test_module_import_never_starts_the_http_server():
    # importing/using record_* functions above must not have started the
    # server as a side effect -- only start_metrics_server() may do that.
    assert _server_started is False
