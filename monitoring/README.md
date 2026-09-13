# Phase 13: Prometheus + Grafana monitoring

Live observability for FL training runs -- round number, centralized
validation loss/MSE, and (for Phase 8 DP runs) the adaptive clip norm.

## Usage

1. Start the stack (from this directory):
   ```
   docker compose up -d
   ```
2. Start a training run with monitoring enabled, e.g.:
   ```
   python scripts/train_personalized.py --dataset cicids2017 --alpha 5 --enable-monitoring
   ```
   This starts a Prometheus metrics HTTP server on `localhost:8000` (the
   training script itself, not a container) that Prometheus scrapes
   every 5s.
3. Open Grafana at http://localhost:3000 (user: `admin`, password:
   `admin` -- change on first login) and open the **FedPDA-IDS
   Training** dashboard. Use the `run_name` dropdown at the top to pick
   which run to view.
4. Prometheus's own UI is at http://localhost:9090 -- check
   **Status > Targets** to confirm the `fedpda_ids_training` scrape
   target is `UP` if the dashboard shows no data.

## Notes

- Monitoring is opt-in per run (`--enable-monitoring`, default off) --
  it never affects training/aggregation logic, only records metrics as
  a side effect. See `src/fedpda_ids/monitoring/metrics_exporter.py`.
- Only ONE training run's metrics server can bind port 8000 at a time
  on this machine; use `--metrics-port` to run more than one
  simultaneously (and add the extra port to `prometheus/prometheus.yml`'s
  `targets` list).
- `fedpda_dp_clip_norm` is only ever populated by Phase 8 DP runs
  (`train_dp_personalized.py --enable-monitoring`) -- empty for
  FedAvg/Personalized/SecAgg+ runs, by design.
- Grafana/Prometheus data lives in a named Docker volume
  (`grafana_data`) and Prometheus's default in-container storage --
  stop the stack with `docker compose down` (add `-v` to also wipe
  historical data).
