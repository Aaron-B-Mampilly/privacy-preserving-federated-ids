"""E1-E6/T7 research results.

Every number served here comes straight out of
`experiments/results/e1_e6_t7_tables.json` -- the consolidated file
scripts/build_e1_e6_tables.py produces and the published thesis artifact
is written from. This router reshapes it for display and attaches the
written interpretation; it never recomputes or adjusts a value.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.repository import repository
from api.schemas import ExperimentDetail, ExperimentSummary

router = APIRouter(prefix="/results", tags=["results"])

EXPERIMENTS: dict[str, dict] = {
    "E1": {
        "table_key": "E1_core_comparison",
        "title": "Core comparison",
        "subtitle": "Table T1",
        "table_ref": "T1",
        "description": (
            "Six comparators on both representative scopes: Local-only, Centralized, FedAvg, "
            "FedProx, Base-paper replication, and Ours (no DP) -- measured on accuracy, macro-F1, "
            "rare-class recall, FPR, rounds-to-convergence and MB/round."
        ),
        "interpretation": [
            "CICIDS2017: all six comparators are statistically indistinguishable (macro-F1 0.0831 "
            "everywhere). The model collapses toward predicting BENIGN for ~99.998% of traffic "
            "regardless of mechanism, so there is no headroom for any FL variant to move.",
            "N-BaIoT: a real spread opens up. Centralized (0.843 macro-F1) leads, then Ours/no-DP "
            "(0.667), FedAvg (0.698 macro-F1 but at an early round-2 checkpoint), Base-paper "
            "replication (0.637) and FedProx (0.658).",
            "Base-paper replication's uniform (not size-weighted) averaging and N=3-round periodic "
            "transfer measurably cost it against Ours -- the two mechanism differences this project "
            "deliberately made from the original paper.",
        ],
    },
    "E2": {
        "table_key": "E2_non_iid_severity",
        "title": "Non-IID severity",
        "subtitle": "Table T2",
        "table_ref": "T2",
        "description": (
            "FedAvg, Base-paper replication and Ours across Dirichlet alpha in {5, 0.5, 0.1} on "
            "CICIDS2017, measuring macro-F1 and rare-class recall per alpha."
        ),
        "interpretation": [
            "Personalization does not help more under heterogeneity here -- it degrades and becomes "
            "far more erratic as non-IID severity increases (per-client macro-F1 std rises from "
            "0.0001 at alpha=5 to 0.0260 at alpha=0.1).",
            "Base-paper replication degrades in the same direction under the same sweep, which is "
            "itself informative: this is extreme non-IID data starving both mechanisms' local "
            "models, not something specific to this project's personalization design.",
            "Rare-class recall is near-zero and extremely noisy for both mechanisms at every "
            "non-trivial alpha; differences between them are within noise, not a systematic edge.",
        ],
    },
    "E3": {
        "table_key": "E3_privacy_utility_curve",
        "title": "Privacy-utility curve",
        "subtitle": "Table T3 + Figure F1 - headline experiment",
        "table_ref": "T3",
        "description": (
            "Client-level DP on the shared encoder/decoder across epsilon in {inf, 8, 3, 1, 0.5}, "
            "at alpha=0.5 and alpha=0.1, measuring epsilon vs macro-F1 and epsilon vs rare-class recall."
        ),
        "interpretation": [
            "Macro-F1 degrades gently under DP, consistent with the Phase 8 pattern.",
            "Rare-class recall does not degrade gently -- it is wiped to exactly 0.000 at every "
            "finite epsilon tested, at both alpha values. Even the loosest budget (epsilon=8) is "
            "already enough to eliminate it.",
            "That makes DP's real cost here fall almost entirely on the classes an intrusion "
            "detector exists to catch -- a sharper finding than the aggregate macro-F1 curve alone.",
        ],
    },
    "E4": {
        "table_key": "E4_zero_day_under_dp",
        "title": "Zero-day detection under DP",
        "subtitle": "Table T4",
        "table_ref": "T4",
        "description": (
            "Three metrics swept across epsilon: (a) unknown-vs-benign detection rate, (b) unknown "
            "correctly flagged NEW CLASS rather than assigned to a known attack, and (c) known-class "
            "F1 unchanged. Mechanism: latent class prototypes with a distance threshold."
        ),
        "interpretation": [
            "(a) is saturated at 1.000 in every cell, DP or not -- on this data virtually nothing a "
            "novel class produces lands close enough to the BENIGN prototype to be missed.",
            "(b) is the hard test, and DP breaks it in two opposite ways: on CICIDS2017 any finite "
            "epsilon makes prototypes so noisy that everything is flagged NEW CLASS (1.000, "
            "uninformative); on N-BaIoT it instead collapses toward 0 as epsilon tightens.",
            "(c) is unchanged by construction: prototypes are a post-hoc, non-invasive computation "
            "on an already-trained checkpoint, so the classifier's own F1 is exactly E3's number.",
        ],
    },
    "E5": {
        "table_key": "E5_drift_and_recovery",
        "title": "Concept drift and recovery",
        "subtitle": "Table T5",
        "table_ref": "T5",
        "description": (
            "F1 before drift, F1 after drift without adaptation, drift detection delay in rounds, "
            "and F1 after triggered retraining -- with and without the drift monitor."
        ),
        "interpretation": [
            "N-BaIoT: on the clients that had data before the trigger, F1 falls 0.673 -> 0.577 with "
            "no adaptation, and retraining recovers most of it to 0.636. Detect, retrain, recover -- "
            "the mechanism working as designed.",
            "Those three numbers must be read on a MATCHED client subset. Only 8 of 45 clients had "
            "any pre-cutoff data to retrain on; comparing the retrain arm's 8-client mean against "
            "the other arms' 45-client means made retraining look harmful, which was a pure "
            "client-mismatch artifact.",
            "CICIDS2017 shows a drop (0.167 -> 0.083) that retraining does not move, consistent "
            "with its majority-collapse: there is no class discrimination to lose or recover.",
        ],
    },
    "E6": {
        "table_key": "E6_empirical_privacy",
        "title": "Empirical privacy evaluation",
        "subtitle": "Table T6",
        "table_ref": "T6",
        "description": (
            "Gradient inversion and membership inference run against FedAvg updates, Ours+DP and "
            "Ours+DP+SecAgg, measuring reconstruction quality and MIA AUC. CICIDS2017 alpha=5."
        ),
        "interpretation": [
            "MIA: adding SecAgg+ on top of DP gives essentially no extra membership-inference "
            "protection (0.605 vs 0.606 AUC) -- the attack exploits each client's own train/test "
            "loss asymmetry, which server-side secrecy does not touch.",
            "Gradient inversion: the unprotected FedAvg surface reconstructs real client data almost "
            "exactly (MSE~1.2). Ours+DP never converges across 5 attempts -- a total optimization "
            "failure, not a bounded-but-imperfect reconstruction. Ours+DP+SecAgg reconstructs ~125x "
            "worse (MSE~149) when it does converge.",
            "Divergence is reported, not hidden: 1 of 5 fully-unprotected FedAvg attacks also "
            "diverged, so L-BFGS gradient matching is unstable on this LSTM in general.",
        ],
    },
    "T7": {
        "table_key": "T7_ablation",
        "title": "Ablation table",
        "subtitle": "Incremental module contribution",
        "table_ref": "T7",
        "description": (
            "One row per module added: FedAvg -> +Personalized layers -> +Prototypes -> +DP -> "
            "+SecAgg -> +Drift monitor, each reporting macro-F1, rare recall, zero-day (b), epsilon "
            "and MB/round. CICIDS2017 alpha=5."
        ),
        "interpretation": [
            "Macro-F1 stays essentially flat across all six rows (0.0817-0.0831) -- the CICIDS2017 "
            "majority-collapse means no module costs real classification performance, because there "
            "was never real classification performance at stake on this scope.",
            "The signal is entirely in the zero-day column: 0.026 unprotected -> 1.000 under DP "
            "alone (a false-positive explosion, not detection) -> 0.124 with SecAgg+ -> 0.196 after "
            "the drift-monitor retrain.",
            "MB/round for the last three rows undercounts SecAgg+'s cryptographic share overhead; "
            "only the exchanged parameters' own shapes are counted (a documented simplification).",
        ],
    },
}


@router.get("", response_model=list[ExperimentSummary])
def list_experiments() -> list[ExperimentSummary]:
    tables = repository.load_consolidated_tables() or {}
    return [
        ExperimentSummary(
            id=exp_id,
            title=meta["title"],
            subtitle=meta["subtitle"],
            table_ref=meta["table_ref"],
            available=meta["table_key"] in tables,
        )
        for exp_id, meta in EXPERIMENTS.items()
    ]


@router.get("/{experiment}", response_model=ExperimentDetail)
def get_experiment(experiment: str) -> ExperimentDetail:
    exp_id = experiment.upper()
    meta = EXPERIMENTS.get(exp_id)
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown experiment {experiment!r}; valid: {', '.join(EXPERIMENTS)}",
        )

    tables = repository.load_consolidated_tables()
    if tables is None:
        raise HTTPException(
            status_code=503,
            detail="Consolidated results not found. Run scripts/build_e1_e6_tables.py to generate them.",
        )

    data = tables.get(meta["table_key"])
    if data is None:
        raise HTTPException(status_code=404, detail=f"{exp_id} is not present in the consolidated tables")

    scopes_covered = sorted(data.keys()) if isinstance(data, dict) else []

    return ExperimentDetail(
        id=exp_id,
        title=meta["title"],
        subtitle=meta["subtitle"],
        description=meta["description"],
        interpretation=meta["interpretation"],
        data=data,
        scopes_covered=scopes_covered,
        provenance_note=(
            "Values are read verbatim from experiments/results/e1_e6_t7_tables.json, the same file "
            "the published results artifact is written from. Cells marked pending were never "
            "measured for that scope and are shown as such, never as zero."
        ),
    )
