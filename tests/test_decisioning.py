from __future__ import annotations

import numpy as np

from src.decisioning import (
    decisions,
    portfolio_notional_exposure,
    risk_tier,
    simple_expected_loss_per_approved,
    threshold_sweep,
)


def test_risk_tier_mapping() -> None:
    p_good = np.array([0.2, 0.5, 0.69, 0.7, 0.95])
    got = risk_tier(p_good, prime_cut=0.70, near_cut=0.50).tolist()
    assert got == ["subprime", "near-prime", "near-prime", "prime", "prime"]


def test_decision_mapping() -> None:
    p_good = np.array([0.40, 0.50, 0.64, 0.65, 0.91])
    got = decisions(p_good, approve_min=0.65, review_min=0.50).tolist()
    assert got == ["decline", "review", "review", "approve", "approve"]


def test_threshold_sweep_outputs_rates() -> None:
    p_good = np.array([0.9, 0.8, 0.6, 0.4])
    y_true = np.array([1, 1, 0, 0])
    thresholds = np.array([0.5, 0.7, 0.95])
    sim = threshold_sweep(p_good, y_true, thresholds)

    assert sim.shape[0] == 3
    # at 0.5: first 3 approved, one bad -> 1/3 default among approved
    row = sim.loc[sim["threshold"] == 0.5].iloc[0]
    assert row["approval_rate"] == 0.75
    assert np.isclose(row["default_rate_among_approved"], 1 / 3)

    # at 0.95: none approved -> nan default among approved
    row = sim.loc[sim["threshold"] == 0.95].iloc[0]
    assert row["approval_rate"] == 0.0
    assert np.isnan(row["default_rate_among_approved"])


def test_simple_capital_helpers() -> None:
    el = simple_expected_loss_per_approved(0.04, avg_loan_amount=15000, lgd=0.45)
    assert np.isclose(el, 270.0)

    exposure = portfolio_notional_exposure(n_approved=120, avg_loan_amount=15000)
    assert exposure == 1_800_000.0
