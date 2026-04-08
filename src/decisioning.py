"""
Decision layer and portfolio-style simulation on top of existing classifier outputs.

No model training here — consumes P(good) and labels only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_tier(p_good: np.ndarray, prime_cut: float = 0.70, near_cut: float = 0.50) -> np.ndarray:
    """Map P(Fully Paid) to tiers: prime >= prime_cut, near-prime [near_cut, prime_cut), else subprime."""
    p = np.asarray(p_good, dtype=float)
    tiers = np.full(p.shape, "subprime", dtype=object)
    tiers[(p >= near_cut) & (p < prime_cut)] = "near-prime"
    tiers[p >= prime_cut] = "prime"
    return tiers


def decisions(
    p_good: np.ndarray,
    approve_min: float = 0.65,
    review_min: float = 0.50,
) -> np.ndarray:
    """approve / review / decline from P(good), higher is better."""
    p = np.asarray(p_good, dtype=float)
    out = np.full(p.shape, "decline", dtype=object)
    out[(p >= review_min) & (p < approve_min)] = "review"
    out[p >= approve_min] = "approve"
    return out


def threshold_sweep(
    p_good: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """
    y_true: 1 = good (Fully Paid), 0 = bad (Charged Off).
    For each approval threshold on p_good, compute approval rate and default rate among approved.
    """
    p = np.asarray(p_good, dtype=float)
    y = np.asarray(y_true, dtype=int)
    rows = []
    for t in thresholds:
        approved = p >= t
        n = approved.sum()
        approval_rate = approved.mean()
        if n == 0:
            default_rate_approved = np.nan
        else:
            default_rate_approved = (y[approved] == 0).mean()
        rows.append(
            {
                "threshold": t,
                "approval_rate": approval_rate,
                "default_rate_among_approved": default_rate_approved,
                "n_approved": int(n),
            }
        )
    return pd.DataFrame(rows)


def simple_expected_loss_per_approved(
    default_rate_among_approved: float,
    avg_loan_amount: float,
    lgd: float,
) -> float:
    """Directional expected loss per approved loan (single-period, illustrative)."""
    if np.isnan(default_rate_among_approved):
        return float("nan")
    return float(default_rate_among_approved * lgd * avg_loan_amount)


def portfolio_notional_exposure(n_approved: int, avg_loan_amount: float) -> float:
    return float(n_approved * avg_loan_amount)
