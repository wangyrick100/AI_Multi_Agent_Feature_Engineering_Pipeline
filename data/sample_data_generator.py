"""
Sample Data Generator
──────────────────────
Generates a realistic synthetic e-commerce / fintech transaction dataset
suitable for demonstrating all feature engineering categories:

  - Temporal: timestamps with realistic patterns
  - Behavioral: user activity sequences
  - RFM: recency, frequency, monetary amounts
  - Categorical: merchant, category, channel
  - Graph: user-merchant edges for network features
  - Text: memo/description field for semantic features
  - Target: binary fraud label (imbalanced, ~3.5%)

Usage
-----
    from data.sample_data_generator import generate_transactions
    df = generate_transactions(n_users=1000, n_rows=50_000, seed=42)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


CATEGORIES = ["grocery", "restaurant", "travel", "entertainment", "retail",
               "utilities", "healthcare", "education", "tech", "fashion"]
CHANNELS = ["web", "mobile", "pos", "in-store", "atm"]
COUNTRIES = ["US", "UK", "DE", "FR", "CA", "AU", "SG", "JP"]
MEMO_TEMPLATES = [
    "Payment for {cat} services",
    "Online purchase at {merchant}",
    "Subscription renewal - {cat}",
    "ATM withdrawal",
    "Transfer to {merchant}",
    "Refund from {merchant}",
    "International payment {country}",
    "Split bill - {cat}",
]


def generate_transactions(
    n_users: int = 1_000,
    n_rows: int = 50_000,
    seed: int = 42,
    start_date: str = "2022-01-01",
    end_date: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Generate a synthetic transaction DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        transaction_id, user_id, event_timestamp, amount, category,
        merchant_id, merchant_name, channel, country, memo,
        user_age_days, account_tier, is_fraud
    """
    rng = np.random.default_rng(seed)

    # ── User pool ─────────────────────────────────────────────────────────────
    user_ids = [f"U{i:06d}" for i in range(n_users)]

    # User-level risk profile (hidden variable driving fraud)
    user_risk = rng.beta(0.5, 9, n_users)           # ~10% high-risk users
    user_avg_amount = rng.lognormal(3.5, 1.2, n_users)  # log-normal spend
    user_preferred_cat = rng.integers(0, len(CATEGORIES), n_users)
    user_reg_date = pd.to_datetime(start_date) - pd.to_timedelta(
        rng.integers(0, 1800, n_users), unit="D"
    )

    # ── Merchant pool ─────────────────────────────────────────────────────────
    n_merchants = min(n_users // 3, 500)
    merchant_ids = [f"M{i:05d}" for i in range(n_merchants)]
    merchant_names = [f"{CATEGORIES[i % len(CATEGORIES)].title()} Store {i}" for i in range(n_merchants)]
    merchant_risk = rng.beta(0.3, 7, n_merchants)  # most merchants are legit

    # ── Sample user/transaction assignments ───────────────────────────────────
    user_idx = rng.integers(0, n_users, n_rows)
    merchant_idx = rng.integers(0, n_merchants, n_rows)

    # ── Timestamps ───────────────────────────────────────────────────────────
    ts_start = pd.Timestamp(start_date).timestamp()
    ts_end = pd.Timestamp(end_date).timestamp()
    raw_ts = rng.uniform(ts_start, ts_end, n_rows)
    # Inject time-of-day bias (more transactions 8am-10pm)
    hour_weights = np.array([
        1, 1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 9,   # 0–11
        9, 8, 7, 7, 8, 9, 9, 8, 7, 5, 3, 2,    # 12–23
    ], dtype=float)
    hour_weights /= hour_weights.sum()
    hour_offsets = rng.choice(range(24), n_rows, p=hour_weights)
    timestamps = pd.to_datetime(raw_ts - raw_ts % 86400 + hour_offsets * 3600, unit="s")

    # ── Amounts ──────────────────────────────────────────────────────────────
    base_amounts = rng.lognormal(
        np.log(user_avg_amount[user_idx]),
        0.8,
        n_rows,
    )
    # Fraud transactions tend to be unusually large or small
    fraud_mask_prelim = (
        rng.random(n_rows) < (user_risk[user_idx] * 0.4 + merchant_risk[merchant_idx] * 0.3)
    )
    amounts = np.where(
        fraud_mask_prelim,
        base_amounts * rng.choice([0.1, 2.0, 5.0, 10.0], n_rows, p=[0.2, 0.3, 0.3, 0.2]),
        base_amounts,
    ).round(2)
    amounts = np.clip(amounts, 0.01, 50_000)

    # ── Categories ───────────────────────────────────────────────────────────
    cat_idx = np.where(
        rng.random(n_rows) < 0.6,
        user_preferred_cat[user_idx],
        rng.integers(0, len(CATEGORIES), n_rows),
    )
    categories = [CATEGORIES[i] for i in cat_idx]

    # ── Channels ─────────────────────────────────────────────────────────────
    ch_weights = [0.35, 0.35, 0.15, 0.1, 0.05]
    channels = rng.choice(CHANNELS, n_rows, p=ch_weights)

    # ── Countries ────────────────────────────────────────────────────────────
    # Most from home country, occasional international
    country_weights = [0.60, 0.10, 0.07, 0.07, 0.05, 0.04, 0.04, 0.03]
    countries = rng.choice(COUNTRIES, n_rows, p=country_weights)

    # ── Memo text ────────────────────────────────────────────────────────────
    memos = [
        MEMO_TEMPLATES[rng.integers(0, len(MEMO_TEMPLATES))].format(
            cat=categories[i], merchant=merchant_names[merchant_idx[i]], country=countries[i]
        )
        for i in range(n_rows)
    ]

    # ── User age at transaction ───────────────────────────────────────────────
    user_age_days = [
        max(0, (timestamps[i] - user_reg_date[user_idx[i]]).days)
        for i in range(n_rows)
    ]

    # ── Account tier ─────────────────────────────────────────────────────────
    tier_thresholds = np.percentile(user_avg_amount, [33, 66])
    tiers = np.where(
        user_avg_amount[user_idx] < tier_thresholds[0], "bronze",
        np.where(user_avg_amount[user_idx] < tier_thresholds[1], "silver", "gold")
    )

    # ── Fraud label ──────────────────────────────────────────────────────────
    # Fraud probability driven by user risk, merchant risk, amount spike, channel
    channel_risk = np.array([0.01, 0.015, 0.02, 0.01, 0.05])[
        [CHANNELS.index(c) for c in channels]
    ]
    amount_spike = (amounts > np.percentile(amounts, 97)).astype(float) * 0.05
    fraud_prob = np.clip(
        user_risk[user_idx] * 0.5
        + merchant_risk[merchant_idx] * 0.3
        + channel_risk * 0.1
        + amount_spike,
        0, 0.9,
    )
    is_fraud = (rng.random(n_rows) < fraud_prob).astype(int)

    # ── Assemble DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame({
        "transaction_id": [f"T{i:08d}" for i in range(n_rows)],
        "user_id": [user_ids[i] for i in user_idx],
        "event_timestamp": pd.DatetimeIndex(timestamps),
        "amount": amounts,
        "category": categories,
        "merchant_id": [merchant_ids[i] for i in merchant_idx],
        "merchant_name": [merchant_names[i] for i in merchant_idx],
        "channel": channels,
        "country": countries,
        "memo": memos,
        "user_age_days": user_age_days,
        "account_tier": tiers,
        "is_fraud": is_fraud,
    })

    # Sort by timestamp (realistic order)
    df = df.sort_values("event_timestamp").reset_index(drop=True)

    # Print quick summary
    fraud_rate = df["is_fraud"].mean()
    print(
        f"Generated {len(df):,} transactions · {n_users} users · {n_merchants} merchants · "
        f"fraud rate {fraud_rate:.2%}"
    )
    return df


if __name__ == "__main__":
    df = generate_transactions(n_users=2_000, n_rows=100_000, seed=42)
    out = Path(__file__).parent / "sample_transactions.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved → {out}")
    print(df.head())
    print(df.dtypes)
    print(df.describe())

