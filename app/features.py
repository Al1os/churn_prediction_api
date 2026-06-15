import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    # ── Ratio / Proportion (≥2) ──────────────────────────────────────
    # How much discount did the user get relative to spend?
    df["discount_ratio"] = (
        (df["total_amount"] - df["discounted_total"])
        / df["total_amount"].replace(0, np.nan)
    ).fillna(0)

    # Average items per product line — high = bulk buyer
    df["avg_quantity_per_product"] = (
        df["total_quantity"] / df["total_products"].replace(0, np.nan)
    ).fillna(1)

    # ── Time-Based (≥2) ─────────────────────────────────────────────
    # Recency: days since last purchase session
    df["days_inactive"] = df["days_inactive"]  # already in raw data

    # Tenure: how long has this user been around?
    df["days_since_registration"] = df["days_since_registration"]

    # Ratio of inactivity to total tenure (high = disengaging)
    df["inactivity_ratio"] = (
        df["days_inactive"] / df["days_since_registration"].replace(0, np.nan)
    ).fillna(0)

    # ── Aggregation (≥2) ────────────────────────────────────────────
    # Total spend value per product
    df["spend_per_product"] = (
        df["total_amount"] / df["total_products"].replace(0, np.nan)
    ).fillna(0)

    # Total items bought per day of tenure
    df["purchases_per_day"] = (
        df["total_quantity"] / df["days_since_registration"].replace(0, np.nan)
    ).fillna(0)

    # ── Binary / Categorical (≥2) ────────────────────────────────────
    # Is this a large cart? (above median)
    df["large_cart"] = (
        df["total_products"] > df["total_products"].median()
    ).astype(int)

    # Is this a high spender? (above median)
    df["high_spender"] = (
        df["total_amount"] > df["total_amount"].median()
    ).astype(int)

    # ── Churn Label ──────────────────────────────────────────────────
    # Definition: inactive for more than 180 days = churned
    # Business justification: e-commerce users who haven't purchased
    # in 6 months are considered lost customers
    threshold = 180
    df["churned"] = (df["days_inactive"] > threshold).astype(int)

    return df