"""
Phase 3: Feature Engineering
Builds a daily feature matrix from raw tweet data for the prediction model.

Features per the spec:
  3a. Structural: day of week, hour, is_weekend, days_until_period_end
  3b. Regime/Context: (placeholder for external data - events calendar)
  3c. Behavioral: posting velocity, sleep cycle, content mix, gap patterns,
                  flurry indicators, reply momentum
  3d. Market-Aware: elapsed time, current count, implied rate
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent.parent / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis"


def load_tweets():
    """Load all tweets and eligible tweets."""
    eligible = pd.read_csv(DATA_DIR / "musk_tweets_eligible_merged.csv")
    all_tweets = pd.read_csv(DATA_DIR / "musk_tweets_all_merged.csv")

    for df in [eligible, all_tweets]:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["created_at_et"] = pd.to_datetime(df["created_at_et"], utc=True)
        df["hour_et"] = df["created_at_et"].dt.hour
        df["dow"] = df["created_at_et"].dt.dayofweek
        df["date_et"] = df["created_at_et"].dt.date
        df["is_weekend"] = df["dow"].isin([5, 6])

    return eligible, all_tweets


def build_daily_features(eligible, all_tweets):
    """
    Build a feature matrix with one row per day.
    Each row contains features that would be known at end-of-day
    (for training the base rate model).
    """
    dates = sorted(eligible["date_et"].unique())
    rows = []

    # Pre-compute daily aggregates
    elig_daily = eligible.groupby("date_et")
    all_daily = all_tweets.groupby("date_et")

    for date in dates:
        row = {"date": date}

        # --- Target ---
        day_elig = elig_daily.get_group(date) if date in elig_daily.groups else pd.DataFrame()
        row["eligible_count"] = len(day_elig)

        # --- 3a. Structural ---
        dt = pd.Timestamp(date)
        row["dow"] = dt.dayofweek
        row["is_weekend"] = int(dt.dayofweek in [5, 6])
        row["day_of_month"] = dt.day
        row["month"] = dt.month

        # --- 3c. Behavioral (derived from tweet data) ---

        # Tweet type composition
        if len(day_elig) > 0:
            types = day_elig["tweet_type"].value_counts()
            row["pct_retweet"] = round(types.get("retweet", 0) / len(day_elig), 4)
            row["pct_quote"] = round(types.get("quote", 0) / len(day_elig), 4)
            row["pct_original"] = round(types.get("original", 0) / len(day_elig), 4)
        else:
            row["pct_retweet"] = row["pct_quote"] = row["pct_original"] = 0

        # All tweets for this day (including replies)
        day_all = all_daily.get_group(date) if date in all_daily.groups else pd.DataFrame()
        row["total_tweets_incl_replies"] = len(day_all)
        row["reply_count_today"] = len(day_all[day_all["tweet_type"] == "reply"]) if len(day_all) > 0 else 0
        row["reply_rate"] = round(row["reply_count_today"] / max(len(day_all), 1), 4)

        # Posting velocity - hourly distribution
        if len(day_elig) > 0:
            hourly = day_elig.groupby("hour_et").size()
            row["peak_hour_count"] = int(hourly.max())
            row["active_hours"] = len(hourly)
            row["posting_concentration"] = round(float(hourly.max() / max(len(day_elig), 1)), 4)
        else:
            row["peak_hour_count"] = 0
            row["active_hours"] = 0
            row["posting_concentration"] = 0

        # Inter-tweet gaps
        if len(day_elig) > 1:
            sorted_times = day_elig["created_at"].sort_values()
            gaps = sorted_times.diff().dt.total_seconds().dropna() / 60
            row["mean_gap_min"] = round(float(gaps.mean()), 2)
            row["median_gap_min"] = round(float(gaps.median()), 2)
            row["max_gap_min"] = round(float(gaps.max()), 2)
            row["pct_gaps_under_5min"] = round(float((gaps < 5).sum() / len(gaps)), 4)
        else:
            row["mean_gap_min"] = row["median_gap_min"] = row["max_gap_min"] = 0
            row["pct_gaps_under_5min"] = 0

        # Sleep cycle estimation
        if len(day_elig) > 0:
            hours = day_elig["hour_et"].sort_values()
            row["first_tweet_hour"] = int(hours.iloc[0])
            row["last_tweet_hour"] = int(hours.iloc[-1])
            row["posting_span_hours"] = int(hours.iloc[-1] - hours.iloc[0])
            # Late night posting (after midnight ET, before 5am)
            row["late_night_tweets"] = int(((hours >= 0) & (hours < 5)).sum())
            row["late_night_pct"] = round(row["late_night_tweets"] / len(day_elig), 4)
        else:
            row["first_tweet_hour"] = row["last_tweet_hour"] = 0
            row["posting_span_hours"] = 0
            row["late_night_tweets"] = 0
            row["late_night_pct"] = 0

        # Flurry indicator (any hour with >= 5 tweets AND > 2x the day's hourly average)
        if len(day_elig) > 0:
            hourly = day_elig.groupby("hour_et").size()
            hourly_avg = len(day_elig) / 24
            flurry_hours = hourly[(hourly >= 5) & (hourly > 2 * hourly_avg)]
            row["flurry_hours"] = len(flurry_hours)
            row["max_hourly_count"] = int(hourly.max())
            row["has_flurry"] = int(len(flurry_hours) > 0)
        else:
            row["flurry_hours"] = 0
            row["max_hourly_count"] = 0
            row["has_flurry"] = 0

        # Engagement (mean impressions for eligible tweets)
        if len(day_elig) > 0:
            imp = pd.to_numeric(day_elig["impression_count"], errors="coerce").dropna()
            row["mean_impressions"] = round(float(imp.mean()), 0) if len(imp) > 0 else 0
            likes = pd.to_numeric(day_elig["like_count"], errors="coerce").dropna()
            row["mean_likes"] = round(float(likes.mean()), 0) if len(likes) > 0 else 0
        else:
            row["mean_impressions"] = row["mean_likes"] = 0

        rows.append(row)

    df = pd.DataFrame(rows)

    # --- Lagged features (from prior days) ---
    df = df.sort_values("date").reset_index(drop=True)

    for lag in [1, 2, 3, 7]:
        df[f"eligible_count_lag{lag}"] = df["eligible_count"].shift(lag)
        df[f"reply_rate_lag{lag}"] = df["reply_rate"].shift(lag)

    # Rolling averages
    for window in [3, 7, 14]:
        df[f"eligible_rolling_{window}d_mean"] = df["eligible_count"].rolling(window).mean().round(2)
        df[f"eligible_rolling_{window}d_std"] = df["eligible_count"].rolling(window).std().round(2)

    # Trend: difference between 3-day and 14-day rolling mean
    df["trend_3d_vs_14d"] = (df["eligible_rolling_3d_mean"] - df["eligible_rolling_14d_mean"]).round(2)

    # Momentum: change from yesterday
    df["daily_change"] = df["eligible_count"].diff()
    df["daily_change_pct"] = (df["daily_change"] / df["eligible_count"].shift(1)).round(4)

    return df


def build_intraday_features(eligible, period_start, period_end, as_of_time):
    """
    Build market-aware features for real-time prediction within a period.
    Given a market period [period_start, period_end) and an observation time,
    compute features about progress so far.

    This is used by Layer 2 (real-time adjustment).
    """
    period_tweets = eligible[
        (eligible["created_at"] >= period_start) & (eligible["created_at"] < as_of_time)
    ]

    total_period_hours = (period_end - period_start).total_seconds() / 3600
    elapsed_hours = (as_of_time - period_start).total_seconds() / 3600
    remaining_hours = max(total_period_hours - elapsed_hours, 0.01)

    count_so_far = len(period_tweets)
    elapsed_fraction = elapsed_hours / total_period_hours

    # Current posting rate
    rate_per_hour = count_so_far / max(elapsed_hours, 0.01)

    # Recent velocity (last 1h, 3h, 6h)
    velocities = {}
    for window_h in [1, 3, 6]:
        cutoff = as_of_time - pd.Timedelta(hours=window_h)
        recent = period_tweets[period_tweets["created_at"] >= cutoff]
        velocities[f"velocity_{window_h}h"] = len(recent) / window_h

    # Flurry in progress?
    last_1h_count = velocities["velocity_1h"] * 1  # count in last hour
    last_6h_avg = velocities["velocity_6h"]
    flurry_active = int(last_1h_count >= 5 and last_1h_count > 2 * last_6h_avg)

    # Time since last tweet
    if len(period_tweets) > 0:
        last_tweet_time = period_tweets["created_at"].max()
        minutes_since_last = (as_of_time - last_tweet_time).total_seconds() / 60
    else:
        minutes_since_last = elapsed_hours * 60

    # Hour of day (ET) for the observation time
    hour_et = as_of_time.hour  # assuming as_of_time is in ET

    return {
        "count_so_far": count_so_far,
        "elapsed_hours": round(elapsed_hours, 2),
        "remaining_hours": round(remaining_hours, 2),
        "elapsed_fraction": round(elapsed_fraction, 4),
        "rate_per_hour": round(rate_per_hour, 4),
        "projected_total": round(rate_per_hour * total_period_hours, 1),
        "minutes_since_last_tweet": round(minutes_since_last, 1),
        "flurry_active": flurry_active,
        "hour_et": hour_et,
        **{k: round(v, 4) for k, v in velocities.items()},
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading tweets...")
    eligible, all_tweets = load_data()
    print(f"  Eligible: {len(eligible):,}, All: {len(all_tweets):,}")

    print("\nBuilding daily feature matrix...")
    daily_features = build_daily_features(eligible, all_tweets)
    print(f"  Shape: {daily_features.shape}")
    print(f"  Features: {list(daily_features.columns)}")

    output_path = OUTPUT_DIR / "daily_features.csv"
    daily_features.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Print feature summary
    print("\nFeature summary (last 7 days):")
    print(daily_features.tail(7)[
        ["date", "eligible_count", "dow", "pct_retweet", "reply_rate",
         "has_flurry", "eligible_rolling_7d_mean", "trend_3d_vs_14d"]
    ].to_string(index=False))


def load_data():
    return load_tweets()


if __name__ == "__main__":
    main()
