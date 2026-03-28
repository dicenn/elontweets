"""
Phase 2: Exploratory Data Analysis
Analyzes 6 months of Elon Musk tweet data per the spec requirements.

Outputs a structured report covering:
  - Daily eligible count distribution (mean, median, stdev, skewness, CV)
  - Temporal patterns (hourly, day-of-week, weekend/weekday)
  - Autocorrelation (lag-1 through lag-7)
  - Tweet type composition
  - Flurry detection (hourly rate > 2x trailing 6-hour avg AND count >= 5)
  - Slope/regime shift detection
  - Inter-tweet gap analysis
  - Engagement analysis by tweet type
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
from datetime import timedelta
from statsmodels.tsa.stattools import acf

DATA_DIR = Path(__file__).resolve().parent.parent / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis"


def load_data():
    """Load merged eligible and all-tweets CSVs."""
    eligible = pd.read_csv(DATA_DIR / "musk_tweets_eligible_merged.csv")
    all_tweets = pd.read_csv(DATA_DIR / "musk_tweets_all_merged.csv")

    for df in [eligible, all_tweets]:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["created_at_et"] = pd.to_datetime(df["created_at_et"], utc=True)
        # Extract ET hour/day without timezone issues
        df["hour_et"] = df["created_at_et"].dt.hour
        df["dow"] = df["created_at_et"].dt.dayofweek  # 0=Mon, 6=Sun
        df["dow_name"] = df["created_at_et"].dt.day_name()
        df["date_et"] = df["created_at_et"].dt.date
        df["is_weekend"] = df["dow"].isin([5, 6])

    return eligible, all_tweets


def daily_distribution(eligible):
    """Compute daily eligible tweet count statistics."""
    daily = eligible.groupby("date_et").size()
    result = {
        "n_days": len(daily),
        "total_tweets": int(daily.sum()),
        "mean": round(float(daily.mean()), 2),
        "median": round(float(daily.median()), 2),
        "std": round(float(daily.std()), 2),
        "min": int(daily.min()),
        "max": int(daily.max()),
        "q25": round(float(daily.quantile(0.25)), 2),
        "q75": round(float(daily.quantile(0.75)), 2),
        "skewness": round(float(stats.skew(daily)), 4),
        "kurtosis": round(float(stats.kurtosis(daily)), 4),
        "cv": round(float(daily.std() / daily.mean()), 4),
    }
    return daily, result


def temporal_patterns(eligible, daily):
    """Hourly, day-of-week, and weekend/weekday breakdowns."""
    hourly = eligible.groupby("hour_et").size()
    hourly_pct = (hourly / hourly.sum() * 100).round(2)

    dow_daily = eligible.groupby(["date_et", "dow_name", "is_weekend"]).size().reset_index(name="count")
    dow_summary = dow_daily.groupby("dow_name")["count"].agg(["mean", "median", "std"]).round(2)
    # Reorder to Mon-Sun
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_summary = dow_summary.reindex(day_order)

    weekend_days = dow_daily[dow_daily["is_weekend"]]["count"]
    weekday_days = dow_daily[~dow_daily["is_weekend"]]["count"]

    return {
        "hourly_distribution": {int(h): {"count": int(c), "pct": float(hourly_pct[h])} for h, c in hourly.items()},
        "peak_hours_et": sorted(hourly.nlargest(5).index.tolist()),
        "day_of_week": {
            day: {"mean": float(row["mean"]), "median": float(row["median"]), "std": float(row["std"])}
            for day, row in dow_summary.iterrows()
        },
        "weekend_vs_weekday": {
            "weekday_mean": round(float(weekday_days.mean()), 2),
            "weekend_mean": round(float(weekend_days.mean()), 2),
            "ratio": round(float(weekend_days.mean() / weekday_days.mean()), 4) if weekday_days.mean() > 0 else None,
        },
    }


def autocorrelation_analysis(daily):
    """Compute autocorrelation at various lags."""
    if len(daily) < 10:
        return {"error": "insufficient data"}
    nlags = min(14, len(daily) // 2 - 1)
    acf_values = acf(daily.values, nlags=nlags, fft=True)
    result = {}
    for lag in range(1, nlags + 1):
        result[f"lag_{lag}"] = round(float(acf_values[lag]), 4)
    return result


def tweet_type_composition(eligible, all_tweets):
    """Break down tweet types."""
    eligible_types = eligible["tweet_type"].value_counts()
    eligible_pct = (eligible_types / len(eligible) * 100).round(2)

    all_types = all_tweets["tweet_type"].value_counts()
    all_pct = (all_types / len(all_tweets) * 100).round(2)

    return {
        "eligible_only": {
            t: {"count": int(eligible_types.get(t, 0)), "pct": float(eligible_pct.get(t, 0))}
            for t in ["retweet", "quote", "original"]
        },
        "all_tweets": {
            t: {"count": int(all_types.get(t, 0)), "pct": float(all_pct.get(t, 0))}
            for t in ["retweet", "quote", "original", "reply"]
        },
        "reply_share_of_all": round(float(all_types.get("reply", 0) / len(all_tweets) * 100), 2),
    }


def flurry_detection(eligible):
    """
    Detect flurries: hourly rate > 2x trailing 6-hour average AND count >= 5.
    Returns summary stats and top flurries.
    """
    eligible_sorted = eligible.sort_values("created_at").copy()
    eligible_sorted["hour_bucket"] = eligible_sorted["created_at"].dt.floor("h")

    hourly_counts = eligible_sorted.groupby("hour_bucket").size().reset_index(name="count")
    hourly_counts = hourly_counts.set_index("hour_bucket").asfreq("h", fill_value=0)

    # Trailing 6-hour average (excluding current hour)
    hourly_counts["trailing_6h_avg"] = hourly_counts["count"].rolling(6, min_periods=1).mean().shift(1)
    hourly_counts["trailing_6h_avg"] = hourly_counts["trailing_6h_avg"].fillna(hourly_counts["count"].mean())

    flurries = hourly_counts[
        (hourly_counts["count"] > 2 * hourly_counts["trailing_6h_avg"]) & (hourly_counts["count"] >= 5)
    ].copy()
    flurries["ratio"] = (flurries["count"] / flurries["trailing_6h_avg"]).round(2)

    n_days = (hourly_counts.index.max() - hourly_counts.index.min()).days or 1

    top_flurries = flurries.nlargest(10, "count")

    return {
        "total_flurry_hours": len(flurries),
        "flurries_per_day": round(len(flurries) / n_days, 2),
        "avg_flurry_count": round(float(flurries["count"].mean()), 2) if len(flurries) > 0 else 0,
        "max_flurry_count": int(flurries["count"].max()) if len(flurries) > 0 else 0,
        "top_10_flurries": [
            {
                "hour": str(idx),
                "count": int(row["count"]),
                "trailing_avg": round(float(row["trailing_6h_avg"]), 2),
                "ratio": float(row["ratio"]),
            }
            for idx, row in top_flurries.iterrows()
        ],
    }


def slope_shift_detection(daily):
    """
    Detect regime shifts in daily posting rate using rolling mean comparison.
    """
    if len(daily) < 30:
        return {"error": "insufficient data for slope shift detection (need 30+ days)"}

    daily_df = daily.reset_index()
    daily_df.columns = ["date", "count"]
    daily_df = daily_df.sort_values("date")
    daily_df["rolling_7d"] = daily_df["count"].rolling(7, center=True).mean()
    daily_df["rolling_30d"] = daily_df["count"].rolling(30, center=True).mean()

    # Detect shifts: 7d avg deviates >50% from 30d avg
    daily_df["shift_ratio"] = daily_df["rolling_7d"] / daily_df["rolling_30d"]
    shifts = daily_df[
        (daily_df["shift_ratio"] > 1.5) | (daily_df["shift_ratio"] < 0.67)
    ].dropna(subset=["shift_ratio"])

    # Monthly averages as a simpler regime view
    daily_df["month"] = pd.to_datetime(daily_df["date"]).dt.to_period("M")
    monthly = daily_df.groupby("month")["count"].agg(["mean", "std", "sum"]).round(2)

    return {
        "monthly_rates": {
            str(m): {"mean": float(row["mean"]), "std": float(row["std"]), "total": int(row["sum"])}
            for m, row in monthly.iterrows()
        },
        "significant_shift_days": len(shifts),
        "overall_trend_slope": round(
            float(np.polyfit(range(len(daily_df)), daily_df["count"].values, 1)[0]), 4
        ),
    }


def inter_tweet_gaps(eligible):
    """Analyze time gaps between consecutive eligible tweets."""
    sorted_tweets = eligible.sort_values("created_at")
    gaps = sorted_tweets["created_at"].diff().dt.total_seconds().dropna() / 60  # in minutes

    bins = [0, 1, 2, 5, 10, 30, 60, 120, 360, 720, 1440, float("inf")]
    labels = ["<1m", "1-2m", "2-5m", "5-10m", "10-30m", "30m-1h", "1-2h", "2-6h", "6-12h", "12-24h", ">24h"]
    gap_bins = pd.cut(gaps, bins=bins, labels=labels, right=False)
    gap_dist = gap_bins.value_counts().sort_index()
    gap_pct = (gap_dist / len(gaps) * 100).round(2)

    # Cumulative: % under 5min
    under_5 = float(gaps[gaps < 5].count() / len(gaps) * 100)
    over_2h = float(gaps[gaps > 120].count() / len(gaps) * 100)

    return {
        "total_gaps": len(gaps),
        "mean_gap_minutes": round(float(gaps.mean()), 2),
        "median_gap_minutes": round(float(gaps.median()), 2),
        "std_gap_minutes": round(float(gaps.std()), 2),
        "min_gap_minutes": round(float(gaps.min()), 4),
        "max_gap_minutes": round(float(gaps.max()), 2),
        "pct_under_5min": round(under_5, 2),
        "pct_over_2h": round(over_2h, 2),
        "distribution": {
            label: {"count": int(gap_dist[label]), "pct": float(gap_pct[label])} for label in labels
        },
    }


def engagement_analysis(eligible):
    """Engagement metrics by tweet type."""
    metrics = ["like_count", "reply_count", "retweet_count", "quote_count", "impression_count", "bookmark_count"]
    result = {}

    for tweet_type in ["original", "retweet", "quote"]:
        subset = eligible[eligible["tweet_type"] == tweet_type]
        if len(subset) == 0:
            continue
        type_stats = {}
        for m in metrics:
            col = pd.to_numeric(subset[m], errors="coerce").dropna()
            if len(col) == 0:
                continue
            type_stats[m] = {
                "mean": round(float(col.mean()), 0),
                "median": round(float(col.median()), 0),
                "max": int(col.max()),
            }
        result[tweet_type] = {"count": len(subset), "metrics": type_stats}

    return result


def weekly_period_analysis(eligible):
    """
    Analyze weekly periods matching Polymarket resolution windows.
    This helps calibrate the model against actual market periods.
    """
    eligible_sorted = eligible.sort_values("created_at").copy()
    eligible_sorted["week"] = eligible_sorted["created_at"].dt.isocalendar().week.astype(int)
    eligible_sorted["year"] = eligible_sorted["created_at"].dt.isocalendar().year.astype(int)

    weekly = eligible_sorted.groupby(["year", "week"]).agg(
        count=("tweet_id", "size"),
        first_tweet=("created_at", "min"),
        last_tweet=("created_at", "max"),
    )

    return {
        "n_weeks": len(weekly),
        "weekly_mean": round(float(weekly["count"].mean()), 2),
        "weekly_std": round(float(weekly["count"].std()), 2),
        "weekly_min": int(weekly["count"].min()),
        "weekly_max": int(weekly["count"].max()),
        "weekly_cv": round(float(weekly["count"].std() / weekly["count"].mean()), 4),
        "weekly_counts": [
            {
                "year": int(yr),
                "week": int(wk),
                "count": int(row["count"]),
                "start": str(row["first_tweet"]),
                "end": str(row["last_tweet"]),
            }
            for (yr, wk), row in weekly.iterrows()
        ],
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    eligible, all_tweets = load_data()
    print(f"  Eligible tweets: {len(eligible):,}")
    print(f"  All tweets: {len(all_tweets):,}")
    print(f"  Date range: {eligible['date_et'].min()} to {eligible['date_et'].max()}")

    print("\n1. Daily distribution analysis...")
    daily, daily_stats = daily_distribution(eligible)
    print(f"  Mean: {daily_stats['mean']}, Median: {daily_stats['median']}, "
          f"Std: {daily_stats['std']}, CV: {daily_stats['cv']}")

    print("\n2. Temporal patterns...")
    temporal = temporal_patterns(eligible, daily)
    print(f"  Peak hours (ET): {temporal['peak_hours_et']}")
    print(f"  Weekday mean: {temporal['weekend_vs_weekday']['weekday_mean']}, "
          f"Weekend mean: {temporal['weekend_vs_weekday']['weekend_mean']}")

    print("\n3. Autocorrelation...")
    autocorr = autocorrelation_analysis(daily)
    for lag in ["lag_1", "lag_2", "lag_3"]:
        if lag in autocorr:
            print(f"  {lag}: {autocorr[lag]}")

    print("\n4. Tweet type composition...")
    composition = tweet_type_composition(eligible, all_tweets)
    for t, v in composition["eligible_only"].items():
        print(f"  {t}: {v['count']:,} ({v['pct']}%)")
    print(f"  Reply share of all: {composition['reply_share_of_all']}%")

    print("\n5. Flurry detection...")
    flurries = flurry_detection(eligible)
    print(f"  Total flurry hours: {flurries['total_flurry_hours']}")
    print(f"  Flurries/day: {flurries['flurries_per_day']}")
    print(f"  Max flurry count: {flurries['max_flurry_count']}")

    print("\n6. Slope/regime shift detection...")
    slopes = slope_shift_detection(daily)
    if "monthly_rates" in slopes:
        for m, v in slopes["monthly_rates"].items():
            print(f"  {m}: mean={v['mean']}/day, total={v['total']}")
        print(f"  Overall trend slope: {slopes['overall_trend_slope']} tweets/day per day")

    print("\n7. Inter-tweet gap analysis...")
    gaps = inter_tweet_gaps(eligible)
    print(f"  Mean gap: {gaps['mean_gap_minutes']:.1f} min, Median: {gaps['median_gap_minutes']:.1f} min")
    print(f"  Under 5min: {gaps['pct_under_5min']}%, Over 2h: {gaps['pct_over_2h']}%")

    print("\n8. Engagement by type...")
    engagement = engagement_analysis(eligible)
    for t, v in engagement.items():
        imp = v["metrics"].get("impression_count", {})
        print(f"  {t} (n={v['count']}): avg impressions={imp.get('mean', 'N/A'):,.0f}")

    print("\n9. Weekly period analysis (for Polymarket calibration)...")
    weekly = weekly_period_analysis(eligible)
    print(f"  Weeks: {weekly['n_weeks']}, Mean: {weekly['weekly_mean']}, "
          f"Std: {weekly['weekly_std']}, CV: {weekly['weekly_cv']}")

    # Save full results
    report = {
        "metadata": {
            "eligible_tweets": len(eligible),
            "all_tweets": len(all_tweets),
            "date_range": f"{eligible['date_et'].min()} to {eligible['date_et'].max()}",
        },
        "daily_distribution": daily_stats,
        "temporal_patterns": temporal,
        "autocorrelation": autocorr,
        "tweet_type_composition": composition,
        "flurry_detection": flurries,
        "slope_shift_detection": slopes,
        "inter_tweet_gaps": gaps,
        "engagement_by_type": engagement,
        "weekly_periods": weekly,
    }

    output_path = OUTPUT_DIR / "eda_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {output_path}")


if __name__ == "__main__":
    main()
