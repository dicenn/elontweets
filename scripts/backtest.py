"""
Phase 5: Backtest & Validation

Walk-forward backtest on historical weekly periods:
  1. For each week, train only on data available up to that point
  2. Generate probability distribution over Polymarket-style buckets
  3. Compare to actual resolution
  4. Score calibration (Brier score) and simulated P&L

Outputs calibration metrics, per-period results, and aggregate stats.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
from datetime import timedelta

from features import load_tweets, build_daily_features
from model import BaseRateModel, RealTimeAdjuster, MarketTranslator

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"

# Standard Polymarket buckets
BUCKETS = [
    (0, 174, "0-174"),
    (175, 224, "175-224"),
    (225, 274, "225-274"),
    (275, 324, "275-324"),
    (325, 374, "325-374"),
    (375, 424, "375-424"),
    (425, 474, "425-474"),
    (475, float("inf"), "475+"),
]


def get_weekly_periods(daily_features):
    """
    Generate weekly periods for backtesting.
    Uses ISO weeks, requiring at least 4 weeks of warmup data.
    """
    df = daily_features.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)

    # Group by ISO week
    weekly = df.groupby(["iso_year", "iso_week"]).agg(
        start=("date", "min"),
        end=("date", "max"),
        eligible_total=("eligible_count", "sum"),
        n_days=("eligible_count", "count"),
    ).reset_index()

    # Only use complete weeks (7 days) and skip first 4 weeks for warmup
    complete = weekly[weekly["n_days"] == 7].reset_index(drop=True)

    if len(complete) < 5:
        # Relax: use weeks with >= 5 days
        complete = weekly[weekly["n_days"] >= 5].reset_index(drop=True)

    warmup = min(4, len(complete) // 2)
    return complete.iloc[warmup:]


def actual_bucket(count):
    """Determine which bucket an actual count falls into."""
    for lo, hi, label in BUCKETS:
        if lo <= count <= hi:
            return label
    return BUCKETS[-1][2]  # 475+


def brier_score(probs, actual_label):
    """
    Compute Brier score for a set of bucket probabilities.
    Lower is better (0 = perfect, 1 = worst).
    """
    score = 0
    for label, prob in probs.items():
        outcome = 1 if label == actual_label else 0
        score += (prob - outcome) ** 2
    return score / len(probs)


def run_backtest(daily_features, eligible):
    """Execute walk-forward backtest."""
    periods = get_weekly_periods(daily_features)
    print(f"Backtesting on {len(periods)} weekly periods")

    results = []
    brier_scores = []
    correct_bucket = 0
    correct_top2 = 0

    for idx, period in periods.iterrows():
        # Training data: everything before this period
        train = daily_features[pd.to_datetime(daily_features["date"]) < period["start"]].copy()

        if len(train) < 14:
            continue

        # Fit model on training data only
        model = BaseRateModel()
        model.fit(train)

        # Get last training day info
        last_train = train.iloc[-1]
        start_dow = int(period["start"].dayofweek)
        lag1 = int(last_train["eligible_count"])
        n_days = int(period["n_days"])

        # Layer 1: Base rate prediction
        weekly_mu, weekly_var, daily_mus = model.predict_weekly(
            start_dow, lag1_count=lag1, n_days=n_days
        )

        # Get bucket probabilities
        adjuster = RealTimeAdjuster(model)
        result = adjuster.update_forecast(weekly_mu, weekly_var, 0, 0.0)
        model_probs = result["bucket_probabilities"]

        # Actual result
        actual_count = int(period["eligible_total"])
        actual_label = actual_bucket(actual_count)

        # Score
        bs = brier_score(model_probs, actual_label)
        brier_scores.append(bs)

        # Was the highest-probability bucket correct?
        predicted_label = max(model_probs, key=model_probs.get)
        is_correct = predicted_label == actual_label
        if is_correct:
            correct_bucket += 1

        # Was actual in top 2 predicted buckets?
        sorted_buckets = sorted(model_probs.items(), key=lambda x: x[1], reverse=True)
        top2_labels = [b[0] for b in sorted_buckets[:2]]
        is_top2 = actual_label in top2_labels
        if is_top2:
            correct_top2 += 1

        period_result = {
            "period": f"{period['start'].strftime('%Y-%m-%d')} to {period['end'].strftime('%Y-%m-%d')}",
            "actual_count": actual_count,
            "actual_bucket": actual_label,
            "predicted_mu": round(weekly_mu, 1),
            "predicted_std": round(float(np.sqrt(weekly_var)), 1),
            "predicted_bucket": predicted_label,
            "correct": is_correct,
            "in_top2": is_top2,
            "brier_score": round(bs, 4),
            "model_prob_for_actual": round(model_probs.get(actual_label, 0), 4),
            "bucket_probs": model_probs,
        }
        results.append(period_result)

        status = "HIT" if is_correct else ("TOP2" if is_top2 else "MISS")
        print(f"  {period_result['period']}: actual={actual_count} ({actual_label}), "
              f"pred_mu={weekly_mu:.0f} ({predicted_label}), "
              f"brier={bs:.3f} [{status}]")

    n = len(results)
    if n == 0:
        print("No periods to backtest!")
        return {}

    # Aggregate metrics
    metrics = {
        "n_periods": n,
        "mean_brier_score": round(float(np.mean(brier_scores)), 4),
        "median_brier_score": round(float(np.median(brier_scores)), 4),
        "bucket_accuracy": round(correct_bucket / n, 4),
        "top2_accuracy": round(correct_top2 / n, 4),
        "mean_abs_error": round(float(np.mean([
            abs(r["actual_count"] - r["predicted_mu"]) for r in results
        ])), 1),
        "mean_prob_for_actual": round(float(np.mean([
            r["model_prob_for_actual"] for r in results
        ])), 4),
    }

    # Calibration check: for each bucket, compare predicted probability vs actual frequency
    calibration = {}
    for lo, hi, label in BUCKETS:
        predicted_probs = [r["bucket_probs"].get(label, 0) for r in results]
        actual_freq = sum(1 for r in results if r["actual_bucket"] == label) / n
        mean_predicted = float(np.mean(predicted_probs)) if predicted_probs else 0
        calibration[label] = {
            "mean_predicted_prob": round(mean_predicted, 4),
            "actual_frequency": round(actual_freq, 4),
            "calibration_gap": round(abs(mean_predicted - actual_freq), 4),
        }

    # Simulated P&L against example market prices
    # Use the mid-week update for more realistic simulation
    print("\n--- Simulated P&L (assuming uniform market prices as baseline) ---")
    translator = MarketTranslator(edge_threshold=0.05)
    bankroll = 1000
    total_pnl = 0

    for r in results:
        # Assume market prices are uniform (12.5% each) as baseline
        # In production, would use actual Polymarket prices
        uniform_market = {label: 0.125 for _, _, label in BUCKETS}
        positions = translator.evaluate_markets(r["bucket_probs"], uniform_market)

        for pos in positions:
            if pos["action"] == "BUY":
                bet_size = bankroll * pos["kelly_fraction"]
                won = pos["bucket"] == r["actual_bucket"]
                if won:
                    pnl = bet_size * (1 / pos["market_prob"] - 1) * (1 - translator.fee_rate)
                else:
                    pnl = -bet_size
                total_pnl += pnl

    metrics["simulated_pnl_vs_uniform"] = round(total_pnl, 2)
    metrics["simulated_roi_vs_uniform"] = round(total_pnl / bankroll * 100, 2)

    return {
        "metrics": metrics,
        "calibration": calibration,
        "periods": results,
    }


def main():
    ANALYSIS_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    eligible, all_tweets = load_tweets()
    print(f"  {len(eligible):,} eligible tweets")

    print("\nBuilding features...")
    daily_features = build_daily_features(eligible, all_tweets)
    print(f"  {len(daily_features)} days")

    print("\n=== Walk-Forward Backtest ===\n")
    results = run_backtest(daily_features, eligible)

    if not results:
        return

    print("\n=== Summary Metrics ===")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v}")

    print("\n=== Calibration ===")
    print(f"  {'Bucket':<10} {'Predicted':>10} {'Actual':>10} {'Gap':>10}")
    for bucket, cal in results["calibration"].items():
        print(f"  {bucket:<10} {cal['mean_predicted_prob']:>10.1%} "
              f"{cal['actual_frequency']:>10.1%} {cal['calibration_gap']:>10.1%}")

    # Save full results
    output_path = ANALYSIS_DIR / "backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
