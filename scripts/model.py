"""
Phase 4: Three-Layer Prediction Model

Layer 1 - Base Rate Model: Negative binomial regression predicting daily eligible
          tweet count from structural + behavioral features.
Layer 2 - Real-Time Adjustment: Bayesian updating of the weekly distribution given
          observed tweets within a market period.
Layer 3 - Market Translation: Convert probability distribution to trading positions
          vs Polymarket bucket prices.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"


# =============================================================================
# Layer 1: Base Rate Model
# =============================================================================

class BaseRateModel:
    """
    Predicts daily eligible tweet count distribution.

    Uses a gradient-boosted approach with negative binomial output distribution.
    For simplicity and interpretability, we use a weighted historical approach
    with regime-aware adjustments, then fit a negative binomial to the residuals.

    The model estimates (mu, alpha) for a negative binomial distribution where:
      - mu = expected daily count
      - alpha = overdispersion parameter
    """

    def __init__(self):
        self.fitted = False
        self.feature_weights = None
        self.base_mu = None
        self.base_alpha = None
        self.regime_adjustments = {}
        self.daily_autocorrelation = 0.0

    def fit(self, daily_features):
        """
        Fit the base rate model on historical daily data.
        Uses the last 14 days more heavily (recency weighting).
        """
        df = daily_features.dropna(subset=["eligible_count"]).copy()

        # Target
        y = df["eligible_count"].values

        # Fit negative binomial to overall distribution
        self.base_mu = float(np.mean(y))
        self.base_alpha = self._fit_nb_alpha(y, self.base_mu)

        # Compute regime adjustments
        # Day-of-week effect
        dow_means = df.groupby("dow")["eligible_count"].mean()
        overall_mean = df["eligible_count"].mean()
        self.regime_adjustments["dow"] = {
            int(dow): float(m / overall_mean) for dow, m in dow_means.items()
        }

        # Weekend effect
        wkend = df[df["is_weekend"] == 1]["eligible_count"].mean()
        wkday = df[df["is_weekend"] == 0]["eligible_count"].mean()
        self.regime_adjustments["weekend_ratio"] = float(wkend / wkday) if wkday > 0 else 1.0

        # Monthly regime (captures trend/regime shifts)
        df["month_key"] = pd.to_datetime(df["date"]).dt.to_period("M")
        monthly_means = df.groupby("month_key")["eligible_count"].mean()
        self.regime_adjustments["monthly"] = {
            str(m): float(v / overall_mean) for m, v in monthly_means.items()
        }

        # Lag-1 coefficient (autoregressive component)
        valid = df.dropna(subset=["eligible_count_lag1"])
        if len(valid) > 10:
            corr = np.corrcoef(valid["eligible_count"], valid["eligible_count_lag1"])[0, 1]
            self.regime_adjustments["lag1_corr"] = float(corr)
        else:
            self.regime_adjustments["lag1_corr"] = 0

        # Flurry day effect
        flurry_days = df[df["has_flurry"] == 1]["eligible_count"]
        no_flurry = df[df["has_flurry"] == 0]["eligible_count"]
        if len(flurry_days) > 0 and len(no_flurry) > 0:
            self.regime_adjustments["flurry_uplift"] = float(flurry_days.mean() / no_flurry.mean())
        else:
            self.regime_adjustments["flurry_uplift"] = 1.0

        # Recency-weighted parameters (last 10 days — faster regime response)
        if len(df) > 10:
            recent = df.tail(10)["eligible_count"]
            self.recent_mu = float(recent.mean())
            self.recent_alpha = self._fit_nb_alpha(recent.values, self.recent_mu)
        else:
            self.recent_mu = self.base_mu
            self.recent_alpha = self.base_alpha

        # Store empirical daily autocorrelation for weekly variance correction
        if len(df) > 14:
            valid = df.dropna(subset=["eligible_count_lag1"])
            if len(valid) > 10:
                self.daily_autocorrelation = float(
                    np.corrcoef(valid["eligible_count"], valid["eligible_count_lag1"])[0, 1]
                )
            else:
                self.daily_autocorrelation = 0.0
        else:
            self.daily_autocorrelation = 0.0

        self.fitted = True
        return self

    def _fit_nb_alpha(self, counts, mu):
        """Fit negative binomial overdispersion parameter via MLE."""
        var = np.var(counts)
        if var <= mu:
            return 100.0  # Effectively Poisson (low overdispersion)
        # Method of moments estimate
        alpha = mu ** 2 / (var - mu)
        return max(float(alpha), 0.1)

    def predict_daily(self, dow, month=None, lag1_count=None, use_recent=True,
                       classification_features=None):
        """
        Predict the distribution of eligible tweets for a single day.

        Args:
            classification_features: dict with LLM-derived features for the
                previous day (e.g., pct_combative_feud, mean_intensity, feud_pct).
                Used for regime-conditional adjustments when available.

        Returns (mu, alpha) parameterizing a negative binomial distribution.
        """
        if not self.fitted:
            raise ValueError("Model not fitted")

        # Start with base or recent estimate
        mu = self.recent_mu if use_recent else self.base_mu
        alpha = self.recent_alpha if use_recent else self.base_alpha

        # Apply day-of-week adjustment
        dow_factor = self.regime_adjustments["dow"].get(dow, 1.0)
        mu *= dow_factor

        # Apply lag-1 autoregressive adjustment (data shows 0.43 autocorrelation)
        if lag1_count is not None:
            lag1_corr = self.regime_adjustments["lag1_corr"]
            # Stronger adjustment — empirical autocorrelation supports momentum
            expected_base = self.recent_mu if use_recent else self.base_mu
            adjustment = lag1_corr * (lag1_count - expected_base)
            mu += adjustment * 0.75  # Increased from 0.5 — strong momentum signal

        # Apply classification-based regime adjustments when available
        if classification_features is not None:
            # Combative/feud mode drives higher volume (self-reinforcing)
            combative_pct = classification_features.get("pct_combative_feud", 0)
            if combative_pct > 0.15:
                mu *= 1 + (combative_pct - 0.15) * 0.5  # Up to ~10% uplift

            # High intensity days tend to be followed by high volume
            mean_intensity = classification_features.get("mean_intensity", 3)
            if mean_intensity > 3.5:
                mu *= 1 + (mean_intensity - 3.5) * 0.1  # Up to ~15% uplift

            # Low-effort (meme/RT heavy) days = higher raw counts
            low_effort_pct = classification_features.get("pct_low_effort", 0)
            if low_effort_pct > 0.5:
                mu *= 1 + (low_effort_pct - 0.5) * 0.3

            # Feud in progress = volume driver
            feud_pct = classification_features.get("feud_pct", 0)
            if feud_pct > 0.1:
                mu *= 1 + (feud_pct - 0.1) * 0.4

            # High continuation likelihood = more tweets coming
            high_cont = classification_features.get("high_continuation_pct", 0)
            if high_cont > 0.3:
                mu *= 1 + (high_cont - 0.3) * 0.2

        mu = max(mu, 5)  # Floor
        return mu, alpha

    def predict_weekly(self, start_dow, lag1_count=None, n_days=7):
        """
        Predict the distribution of eligible tweets over a week (or n days).
        Sums n independent daily negative binomials.

        Returns (weekly_mu, weekly_var) and the full distribution via simulation.
        """
        daily_mus = []
        current_lag = lag1_count

        for i in range(n_days):
            dow = (start_dow + i) % 7
            mu, alpha = self.predict_daily(dow, lag1_count=current_lag)
            daily_mus.append(mu)
            current_lag = mu  # Use predicted value as lag for next day

        weekly_mu = sum(daily_mus)
        # Variance of sum of correlated NB variables
        # Var(NB) = mu + mu^2/alpha per day, plus covariance terms
        alpha = self.recent_alpha
        daily_vars = [m + m ** 2 / alpha for m in daily_mus]
        independent_var = sum(daily_vars)

        # Add covariance: cov(day_i, day_j) ≈ rho^|i-j| * sqrt(var_i * var_j)
        rho = getattr(self, "daily_autocorrelation", 0.0)
        covariance_sum = 0.0
        for i in range(n_days):
            for j in range(i + 1, n_days):
                cov_ij = (rho ** abs(i - j)) * np.sqrt(daily_vars[i] * daily_vars[j])
                covariance_sum += cov_ij
        weekly_var = independent_var + 2 * covariance_sum

        return weekly_mu, weekly_var, daily_mus


# =============================================================================
# Layer 2: Real-Time Adjustment Layer
# =============================================================================

class RealTimeAdjuster:
    """
    Bayesian updating of the period forecast given observed tweets.

    Uses conjugate prior approach:
    - Prior: Negative binomial from Layer 1
    - Likelihood: Observed Poisson-like arrival process
    - Posterior: Updated negative binomial with adjusted parameters
    """

    def __init__(self, base_model):
        self.base_model = base_model

    def update_forecast(self, period_mu, period_var, count_so_far,
                        elapsed_fraction, velocity_1h=None, flurry_active=False):
        """
        Update the weekly forecast given partial observations.

        Args:
            period_mu: Prior expected total for the period
            period_var: Prior variance
            count_so_far: Tweets observed so far in this period
            elapsed_fraction: Fraction of period elapsed (0-1)
            velocity_1h: Tweets per hour in the last hour
            flurry_active: Whether a flurry is currently detected

        Returns:
            Dict with posterior distribution parameters and bucket probabilities.
        """
        if elapsed_fraction <= 0:
            # Period hasn't started - return prior
            return self._make_result(period_mu, period_var, 0)

        # Expected count so far based on prior
        expected_so_far = period_mu * elapsed_fraction
        remaining_fraction = 1 - elapsed_fraction

        # Bayesian update: blend prior with observed
        # Weight observed data more as we see more of the period
        obs_weight = min(elapsed_fraction * 1.5, 0.95)  # Cap at 95% observation weight
        prior_weight = 1 - obs_weight

        # Implied rate from observations
        if elapsed_fraction > 0.01:
            observed_rate = count_so_far / elapsed_fraction
        else:
            observed_rate = period_mu

        # Posterior mean: weighted blend
        posterior_mu = prior_weight * period_mu + obs_weight * observed_rate
        # Ensure we account for what's already happened
        remaining_mu = max(posterior_mu - count_so_far, 0)

        # Posterior variance: shrinks with more data
        posterior_var = period_var * (remaining_fraction ** 1.5)

        # Velocity adjustment
        if velocity_1h is not None and elapsed_fraction > 0.05:
            # If current velocity differs significantly from expected
            expected_hourly = period_mu / (7 * 24)  # expected per hour
            velocity_ratio = velocity_1h / max(expected_hourly, 0.1)
            if velocity_ratio > 2:
                # Higher than expected - adjust up
                remaining_mu *= min(1 + (velocity_ratio - 1) * 0.3 * remaining_fraction, 2.0)
            elif velocity_ratio < 0.5:
                # Lower than expected - adjust down
                remaining_mu *= max(0.5, velocity_ratio ** 0.5)

        # Flurry adjustment: widen the distribution
        if flurry_active:
            posterior_var *= 1.5

        final_mu = count_so_far + remaining_mu
        final_var = posterior_var + remaining_mu  # Add Poisson variance for remaining

        return self._make_result(final_mu, final_var, count_so_far)

    def _make_result(self, mu, var, count_so_far):
        """Convert mu/var to bucket probabilities."""
        # Use normal approximation for weekly totals (CLT applies)
        std = max(np.sqrt(var), 1)

        # Standard Polymarket buckets (these may need adjustment based on actual market)
        buckets = [
            (0, 174, "0-174"),
            (175, 224, "175-224"),
            (225, 274, "225-274"),
            (275, 324, "275-324"),
            (325, 374, "325-374"),
            (375, 424, "375-424"),
            (425, 474, "425-474"),
            (475, float("inf"), "475+"),
        ]

        bucket_probs = {}
        for lo, hi, label in buckets:
            if hi == float("inf"):
                prob = 1 - stats.norm.cdf(lo - 0.5, mu, std)
            else:
                prob = stats.norm.cdf(hi + 0.5, mu, std) - stats.norm.cdf(lo - 0.5, mu, std)
            bucket_probs[label] = round(float(max(prob, 0)), 6)

        # Normalize
        total = sum(bucket_probs.values())
        if total > 0:
            bucket_probs = {k: round(v / total, 6) for k, v in bucket_probs.items()}

        return {
            "posterior_mu": round(float(mu), 2),
            "posterior_std": round(float(std), 2),
            "count_so_far": int(count_so_far),
            "bucket_probabilities": bucket_probs,
        }


# =============================================================================
# Layer 3: Market Translation Layer
# =============================================================================

class MarketTranslator:
    """
    Converts model probabilities to trading positions.

    Compares model's probability distribution to market-implied probabilities
    and identifies +EV opportunities.
    """

    def __init__(self, edge_threshold=0.05, max_position_pct=0.10, fee_rate=0.02):
        """
        Args:
            edge_threshold: Minimum edge (model_prob - market_prob) to trade
            max_position_pct: Maximum position as fraction of bankroll
            fee_rate: Polymarket fee rate (approximately 2%)
        """
        self.edge_threshold = edge_threshold
        self.max_position_pct = max_position_pct
        self.fee_rate = fee_rate

    def evaluate_markets(self, model_probs, market_prices):
        """
        Compare model probabilities to market prices and recommend positions.

        Args:
            model_probs: Dict of {bucket: probability} from the model
            market_prices: Dict of {bucket: price/implied_prob} from Polymarket

        Returns:
            List of recommended positions with expected value.
        """
        positions = []

        for bucket, model_prob in model_probs.items():
            if bucket not in market_prices:
                continue

            market_prob = market_prices[bucket]

            # Edge = model probability - market implied probability
            edge = model_prob - market_prob

            # Expected value after fees
            # If we buy at market_price and win with model_prob:
            # EV = model_prob * (1 - fee_rate) * (1 / market_price - 1) - (1 - model_prob) * 1
            # Simplified: EV = model_prob * (1 - fee) / market_price - 1
            if market_prob > 0 and market_prob < 1:
                ev_per_dollar = model_prob * (1 - self.fee_rate) / market_prob - 1
            else:
                ev_per_dollar = 0

            # Kelly criterion for position sizing
            # f* = (bp - q) / b where b = (1/market_price - 1), p = model_prob, q = 1-p
            if market_prob > 0 and market_prob < 1:
                b = (1 / market_prob) - 1
                p = model_prob
                q = 1 - p
                kelly = (b * p - q) / b if b > 0 else 0
                kelly = max(0, min(kelly, self.max_position_pct))  # Constrain
                # Use fractional Kelly (25%) for safety
                kelly *= 0.25
            else:
                kelly = 0

            action = "NONE"
            if edge > self.edge_threshold and ev_per_dollar > 0:
                action = "BUY"
            elif edge < -self.edge_threshold:
                # Could short / buy NO token
                action = "SELL"

            positions.append({
                "bucket": bucket,
                "model_prob": round(model_prob, 4),
                "market_prob": round(market_prob, 4),
                "edge": round(edge, 4),
                "ev_per_dollar": round(ev_per_dollar, 4),
                "kelly_fraction": round(kelly, 4),
                "action": action,
            })

        # Sort by absolute edge
        positions.sort(key=lambda x: abs(x["edge"]), reverse=True)
        return positions

    def format_positions(self, positions):
        """Pretty-print position recommendations."""
        lines = []
        lines.append(f"{'Bucket':<10} {'Model':>7} {'Market':>7} {'Edge':>7} {'EV/$$':>7} {'Kelly':>7} {'Action'}")
        lines.append("-" * 65)
        for p in positions:
            lines.append(
                f"{p['bucket']:<10} {p['model_prob']:>7.1%} {p['market_prob']:>7.1%} "
                f"{p['edge']:>+7.1%} {p['ev_per_dollar']:>+7.1%} {p['kelly_fraction']:>7.2%} {p['action']}"
            )
        return "\n".join(lines)


def main():
    """Demo: fit model on historical data and show predictions."""
    from features import load_tweets, build_daily_features

    ANALYSIS_DIR.mkdir(exist_ok=True)

    print("Loading data and building features...")
    eligible, all_tweets = load_tweets()
    daily_features = build_daily_features(eligible, all_tweets)

    print(f"  {len(daily_features)} days of features")

    # --- Layer 1: Fit base rate model ---
    print("\n=== Layer 1: Base Rate Model ===")
    model = BaseRateModel()
    model.fit(daily_features)

    print(f"  Base mu: {model.base_mu:.1f}, Recent mu: {model.recent_mu:.1f}")
    print(f"  Base alpha: {model.base_alpha:.2f}, Recent alpha: {model.recent_alpha:.2f}")
    print(f"  Lag-1 correlation: {model.regime_adjustments['lag1_corr']:.3f}")
    print(f"  Flurry uplift: {model.regime_adjustments['flurry_uplift']:.2f}x")
    print(f"  DOW adjustments: {json.dumps(model.regime_adjustments['dow'], indent=2)}")

    # Predict next week
    last_day = daily_features.iloc[-1]
    start_dow = (int(last_day["dow"]) + 1) % 7
    lag1 = int(last_day["eligible_count"])

    weekly_mu, weekly_var, daily_mus = model.predict_weekly(start_dow, lag1_count=lag1)
    print(f"\n  Weekly forecast: mu={weekly_mu:.0f}, std={np.sqrt(weekly_var):.0f}")
    print(f"  Daily breakdown: {[f'{m:.0f}' for m in daily_mus]}")

    # --- Layer 2: Real-time adjustment demo ---
    print("\n=== Layer 2: Real-Time Adjustment ===")
    adjuster = RealTimeAdjuster(model)

    # Simulate mid-week observation
    for frac, count in [(0.0, 0), (0.14, 45), (0.29, 95), (0.43, 160), (0.57, 210), (0.71, 290), (0.86, 340)]:
        result = adjuster.update_forecast(weekly_mu, weekly_var, count, frac)
        day_num = int(frac * 7)
        print(f"  Day {day_num} ({frac:.0%} elapsed): count={count}, "
              f"posterior_mu={result['posterior_mu']:.0f}, std={result['posterior_std']:.0f}")
        if frac == 0.43:
            mid_week_result = result

    # --- Layer 3: Market translation demo ---
    print("\n=== Layer 3: Market Translation ===")
    translator = MarketTranslator(edge_threshold=0.05)

    # Example market prices (would come from Polymarket API)
    example_market = {
        "0-174": 0.02,
        "175-224": 0.05,
        "225-274": 0.12,
        "275-324": 0.22,
        "325-374": 0.25,
        "375-424": 0.18,
        "425-474": 0.10,
        "475+": 0.06,
    }

    positions = translator.evaluate_markets(mid_week_result["bucket_probabilities"], example_market)
    print(translator.format_positions(positions))

    # Save model state
    model_state = {
        "base_mu": model.base_mu,
        "base_alpha": model.base_alpha,
        "recent_mu": model.recent_mu,
        "recent_alpha": model.recent_alpha,
        "regime_adjustments": model.regime_adjustments,
        "weekly_forecast": {
            "mu": round(weekly_mu, 1),
            "std": round(float(np.sqrt(weekly_var)), 1),
            "daily_mus": [round(m, 1) for m in daily_mus],
        },
    }
    with open(ANALYSIS_DIR / "model_state.json", "w") as f:
        json.dump(model_state, f, indent=2, default=str)
    print(f"\nModel state saved to {ANALYSIS_DIR / 'model_state.json'}")


if __name__ == "__main__":
    main()
