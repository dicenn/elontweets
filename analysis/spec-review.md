# Spec & Model Review — Elon Tweet Prediction System

**Date**: 2026-03-23
**Scope**: Full review of `elon-tweet-model-spec.md`, implemented code (Phases 1-5), EDA findings, and backtest results.

---

## Executive Summary

The system has a solid foundation: 6 months of clean data (8,103 eligible tweets across 184 days), a working three-layer model architecture, and a walk-forward backtest. However, the backtest reveals **systematic regime-tracking failures** — the model's biggest weakness is adapting to volume regime shifts, which are the most exploitable edge in this market. The spec correctly identifies LLM regime classification and external event data as key features, but neither has been implemented yet. These are the highest-leverage improvements available.

**Key numbers**:
- Bucket accuracy: 18.2% (4/22 correct), top-2 accuracy: 36.4%
- Mean Brier score: 0.114 (reasonable for 8 buckets, but with clear failure modes)
- Worst failures: Dec 29-Jan 4 (predicted 329, actual 570) and Feb 2-8 (predicted 468, actual 260)

---

## 1. Spec vs Reality: Preliminary Findings That Changed

The spec (Section "Key Validated Findings") was based on 15 days of data. Several findings reversed on the full 184-day dataset:

| Finding | Spec (15 days) | Full Dataset (184 days) | Impact |
|---|---|---|---|
| Lag-1 autocorrelation | **-0.15** (mean reversion) | **+0.43** (strong momentum) | Critical — the model should lean into momentum, not fight it |
| Daily eligible mean | 47.8 | 44.0 | Minor |
| CV | 46% | 54% | Higher variance than expected — wider distributions needed |
| Hourly peaks | Midnight-3am, 7-9am ET | **5-6am, 2-5pm ET** | Affects sleep cycle feature interpretation |
| Composition | 58% RT, 36% quote, 6% original | 58% RT, 37% quote, 5% original | Stable — good |

**The autocorrelation reversal is critical.** The spec assumed mild mean reversion; the full data shows strong positive autocorrelation out to lag-14 (all >0.21). This means high-volume days cluster together (regime persistence), and the model should weight recent observations much more heavily than it currently does.

---

## 2. Model Performance: Backtest Diagnosis

### 2a. Systematic Failures

The 22-period backtest reveals two distinct failure modes:

**Failure Mode 1: Slow to ramp up**
When volume shifted from ~200/week to ~550/week (Dec-Jan), the model's `recent_mu` (30-day trailing average) lagged badly:
- Dec 1-7: predicted 211, actual 369 (MISS by 158)
- Dec 29-Jan 4: predicted 329, actual 570 (MISS by 241)
- Jan 5-11: predicted 402, actual 588 (MISS by 186)

**Failure Mode 2: Slow to ramp down**
When volume dropped from ~550/week back to ~260/week (late Jan-Feb), the model overshot:
- Jan 26-Feb 1: predicted 503, actual 365 (MISS by 138)
- Feb 2-8: predicted 468, actual 260 (MISS by 208)

The model assigned only **0.5%** probability to the 570-tweet outcome and **0.1%** to the 260-tweet outcome. These are catastrophic calibration failures for a trading system.

### 2b. Calibration Analysis

The biggest calibration gap is in the 325-374 bucket: predicted 12.5% vs actual 31.8% (gap: 19.4%). The model systematically under-weights this bucket because it concentrates probability around its point estimate. The 325-374 bucket was the modal outcome (7/22 periods) but the model only predicted it as the top bucket 2 times.

### 2c. What Works

- When the regime is stable (Feb 16-Mar 8), the model performs well: 3 correct buckets in 4 periods, mean Brier 0.068.
- The day-of-week and flurry adjustments provide real (small) signal.
- The normal approximation for weekly totals is reasonable given CLT — the issue isn't the distribution shape, it's the location parameter (`mu`).

---

## 3. Spec Gaps: Unimplemented Features

### 3a. LLM Regime Classification (Spec Section 3c) — **Highest Priority**

The spec correctly identifies this as "a key alpha source." It remains unimplemented. Based on research:

**Recommended categories** (mutually exclusive per tweet):
- `POLITICAL_DOGE` — Government/policy commentary, DOGE work
- `TECH_PRODUCT` — xAI/Grok, Tesla, SpaceX announcements
- `MEME_SHITPOST` — Emojis, memes, low-effort reactions
- `COMBATIVE_FEUD` — Arguments, attacks, feuding
- `SIGNAL_BOOST` — Amplifying others' content with minimal commentary
- `PERSONAL_PHILOSOPHICAL` — Family, life, gaming

**Why this matters for volume prediction**: Different modes have very different "tweet velocity profiles." Meme/shitpost and signal-boost modes can produce 10-15 tweets in 30 minutes. Combative/feud mode is self-sustaining (each response generates more). The regime shift from ~35/day (Nov) to ~68/day (Jan) was almost certainly driven by a mode shift (likely POLITICAL_DOGE + COMBATIVE_FEUD), and the model had no way to detect it.

**Implementation approach**:
- Batch classify all tweets per day in a single Claude Haiku API call (~$0.02/day, ~$6 for historical backfill)
- Aggregate to daily composition features: `pct_political_doge`, `pct_combative_feud`, etc.
- Add regime transition features: `mode_changed`, `combative_streak`, `mode_stability_7d`
- Condition `predict_daily()` on detected regime instead of using a single `recent_mu`

### 3b. Market Period Alignment — **High Priority, Easy Fix**

The backtest uses ISO weeks (Mon-Sun), but Polymarket resolves Thursday 12pm ET to Thursday 12pm ET. This creates two problems:
1. Backtest results don't match actual trading periods
2. The model can't leverage the mid-week information advantage (by Thursday morning you've seen 6.5 of 7 days)

**Fix**: Parameterize period boundaries in `get_weekly_periods()` and use actual Polymarket resolution times.

### 3c. External Event Data (Spec Section 3b) — **Medium Priority**

The spec mentions:
- Scheduled events (Tesla earnings, SpaceX launches, DOGE hearings)
- Macro news flags
- Flight tracking data

None implemented. An events calendar would be the most practical starting point — even a simple binary `has_major_event` flag for known dates (earnings, launches) would help explain some regime shifts.

### 3d. Source Device Feature (Spec Section 3c) — **Low Priority, Quick Win**

The `source` field is collected but not used as a feature. Could indicate posting mode (phone = casual scrolling/RTing, web = more deliberate posting).

---

## 4. Code-Level Issues

### 4a. `BaseRateModel.fit()` — Regime Adjustment Design

**Problem**: The model uses a single `recent_mu` (30-day trailing average) with multiplicative day-of-week adjustments. This is fundamentally a single-regime model trying to handle multi-regime data.

**Evidence**: Monthly means range from 20.5 (Sep 2025) to 68.2 (Jan 2026) — a 3.3x spread. A 30-day window blends across regimes, producing a `recent_mu` that's wrong for both the old and new regime during transitions.

**Recommendation**: Replace the single `recent_mu` with a regime-conditional approach:
1. Detect current regime from recent tweet content (LLM classification) or posting velocity
2. Maintain separate `mu` estimates per regime
3. Weight by regime transition probability

A simpler intermediate fix: reduce the recency window from 30 days to 7-10 days and increase the autoregressive (lag-1) adjustment from the current dampened 0.5x to 0.7-0.8x. The strong positive autocorrelation (0.43) supports a more aggressive AR component.

### 4b. `BaseRateModel.predict_weekly()` — Independence Assumption

The model sums 7 independent daily negative binomials. But with lag-1 autocorrelation of 0.43, daily counts are **not independent**. This means:
- Weekly variance is underestimated (missing covariance terms)
- The model's distributions are too narrow
- This directly causes the "0.5% probability on actual outcome" failures

**Fix**: Add covariance terms to weekly variance:
```
weekly_var = sum(var_i) + 2 * sum(cov(i, j) for i < j)
```
where `cov(i, j) ≈ rho^|i-j| * sqrt(var_i * var_j)` using the empirical autocorrelation.

### 4c. `RealTimeAdjuster` — Observation Weight Curve

The `obs_weight = min(elapsed_fraction * 1.5, 0.95)` formula means that after 63% of the period (day 4.4), observations dominate at 95%. This is reasonable but could be improved:
- The weight should increase faster early in the period when the prior is poor (regime shift situations)
- Consider a beta-CDF-shaped weight curve instead of linear

### 4d. `features.py` — Missing Features

The daily feature matrix is missing several features the spec calls for:
- **Posting velocity trailing windows** (1h, 3h, 6h) — only `build_intraday_features()` computes these, not `build_daily_features()`
- **Content type mix as predictor** — `pct_retweet`, `pct_quote`, `pct_original` are computed but the model doesn't use them
- **"Fighting" flag** — `reply_rate` exists but no explicit feud detection logic
- **Engagement momentum** — no feature for whether engagement on recent posts is elevated

### 4e. `backtest.py` — Simulated P&L Methodology

The simulated P&L uses uniform market prices (12.5% per bucket) as the baseline. This is unrealistic — real Polymarket prices cluster around 2-3 buckets. The +91% simulated ROI is inflated because beating a uniform distribution is easy. A better baseline would use the model's own prior-week probabilities or historical bucket frequencies.

---

## 5. Data Quality Notes

### 5a. Strong Points
- 184 days of continuous data with no gaps >24h
- Composition ratios are stable across the period (58/37/5 RT/quote/original)
- 8,103 eligible tweets is sufficient for the feature set complexity

### 5b. Concerns
- **September 2025 partial data**: The first week (week 38) has only 54 tweets and 2 days. This should be excluded from training or at minimum flagged as partial.
- **No xtracker ground truth**: The spec calls for xtracker.polymarket.com validation of counts. If API counts differ from xtracker by even 2-3%, this compounds over a week to 15-20 tweet discrepancies — enough to shift bucket outcomes.
- **Weekly counts vary 54-588**: The 10x range suggests either the Sep partial data is distorting or there are genuine extreme regimes. The model needs to handle both.

---

## 6. Prioritized Recommendations

### Tier 1: Highest Impact (address first)

1. **Increase AR aggressiveness**: Change `adjustment * 0.5` to `adjustment * 0.75` in `predict_daily()`. The data strongly supports momentum (autocorrelation 0.43), and the model is under-leveraging it. This is a one-line change.

2. **Fix weekly variance calculation**: Add covariance terms to `predict_weekly()` to account for daily autocorrelation. This will widen distributions and avoid the "0.1% probability on actual outcome" failures.

3. **Reduce recency window**: Change from 30-day to 10-day trailing average in `fit()`. This makes `recent_mu` more responsive to regime shifts. Combined with #1, this addresses the core failure mode.

4. **Align backtest periods with Polymarket**: Switch from ISO weeks to Thursday-Thursday resolution windows in `get_weekly_periods()`.

### Tier 2: High Impact (next sprint)

5. **Implement LLM regime classification**: Batch-classify historical tweets, add composition features to `build_daily_features()`, add regime-conditional `mu` to `BaseRateModel`.

6. **Use content mix features**: The model computes `pct_retweet`, `pct_quote`, `pct_original` but doesn't use them in predictions. RT-heavy days are higher-count days (RTs are fast) — the composition encodes information about posting mode.

7. **Improve simulated P&L**: Use historical Polymarket prices or at minimum historical bucket frequencies as the market baseline instead of uniform distribution.

### Tier 3: Medium Impact (later)

8. **Events calendar**: Build a simple lookup of known events (earnings, launches, hearings) and add as binary features.

9. **Feud detection**: Formalize the "fighting flag" — elevated `reply_rate` + reply-to-specific-users patterns → binary `feud_active` feature.

10. **Source device feature**: Add `source` as a categorical feature (phone/web/other).

---

## 7. Quick Wins (< 30 min each)

- [ ] Exclude Sep 2025 partial week from training data
- [ ] Add `pct_retweet` and `pct_original` as features in `predict_daily()`
- [ ] Log-transform `eligible_count` before computing `recent_mu` to reduce outlier sensitivity
- [ ] Add a `regime_volatility` feature: rolling 7-day std / rolling 7-day mean (CV over trailing window)
- [ ] Cap the weekly distribution at the observed historical range (54-588) + 10% buffer, rather than allowing the normal approximation to put probability on impossible values like negative counts
