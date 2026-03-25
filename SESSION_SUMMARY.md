# Elon Tweet Prediction System — Session Summary

**Date**: March 25, 2026
**Branch**: `claude/review-spec-tweets-j4k7B`
**Repo**: `dicenn/elontweets`

---

## What This Project Is

A system to predict weekly Elon Musk tweet volume for trading Polymarket's "How many tweets will Elon post?" market. Polymarket offers buckets (0-174, 175-224, 225-274, ... 475+) and you bet on which bucket the actual count falls in.

---

## What Was Built (in order)

### Phase 1: Data Collection
- **Script**: `scripts/fetch_tweets.py`
- Pulled 6 months of Elon's tweets from X API (Sep 20, 2025 → Mar 23, 2026)
- **8,103 eligible tweets** across 184 days
- Stored in `processed/` as CSVs (eligible vs all tweets, by date range, plus merged files)
- Tracks: tweet type (RT/quote/original), timestamps, engagement, source device

### Phase 2: Exploratory Data Analysis
- **Script**: `scripts/eda.py` → `analysis/eda_report.json`
- Key findings:
  - Daily eligible count: mean 44, stdev 23.8, CV 54%
  - **Strong positive autocorrelation** (lag-1 = 0.43) — high-volume days cluster together
  - Bimodal posting pattern: peaks at 5-6am and 2-5pm ET
  - Composition stable: 58% RT, 37% quote, 5% original
  - 61% of inter-tweet gaps under 5 minutes (burst posting)

### Phase 3: Feature Engineering
- **Script**: `scripts/features.py` → `analysis/daily_features.csv`
- 46 daily features: day-of-week, posting velocity, gap patterns, reply rates, rolling averages, engagement metrics, flurry detection

### Phase 3b: LLM Tweet Classification (NEW — this session)
- **Script**: `scripts/classify_tweets.py` → `analysis/tweet_classifications.json` + `analysis/daily_classification_features.csv`
- Used Claude Haiku to classify every tweet into one of 6 behavioral modes:
  - **POLITICAL_DOGE** (9.1% avg) — government/policy, DOGE work
  - **TECH_PRODUCT** (16.6%) — xAI/Grok, Tesla, SpaceX
  - **MEME_SHITPOST** (16.9%) — memes, emojis, low-effort reactions
  - **COMBATIVE_FEUD** (5.7%) — arguments, attacks, feuding
  - **SIGNAL_BOOST** (46.4%) — amplifying others' content
  - **PERSONAL_PHILOSOPHICAL** (5.4%) — family, life, gaming
- Also tags: intensity (1-5), sentiment, effort level, feud targets, news events
- 183/184 days classified (one API error on Jan 29)
- 42 derived features including rolling 3d/7d averages

### Phase 4: Prediction Model
- **Script**: `scripts/model.py`
- Three-layer architecture:
  1. **Base Rate Model** — negative binomial regression for daily count prediction using 30-day trailing average with day-of-week adjustments
  2. **Real-Time Adjuster** — Bayesian updating within a market period as days are observed
  3. **Market Translator** — converts probability distributions into Polymarket position sizing

### Phase 5: Backtesting
- **Script**: `scripts/backtest.py` → `analysis/backtest_results.json`
- Walk-forward backtest on 22 weekly periods (Oct 2025 → Mar 2026)
- Results:
  - Bucket accuracy: 36.4% (8/22 correct)
  - Top-2 accuracy: 50%
  - Mean Brier score: 0.10 (decent for 8 buckets)
  - Simulated ROI vs uniform baseline: +127%
- **Main weakness**: slow to adapt to regime shifts (the Dec-Jan volume spike from ~200/wk to ~570/wk, and the Feb drop back to ~260/wk)

### Phase 6: Spec Review (this session)
- **File**: `analysis/spec-review.md`
- Comprehensive review identifying:
  - The original 15-day analysis was wrong about autocorrelation (assumed mean-reversion, data shows momentum)
  - Model's core failure: single-regime approach can't handle 3.3x volume swings
  - Weekly variance underestimated (ignores daily autocorrelation covariance)
  - Prioritized 10 recommendations in 3 tiers

### Phase 7: Classification Signal Analysis (this session)
- **Script**: `scripts/analyze_classification_signal.py`
- Tested whether classification features predict weekly volume

---

## Signal Analysis Findings

### What's Statistically Significant

**Daily level (n=183 days):**
| Feature | Correlation with daily volume | p-value |
|---|---|---|
| pct_political_doge | +0.33 | <0.0001 |
| pct_negative (sentiment) | +0.31 | <0.0001 |
| pct_signal_boost | +0.30 | <0.0001 |
| mean_intensity | +0.29 | 0.0001 |
| pct_meme_shitpost | -0.29 | 0.0001 |
| mode_concentration | +0.28 | 0.0001 |

**Weekly level (n=22 weeks):**
| Feature | Correlation with weekly volume | p-value |
|---|---|---|
| pct_meme_shitpost | -0.70 | 0.0003 |
| pct_political_doge | +0.69 | 0.0004 |
| pct_positive sentiment | -0.68 | 0.0006 |
| mean_intensity | +0.67 | 0.0007 |

**Leading indicators (this week → next week, n=21):**
| This week's feature | Correlation with NEXT week's volume | p-value |
|---|---|---|
| pct_political_doge | +0.77 | <0.0001 |
| mean_intensity | +0.76 | 0.0001 |
| pct_negative | +0.75 | 0.0001 |

### Honest Assessment of Edge

**The good:**
- Daily signals (n=183) are robust — political mode and intensity genuinely predict higher volume
- Rolling 7-day averages (r=0.40-0.45) are strong with good sample size
- Classification captures "regime identity" (political mode vs chill mode) which is the model's #1 blind spot
- Content changes likely lead volume changes by 1-2 days (early warning of regime shifts)

**The caveats:**
- Weekly correlations (r=0.7+) are inflated — only 22 data points, 13 features tested, one dominant regime transition driving most of the signal
- Leading indicators partially redundant with volume autocorrelation (volume already predicts volume; classification may just be a proxy)
- Real incremental R² over existing volume-based model likely **5-10%**, not the 40-50% raw correlations suggest
- Feud detection specifically is NOT useful (p=0.84)

**The key unanswered question:**
Does classification add prediction power BEYOND what you already get from just looking at recent volume? Need to run residual analysis (correlate classification features against model prediction errors) to know for sure.

### What the data suggests the model should do:
- When 7-day `pct_political_doge` > 10% AND `mean_intensity` > 1.7 → shift to high-volume regime estimate
- Use `intensity_rolling_7d` and `pct_political_doge_rolling_7d` as additional features
- These would most help at regime transition points (catching the turn 1-2 days faster)

---

## File Map

```
elontweets/
├── elon-tweet-model-spec.md          # Original build specification
├── scripts/
│   ├── fetch_tweets.py               # Phase 1: X API ingestion
│   ├── eda.py                        # Phase 2: Exploratory analysis
│   ├── features.py                   # Phase 3: Feature engineering
│   ├── classify_tweets.py            # Phase 3b: LLM classification
│   ├── model.py                      # Phase 4: Three-layer prediction model
│   ├── backtest.py                   # Phase 5: Walk-forward backtesting
│   └── analyze_classification_signal.py  # Phase 7: Signal analysis
├── processed/                        # Raw + eligible tweet CSVs
│   ├── musk_tweets_eligible_merged.csv
│   └── musk_tweets_all_merged.csv
└── analysis/
    ├── daily_features.csv            # 46 structural daily features
    ├── daily_classification_features.csv  # 42 LLM classification features
    ├── tweet_classifications.json    # Per-tweet classification details
    ├── eda_report.json               # EDA findings
    ├── backtest_results.json         # 22-period backtest results
    └── spec-review.md                # Comprehensive system review
```

---

## Next Steps (if continuing)

1. **Residual analysis** — test if classification adds signal beyond volume momentum
2. **Implement model fixes** from spec-review Tier 1 (AR aggressiveness, weekly variance, shorter recency window)
3. **Integrate classification features** into `model.py`
4. **Align backtest periods** with actual Polymarket Thursday-Thursday windows
5. **Re-run backtest** to measure improvement
