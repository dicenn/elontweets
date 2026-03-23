# Elon Musk Tweet Prediction Model — Build Spec

## Objective

Build a predictive model to trade the weekly "How many tweets will Elon Musk post?" markets on Polymarket. The model should produce a probability distribution over market buckets for each period, enabling identification of mispriced buckets.

---

## Market Structure

- **Resolution source**: xtracker.polymarket.com ("Post Counter" figure)
- **Resolution window**: Weekly, typically Thursday 12:00 PM ET to following Thursday 12:00 PM ET (varies — always check current market)
- **What counts**: Main feed posts (originals), quote tweets, and reposts (retweets)
- **What doesn't count**: Replies (except replies appearing on main feed, which the tracker captures), community reposts not captured by tracker
- **Deleted posts**: Count if captured by tracker within ~5 minutes
- **Secondary resolution source**: x.com/elonmusk (if tracker fails)

---

## Phase 1: Data Ingestion

### 1a. Historical Pull via X API v2

**Endpoint**: `GET /2/users/44196397/tweets` (Elon's user ID)

**Auth**: Bearer token (user has one with credits loaded)

**Target**: 6 months of history (~Sep 20, 2025 → present). At minimum, Feb 4, 2026 → present (user has validation data from Feb 4 onward).

**Cost**: ~$0.005/tweet. At ~87 tweets/day × 180 days = ~15,600 tweets = ~$78 for 6 months. ~$20 for Feb 4 → present.

**Tweet fields to request**:
```
tweet.fields=id,text,created_at,author_id,conversation_id,in_reply_to_user_id,
public_metrics,referenced_tweets,entities,attachments,source,lang
```

**Approach**:
- Pull ALL tweet types (do NOT use `exclude` parameter — we want everything including replies for behavioral analysis)
- Paginate using `pagination_token`, 100 tweets per request
- Use `start_time` and `end_time` for date bounds
- Handle 429 rate limits with exponential backoff
- Save raw JSON responses verbatim before any processing
- Build checkpoint/resume capability (save pagination token on each page)

**Storage**:
- `/raw/` — Raw JSON API responses, one file per page
- `/processed/musk_tweets_all.csv` — Parsed flat CSV of all tweets
- `/processed/musk_tweets_eligible.csv` — Filtered to resolution-eligible only

**Classification logic**:
```
referenced_tweets field:
  - Empty/null → "original"
  - Contains type "replied_to" → "reply"
  - Contains type "retweeted" → "retweet"
  - Contains type "quoted" → "quote"

Resolution-eligible = original OR retweet OR quote (NOT reply)
```

**Parsed fields per tweet**:

| Field | Source | Notes |
|---|---|---|
| tweet_id | id | |
| created_at | created_at | ISO 8601 UTC |
| created_at_et | derived | Convert to ET for market alignment |
| text | text | Truncate to 500 chars for storage |
| tweet_type | derived | original/retweet/quote/reply |
| counts_for_resolution | derived | Boolean |
| in_reply_to_user_id | in_reply_to_user_id | For "fighting" detection |
| conversation_id | conversation_id | For thread detection |
| is_self_reply | derived | in_reply_to_user_id == Elon's ID |
| like_count | public_metrics | |
| reply_count | public_metrics | |
| retweet_count | public_metrics | |
| quote_count | public_metrics | |
| impression_count | public_metrics | |
| source | source | Device/app used |
| lang | lang | |

### 1b. Validation

- User has daily eligible counts from Feb 4 → present in Excel (pulled from official API)
- After pull, compare daily eligible counts against user's ground truth
- Acceptable tolerance: ±2 tweets/day (deleted tweet noise)
- Also cross-reference against xtracker.polymarket.com for market periods where available

### 1c. xtracker True-Up

- Periodically pull the "Post Counter" from xtracker.polymarket.com for active market periods
- This is the actual resolution number — use to calibrate any systematic drift between our count and the market's count
- Especially important near period end when exact count matters for bucket assignment

---

## Phase 2: Exploratory Data Analysis

Run on the full dataset once pulled. Key analyses:

### 2a. Distribution Analysis
- Daily eligible count distribution (mean, median, stdev, skewness, min/max)
- Coefficient of variation (we saw 46% on 15 days of data — validate on full set)
- Check for normality vs heavy tails

### 2b. Temporal Patterns
- **Hourly pattern (ET)**: We found bimodal — late night peak (midnight–3am) and morning peak (7–9am)
- **Day-of-week**: Need more data to validate (only had 2 samples/day before)
- **Weekend vs weekday**: Does the DOGE role create a weekday pattern?

### 2c. Autocorrelation
- Daily lag-1, lag-2, lag-3 autocorrelation (preliminary finding: slightly negative lag-1 at -0.15, suggesting mean reversion)
- Hourly autocorrelation for intraday momentum

### 2d. Tweet Type Composition
- Daily mix of original/quote/retweet (we saw 58% RT, 36% quote, 6% original)
- Does composition shift predict total count?
- RT-heavy days → higher raw counts (retweets are fast)

### 2e. Flurry Detection
- Define flurry: hourly rate > 2x trailing 6-hour average AND count ≥ 5
- Characterize flurries: frequency, duration, magnitude, time-of-day clustering
- Flurry triggers: can we identify what precedes them?

### 2f. Slope Shift Detection
- The cumulative tweet chart shows two patterns:
  1. **Flurries**: Sudden spikes in cumulative chart, then revert to baseline
  2. **Slope shifts**: Baseline posting rate changes (e.g., from 35/day to 55/day)
- Slope shifts correlate with regime changes — detect and characterize

### 2g. Inter-Tweet Gap Analysis
- Gap distribution (we saw: 61% under 5min, 8% over 2hrs — very on/off)
- Gap patterns as predictors of remaining daily count

### 2h. Engagement Analysis
- Engagement by tweet type (originals get ~110K likes, quotes ~97K, RTs ~0)
- Does high-engagement content predict sustained posting?

---

## Phase 3: Feature Engineering

### 3a. Structural Variables (known in advance)
- Day of week
- Hour of day (ET)
- Is weekend
- Days until market period end

### 3b. Regime/Context Variables (external data)
- **Location proxy**: Flight tracking data (user will source). Key question: does he tweet more in the air? Does destination predict posting regime?
- **Scheduled events**: Tesla earnings dates, SpaceX launch windows, DOGE hearings, speaking engagements, product unveilings. Maintain a calendar.
- **Macro news flags**: Political controversies, X/platform drama, major news events. Could use news API or manual tagging.

### 3c. Behavioral Variables (derived from tweet data)
- **LLM regime classification**: Apply an LLM to tweet content to classify the current "mode" — political, tech/product, meme/humor, personal, combative/feud. This is a key alpha source since originals and quotes carry the actual behavioral signal.
- **"Fighting" flag**: Elevated reply-to-others rate, especially to specific accounts. When engaged in a feud, posting rate increases.
- **Reply/engagement momentum**: If his posts are getting 10x normal quote tweets, he's likely to keep posting.
- **Sleep cycle estimation**: Time of last tweet before gap, time of first tweet after gap. Late-night posting (3am ET) signals elevated next-day count.
- **Content type mix**: Meme-repost days produce higher raw counts; policy thread days produce sustained elevated rates.
- **Posting velocity**: Tweets per hour in trailing 1h, 3h, 6h windows. Sharp acceleration = flurry in progress.
- **Source device**: "Twitter for iPhone" vs web vs other — different devices may correlate with different posting patterns.

### 3d. Market-Aware Variables
- Current period elapsed time
- Current period tweet count so far
- Implied posting rate for remainder to hit each bucket
- Time decay: how bucket likelihoods shift as period elapses with no activity change

---

## Phase 4: Model Architecture

Three-layer model:

### Layer 1: Base Rate Model
- Estimates expected daily eligible posting rate given current regime
- Inputs: structural variables, regime/context variables, behavioral variables
- Output: predicted daily rate (μ) and variance (σ²)
- Approach: Start with Poisson regression or negative binomial (count data with overdispersion), potentially upgrade to gradient boosting

### Layer 2: Real-Time Adjustment Layer
- Updates forecast within a market period using observed data
- Inputs: tweets so far this period, posting velocity, flurry detection, time of day, time remaining
- Output: conditional distribution of final period count given what's been observed
- Approach: Bayesian updating — prior from Layer 1, posterior updated with each new observation
- Key insight: flurries add variance (widen distribution), slope shifts move the central estimate

### Layer 3: Market Translation Layer
- Converts probability distribution over final counts into trading positions
- Inputs: model's distribution, market's implied probabilities (from Polymarket prices)
- Output: expected value per bucket, position sizing
- Logic: where model probability > market implied probability by sufficient margin (edge threshold), take position
- Must account for: market fees, liquidity, slippage

---

## Phase 5: Backtest & Validation

- Walk-forward backtest on historical market periods
- For each past market period:
  1. Run model with only data available at that point in time
  2. Compare model's distribution to actual resolution
  3. Score calibration (are 70% confidence intervals right 70% of the time?)
  4. Score profitability against historical market prices (if available from Polymarket API)
- Key metric: Brier score for probability calibration, simulated P&L for trading viability

---

## Technical Notes

### API Credentials
- **X API bearer token**: `AAAAAAAAAAAAAAAAAAAAAIQy8QEAAAAA6RMUnlNl4q%2F0My%2BEF0yZ64t0hqA%3DAuvT0VtHSqSsG9B1GoEaW4FwpdP9dDl933suIHypx6alqE4l6T`
- **Elon user ID**: 44196397
- Cost: $0.005/tweet pull

### Existing Data
- 15 days of validated X API data (March 5–20, 2026): 1,298 tweets, 765 eligible
- User has Excel data with daily eligible counts from Feb 4 → present
- User has data in GitHub repo: `github.com/dicenn/sandbox` branch `claude`, path `elon-tweets-jet-analysis-vEdra/data/`

### Key Validated Findings (from initial 15-day EDA)
- Daily eligible mean: 47.8, stdev: 21.9, CV: 46%
- Hourly pattern: bimodal (midnight–3am ET peak, 7–9am ET peak)
- Autocorrelation: slightly negative lag-1 (-0.15) — mild mean reversion
- Composition: 58% RT, 36% quote, 6% original
- Inter-tweet gaps: 61% under 5 min (burst posting pattern)
- Flurry frequency: ~1-2 per day, biggest spikes 10x trailing average

### Build Priority
1. Data pull (X API, 6 months) — get this running first
2. EDA on full dataset — validate preliminary findings
3. Feature engineering + LLM regime classification
4. Base rate model
5. Real-time adjustment layer
6. Market translation + backtest
