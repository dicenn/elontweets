"""
Phase 3b: LLM Tweet Classification Pipeline

Sends each day's tweets to Claude (Haiku) in batches and extracts a rich
multi-dimensional classification per tweet. Designed to run once for historical
backfill, then incrementally for new days.

Output schema per tweet:
  - mode: primary behavioral category
  - topics: specific subjects mentioned
  - intensity: emotional charge (1-5)
  - effort_level: low/medium/high
  - is_feud: whether directed at someone combatively
  - feud_target: who (if applicable)
  - is_reactive: responding to external event vs spontaneous
  - news_event: what triggered this (if reactive)
  - continuation_likelihood: will this lead to more tweets on same topic (1-5)
  - sentiment: negative/neutral/positive
  - is_thread_part: part of a multi-tweet thread
  - meme_or_emoji_only: low-content tweet (emoji, single word, meme)

Daily aggregate features are also computed for model consumption.
"""

import pandas as pd
import numpy as np
import json
import os
import time
import re
from pathlib import Path
from datetime import datetime

try:
    import anthropic
except ImportError:
    anthropic = None

DATA_DIR = Path(__file__).resolve().parent.parent / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis"

# Behavioral mode categories
MODES = [
    "POLITICAL_DOGE",
    "TECH_PRODUCT",
    "MEME_SHITPOST",
    "COMBATIVE_FEUD",
    "SIGNAL_BOOST",
    "PERSONAL_PHILOSOPHICAL",
]

CLASSIFICATION_PROMPT = """You are classifying Elon Musk's tweets for a volume prediction model.
We need to understand his behavioral modes to predict how many tweets he'll post.

Below are all of Elon Musk's tweets from a single day ({date}), in chronological order.
For context, here are the top news headlines from around this date:
{news_context}

For EACH tweet, return a JSON object with these fields:

- "tweet_id": the tweet ID (string)
- "mode": one of {modes}
  - POLITICAL_DOGE: government, policy, regulation, DOGE department, political commentary
  - TECH_PRODUCT: xAI, Grok, Tesla, SpaceX, Neuralink, Boring Company, product announcements
  - MEME_SHITPOST: memes, jokes, emojis, low-effort humor, shitposting, trolling
  - COMBATIVE_FEUD: arguments, attacks, feuding, calling people out, defensive responses
  - SIGNAL_BOOST: amplifying others' content (RTs, quotes with minimal commentary like "Exactly" or "True")
  - PERSONAL_PHILOSOPHICAL: family, gaming, life observations, philosophical musings
- "topics": array of 1-3 specific topics (e.g. ["DOGE", "government spending", "Vivek"])
- "intensity": 1-5 (1=calm/neutral, 5=extremely charged/emotional)
- "effort_level": "low" (emoji, RT, one word), "medium" (sentence or two), "high" (paragraph, thread, detailed)
- "is_feud": boolean - is this directed combatively at a specific person/org?
- "feud_target": string or null - who is the target if is_feud is true
- "is_reactive": boolean - is this responding to an external event/news?
- "news_event": string or null - brief description of what external event triggered this
- "continuation_likelihood": 1-5 (1=standalone, 5=very likely to spark more tweets on same topic)
- "sentiment": "negative", "neutral", or "positive"
- "is_thread_part": boolean - part of a multi-tweet thread or conversation
- "meme_or_emoji_only": boolean - tweet is just emoji(s), a single meme word, or minimal text

Return a JSON array of objects, one per tweet. Return ONLY the JSON array, no other text.

TWEETS FOR {date}:
{tweets_text}
"""


def load_all_tweets():
    """Load all tweets (both eligible and replies for full context)."""
    all_tweets = pd.read_csv(DATA_DIR / "musk_tweets_all_merged.csv")
    all_tweets["created_at"] = pd.to_datetime(all_tweets["created_at"])
    all_tweets["created_at_et"] = pd.to_datetime(all_tweets["created_at_et"], utc=True)
    all_tweets["date_et"] = all_tweets["created_at_et"].dt.date
    return all_tweets


def format_tweets_for_prompt(day_tweets):
    """Format a day's tweets for the LLM prompt."""
    lines = []
    for _, tweet in day_tweets.iterrows():
        tweet_id = tweet["tweet_id"]
        time_et = tweet["created_at_et"]
        if hasattr(time_et, "strftime"):
            time_str = time_et.strftime("%H:%M ET")
        else:
            time_str = str(time_et)
        tweet_type = tweet["tweet_type"]
        text = str(tweet["text"])[:500]

        lines.append(f"[{tweet_id}] ({time_str}, {tweet_type}) {text}")

    return "\n".join(lines)


def classify_day(client, date, day_tweets, news_context="No news context available."):
    """
    Send one day's tweets to Claude for classification.
    Returns list of classification dicts.
    """
    tweets_text = format_tweets_for_prompt(day_tweets)

    prompt = CLASSIFICATION_PROMPT.format(
        date=str(date),
        news_context=news_context,
        modes=json.dumps(MODES),
        tweets_text=tweets_text,
    )

    # Use Haiku for cost efficiency (~$0.02/day)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse JSON response
    raw_text = response.content[0].text.strip()

    # Handle markdown code blocks
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text)

    try:
        classifications = json.loads(raw_text)
    except json.JSONDecodeError:
        print(f"  WARNING: Failed to parse JSON for {date}, attempting repair...")
        # Try to extract JSON array from response
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if match:
            classifications = json.loads(match.group())
        else:
            print(f"  ERROR: Could not parse response for {date}")
            return []

    return classifications


def build_daily_classification_features(classifications_by_date):
    """
    Aggregate per-tweet classifications into daily features for the model.

    Returns a DataFrame with one row per date and columns:
    - mode composition: pct_political_doge, pct_tech_product, etc.
    - intensity stats: mean_intensity, max_intensity
    - feud stats: feud_count, feud_pct
    - reactive stats: reactive_pct, unique_news_events
    - effort distribution: pct_low_effort, pct_high_effort
    - continuation stats: mean_continuation, high_continuation_pct
    - sentiment distribution: pct_negative, pct_positive
    - meme stats: meme_only_pct
    - regime transition features: mode_changed, mode_stability_3d, dominant_mode_streak
    """
    rows = []

    sorted_dates = sorted(classifications_by_date.keys())
    prev_dominant_mode = None

    for date in sorted_dates:
        cls_list = classifications_by_date[date]
        n = len(cls_list)
        if n == 0:
            continue

        row = {"date": date, "n_classified": n}

        # Mode composition
        modes_today = [c.get("mode", "UNKNOWN") for c in cls_list]
        for mode in MODES:
            row[f"pct_{mode.lower()}"] = round(modes_today.count(mode) / n, 4)

        # Dominant mode
        from collections import Counter
        mode_counts = Counter(modes_today)
        dominant_mode = mode_counts.most_common(1)[0][0]
        row["dominant_mode"] = dominant_mode
        row["mode_concentration"] = round(mode_counts[dominant_mode] / n, 4)

        # Mode changed from yesterday?
        row["mode_changed"] = int(prev_dominant_mode is not None and dominant_mode != prev_dominant_mode)
        prev_dominant_mode = dominant_mode

        # Intensity
        intensities = [c.get("intensity", 3) for c in cls_list]
        row["mean_intensity"] = round(np.mean(intensities), 2)
        row["max_intensity"] = max(intensities)
        row["high_intensity_pct"] = round(sum(1 for i in intensities if i >= 4) / n, 4)

        # Feuds
        feuds = [c for c in cls_list if c.get("is_feud", False)]
        row["feud_count"] = len(feuds)
        row["feud_pct"] = round(len(feuds) / n, 4)
        feud_targets = list(set(c.get("feud_target", "") for c in feuds if c.get("feud_target")))
        row["n_feud_targets"] = len(feud_targets)
        row["feud_targets"] = json.dumps(feud_targets) if feud_targets else ""

        # Reactive/news
        reactive = [c for c in cls_list if c.get("is_reactive", False)]
        row["reactive_pct"] = round(len(reactive) / n, 4)
        news_events = list(set(
            c.get("news_event", "") for c in reactive
            if c.get("news_event")
        ))
        row["n_news_events"] = len(news_events)
        row["news_events"] = json.dumps(news_events) if news_events else ""

        # Effort
        efforts = [c.get("effort_level", "medium") for c in cls_list]
        row["pct_low_effort"] = round(efforts.count("low") / n, 4)
        row["pct_high_effort"] = round(efforts.count("high") / n, 4)

        # Continuation likelihood
        conts = [c.get("continuation_likelihood", 3) for c in cls_list]
        row["mean_continuation"] = round(np.mean(conts), 2)
        row["high_continuation_pct"] = round(sum(1 for c in conts if c >= 4) / n, 4)

        # Sentiment
        sentiments = [c.get("sentiment", "neutral") for c in cls_list]
        row["pct_negative"] = round(sentiments.count("negative") / n, 4)
        row["pct_positive"] = round(sentiments.count("positive") / n, 4)
        row["pct_neutral"] = round(sentiments.count("neutral") / n, 4)

        # Meme/emoji
        meme = [c for c in cls_list if c.get("meme_or_emoji_only", False)]
        row["meme_only_pct"] = round(len(meme) / n, 4)

        # Thread parts
        thread = [c for c in cls_list if c.get("is_thread_part", False)]
        row["thread_pct"] = round(len(thread) / n, 4)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add rolling regime features
    if len(df) > 3:
        # Mode stability: how often did the dominant mode change in last 3/7 days?
        df["mode_changes_3d"] = df["mode_changed"].rolling(3, min_periods=1).sum()
        df["mode_changes_7d"] = df["mode_changed"].rolling(7, min_periods=1).sum()

        # Rolling intensity
        df["intensity_rolling_3d"] = df["mean_intensity"].rolling(3, min_periods=1).mean().round(2)
        df["intensity_rolling_7d"] = df["mean_intensity"].rolling(7, min_periods=1).mean().round(2)

        # Rolling feud activity
        df["feud_pct_rolling_3d"] = df["feud_pct"].rolling(3, min_periods=1).mean().round(4)
        df["combative_pct_rolling_3d"] = df["pct_combative_feud"].rolling(3, min_periods=1).mean().round(4)

        # Rolling mode composition (for detecting regime shifts)
        for mode in MODES:
            col = f"pct_{mode.lower()}"
            df[f"{col}_rolling_7d"] = df[col].rolling(7, min_periods=1).mean().round(4)

    return df


def fetch_news_for_date(date_str):
    """
    Fetch top news headlines for a given date.
    Uses Google News RSS as a free, no-auth source.
    Falls back to a stub if unavailable.
    """
    try:
        import urllib.request
        import xml.etree.ElementTree as ET

        url = f"https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read().decode("utf-8")

        root = ET.fromstring(xml_data)
        items = root.findall(".//item")[:10]
        headlines = []
        for item in items:
            title = item.find("title")
            pub_date = item.find("pubDate")
            if title is not None:
                headlines.append(title.text)

        if headlines:
            return "\n".join(f"- {h}" for h in headlines[:8])
    except Exception as e:
        pass

    return "No news headlines available for this date."


def run_classification(api_key=None, start_date=None, end_date=None,
                       max_days=None, skip_existing=True, include_news=True):
    """
    Main classification pipeline.

    Args:
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        start_date: Start date (inclusive), defaults to earliest in data
        end_date: End date (inclusive), defaults to latest in data
        max_days: Maximum number of days to classify (for cost control)
        skip_existing: Skip dates already classified
        include_news: Whether to fetch news context
    """
    if anthropic is None:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        return

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: No API key. Set ANTHROPIC_API_KEY or pass api_key parameter.")
        return

    client = anthropic.Anthropic(api_key=api_key)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load existing classifications if any
    classifications_path = OUTPUT_DIR / "tweet_classifications.json"
    if classifications_path.exists() and skip_existing:
        with open(classifications_path) as f:
            all_classifications = json.load(f)
        print(f"Loaded {len(all_classifications)} existing classified days")
    else:
        all_classifications = {}

    # Load tweets
    print("Loading tweets...")
    all_tweets = load_all_tweets()
    dates = sorted(all_tweets["date_et"].unique())

    if start_date:
        start_date = pd.Timestamp(start_date).date()
        dates = [d for d in dates if d >= start_date]
    if end_date:
        end_date = pd.Timestamp(end_date).date()
        dates = [d for d in dates if d <= end_date]

    # Skip already classified dates
    if skip_existing:
        dates = [d for d in dates if str(d) not in all_classifications]

    if max_days:
        dates = dates[:max_days]

    print(f"Classifying {len(dates)} days ({len(all_tweets)} total tweets)")

    total_cost_est = len(dates) * 0.02  # ~$0.02/day with Haiku
    print(f"Estimated cost: ~${total_cost_est:.2f}")

    for i, date in enumerate(dates):
        day_tweets = all_tweets[all_tweets["date_et"] == date].sort_values("created_at")
        n_tweets = len(day_tweets)

        if n_tweets == 0:
            continue

        # Fetch news context if enabled
        news_context = "No news context available."
        if include_news:
            news_context = fetch_news_for_date(str(date))

        print(f"  [{i+1}/{len(dates)}] {date}: {n_tweets} tweets...", end=" ", flush=True)

        try:
            # For days with many tweets, batch into chunks of ~80
            if n_tweets > 80:
                all_cls = []
                for chunk_start in range(0, n_tweets, 80):
                    chunk = day_tweets.iloc[chunk_start:chunk_start + 80]
                    cls = classify_day(client, date, chunk, news_context)
                    all_cls.extend(cls)
                    if chunk_start + 80 < n_tweets:
                        time.sleep(0.5)  # Rate limit courtesy
                classifications = all_cls
            else:
                classifications = classify_day(client, date, day_tweets, news_context)

            all_classifications[str(date)] = classifications
            print(f"OK ({len(classifications)} classified)")

            # Save incrementally (crash-safe)
            with open(classifications_path, "w") as f:
                json.dump(all_classifications, f, indent=2, default=str)

        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(2)
            continue

        # Rate limit: ~1 request/sec for Haiku
        time.sleep(0.5)

    # Build aggregate features
    print("\nBuilding daily classification features...")
    daily_cls_features = build_daily_classification_features(all_classifications)
    features_path = OUTPUT_DIR / "daily_classification_features.csv"
    daily_cls_features.to_csv(features_path, index=False)
    print(f"Saved {len(daily_cls_features)} days of features to {features_path}")

    # Print summary
    if len(daily_cls_features) > 0:
        print("\n=== Classification Summary ===")
        for mode in MODES:
            col = f"pct_{mode.lower()}"
            if col in daily_cls_features.columns:
                mean_pct = daily_cls_features[col].mean()
                print(f"  {mode}: {mean_pct:.1%} avg daily share")

        print(f"\n  Mean intensity: {daily_cls_features['mean_intensity'].mean():.2f}")
        print(f"  Mean feud %: {daily_cls_features['feud_pct'].mean():.1%}")
        print(f"  Mean reactive %: {daily_cls_features['reactive_pct'].mean():.1%}")
        print(f"  Mean low-effort %: {daily_cls_features['pct_low_effort'].mean():.1%}")

    return all_classifications, daily_cls_features


def rebuild_features_from_existing():
    """
    Rebuild daily features from existing classifications (no API calls needed).
    Useful after tweaking the feature engineering logic.
    """
    classifications_path = OUTPUT_DIR / "tweet_classifications.json"
    if not classifications_path.exists():
        print("No classifications found. Run classify first.")
        return None

    with open(classifications_path) as f:
        all_classifications = json.load(f)

    print(f"Loaded {len(all_classifications)} classified days")
    daily_cls_features = build_daily_classification_features(all_classifications)
    features_path = OUTPUT_DIR / "daily_classification_features.csv"
    daily_cls_features.to_csv(features_path, index=False)
    print(f"Saved features to {features_path}")
    return daily_cls_features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify Elon tweets via LLM")
    parser.add_argument("--api-key", help="Anthropic API key")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-days", type=int, help="Max days to classify")
    parser.add_argument("--no-skip", action="store_true", help="Re-classify existing days")
    parser.add_argument("--no-news", action="store_true", help="Skip news context fetching")
    parser.add_argument("--rebuild-only", action="store_true",
                        help="Rebuild features from existing classifications (no API calls)")
    args = parser.parse_args()

    if args.rebuild_only:
        rebuild_features_from_existing()
    else:
        run_classification(
            api_key=args.api_key,
            start_date=args.start_date,
            end_date=args.end_date,
            max_days=args.max_days,
            skip_existing=not args.no_skip,
            include_news=not args.no_news,
        )
