"""
Elon Musk Tweet Ingestion Script
Pulls tweets via X API v2 with batching, checkpointing, and classification.

Usage:
    python fetch_tweets.py --start 2026-03-15 --end 2026-03-22          # specific range
    python fetch_tweets.py --start 2026-03-15 --end 2026-03-22 --dry-run  # cost estimate only
    python fetch_tweets.py --resume                                      # resume interrupted pull
"""

import os
import sys
import json
import csv
import time
import argparse
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
ELON_USER_ID = "44196397"
TIMELINE_URL = f"https://api.twitter.com/2/users/{ELON_USER_ID}/tweets"
ARCHIVE_URL = "https://api.twitter.com/2/tweets/search/all"
COST_PER_TWEET = 0.005
MAX_RESULTS_PER_PAGE = 100
MAX_RESULTS_ARCHIVE = 500  # archive endpoint supports up to 500

TWEET_FIELDS = (
    "id,text,created_at,author_id,conversation_id,in_reply_to_user_id,"
    "public_metrics,referenced_tweets,entities,attachments,source,lang"
)

# Directories (relative to project root)
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DIR = PROJECT_DIR / "raw"
PROCESSED_DIR = PROJECT_DIR / "processed"
CHECKPOINT_FILE = PROJECT_DIR / "raw" / "_checkpoint.json"

# ── Auth ────────────────────────────────────────────────────────────────────
def get_bearer_token():
    """Get bearer token from env var or .env file."""
    token = os.environ.get("TWITTER_BEARER_TOKEN")
    if token:
        return token

    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("TWITTER_BEARER_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    print("ERROR: Set TWITTER_BEARER_TOKEN env var or create .env file")
    sys.exit(1)


# ── Classification ──────────────────────────────────────────────────────────
def classify_tweet(tweet: dict) -> dict:
    """Classify tweet type and resolution eligibility."""
    ref_tweets = tweet.get("referenced_tweets") or []
    ref_types = {r["type"] for r in ref_tweets}

    if "retweeted" in ref_types:
        tweet_type = "retweet"
    elif "quoted" in ref_types:
        tweet_type = "quote"
    elif "replied_to" in ref_types:
        tweet_type = "reply"
    else:
        tweet_type = "original"

    # Resolution-eligible: original, retweet, or quote (NOT reply)
    counts_for_resolution = tweet_type != "reply"

    # Self-reply detection (reply to own tweet = thread continuation)
    is_self_reply = (
        tweet_type == "reply"
        and tweet.get("in_reply_to_user_id") == ELON_USER_ID
    )

    return {
        "tweet_type": tweet_type,
        "counts_for_resolution": counts_for_resolution,
        "is_self_reply": is_self_reply,
    }


def convert_to_et(utc_str: str) -> str:
    """Convert ISO 8601 UTC timestamp to US Eastern time string."""
    from zoneinfo import ZoneInfo
    utc_dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    et_dt = utc_dt.astimezone(ZoneInfo("America/New_York"))
    return et_dt.isoformat()


def parse_tweet(tweet: dict) -> dict:
    """Parse raw API tweet into flat row for CSV."""
    classification = classify_tweet(tweet)
    metrics = tweet.get("public_metrics", {})

    return {
        "tweet_id": tweet["id"],
        "created_at": tweet["created_at"],
        "created_at_et": convert_to_et(tweet["created_at"]),
        "text": (tweet.get("text") or "")[:500],
        "tweet_type": classification["tweet_type"],
        "counts_for_resolution": classification["counts_for_resolution"],
        "in_reply_to_user_id": tweet.get("in_reply_to_user_id", ""),
        "conversation_id": tweet.get("conversation_id", ""),
        "is_self_reply": classification["is_self_reply"],
        "like_count": metrics.get("like_count", 0),
        "reply_count": metrics.get("reply_count", 0),
        "retweet_count": metrics.get("retweet_count", 0),
        "quote_count": metrics.get("quote_count", 0),
        "impression_count": metrics.get("impression_count", 0),
        "bookmark_count": metrics.get("bookmark_count", 0),
        "source": tweet.get("source", ""),
        "lang": tweet.get("lang", ""),
    }


# ── API Fetch ───────────────────────────────────────────────────────────────
def fetch_page(bearer_token: str, params: dict, url: str = None) -> dict:
    """Fetch one page of tweets from the API with retry logic."""
    headers = {"Authorization": f"Bearer {bearer_token}"}
    if url is None:
        url = TIMELINE_URL

    max_retries = 5
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params)

        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            # Rate limited — read reset header or use exponential backoff
            reset_time = resp.headers.get("x-rate-limit-reset")
            if reset_time:
                wait = max(int(reset_time) - int(time.time()), 1) + 2
            else:
                wait = (2 ** attempt) * 5
            print(f"  Rate limited. Waiting {wait}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)
        else:
            print(f"  API error {resp.status_code}: {resp.text}")
            if attempt < max_retries - 1:
                wait = (2 ** attempt) * 2
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print("  Max retries exceeded. Saving checkpoint.")
                raise Exception(f"API error {resp.status_code}: {resp.text}")

    raise Exception("Max retries exceeded")


def save_checkpoint(start_time: str, end_time: str, next_token: str,
                    pages_fetched: int, tweets_fetched: int):
    """Save progress so we can resume if interrupted."""
    checkpoint = {
        "start_time": start_time,
        "end_time": end_time,
        "next_token": next_token,
        "pages_fetched": pages_fetched,
        "tweets_fetched": tweets_fetched,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2))


def load_checkpoint() -> dict | None:
    """Load saved checkpoint if it exists."""
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return None


def clear_checkpoint():
    """Remove checkpoint file after successful completion."""
    if CHECKPOINT_FILE.exists():
        try:
            CHECKPOINT_FILE.unlink()
        except PermissionError:
            # Write empty checkpoint instead of deleting
            CHECKPOINT_FILE.write_text("{}")


# ── Main Fetch Logic ────────────────────────────────────────────────────────
def fetch_tweets(start_date: str, end_date: str, bearer_token: str,
                 dry_run: bool = False, resume: bool = False,
                 use_archive: bool = False):
    """
    Fetch tweets in a date range. Saves raw JSON per page + parsed CSV.

    Returns summary dict with counts and cost.
    """
    # Format dates for API (ISO 8601)
    start_time = f"{start_date}T00:00:00Z"
    end_time = f"{end_date}T00:00:00Z"

    # Dry run: estimate cost
    if dry_run:
        days = (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days
        est_tweets = days * 87  # ~87 tweets/day average from spec
        est_cost = est_tweets * COST_PER_TWEET
        print(f"\n  DRY RUN ESTIMATE")
        print(f"  Date range: {start_date} → {end_date} ({days} days)")
        print(f"  Est. tweets: ~{est_tweets:,}")
        print(f"  Est. cost:   ~${est_cost:.2f}")
        print(f"  (Based on ~87 tweets/day average)\n")
        return {"estimated_tweets": est_tweets, "estimated_cost": est_cost}

    # Resume from checkpoint?
    next_token = None
    pages_fetched = 0
    total_tweets = 0

    if resume:
        cp = load_checkpoint()
        if cp:
            next_token = cp["next_token"]
            pages_fetched = cp["pages_fetched"]
            total_tweets = cp["tweets_fetched"]
            start_time = cp["start_time"]
            end_time = cp["end_time"]
            print(f"  Resuming from checkpoint: page {pages_fetched}, {total_tweets} tweets so far")
        else:
            print("  No checkpoint found. Starting fresh.")

    # Ensure dirs exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_parsed = []
    batch_label = f"{start_date}_to_{end_date}"

    print(f"\n  Fetching tweets: {start_date} → {end_date}")
    print(f"  Saving raw JSON to: {RAW_DIR}/")

    # Select endpoint
    if use_archive:
        api_url = ARCHIVE_URL
        page_size = MAX_RESULTS_ARCHIVE
        print(f"  Using full-archive search endpoint (up to {page_size}/page)")
    else:
        api_url = TIMELINE_URL
        page_size = MAX_RESULTS_PER_PAGE
        print(f"  Using user timeline endpoint (up to {page_size}/page)")

    while True:
        params = {
            "max_results": page_size,
            "start_time": start_time,
            "end_time": end_time,
            "tweet.fields": TWEET_FIELDS,
        }
        if use_archive:
            params["query"] = "from:elonmusk"
        if next_token:
            params["pagination_token" if not use_archive else "next_token"] = next_token

        data = fetch_page(bearer_token, params, url=api_url)
        pages_fetched += 1

        # Save raw JSON
        raw_file = RAW_DIR / f"page_{batch_label}_{pages_fetched:04d}.json"
        raw_file.write_text(json.dumps(data, indent=2))

        tweets = data.get("data", [])
        page_count = len(tweets)
        total_tweets += page_count

        # Parse each tweet
        for tweet in tweets:
            all_parsed.append(parse_tweet(tweet))

        # Progress
        eligible = sum(1 for t in all_parsed if t["counts_for_resolution"])
        cost_so_far = total_tweets * COST_PER_TWEET
        print(f"  Page {pages_fetched}: +{page_count} tweets "
              f"(total: {total_tweets}, eligible: {eligible}, cost: ${cost_so_far:.2f})")

        # Check for more pages
        next_token = data.get("meta", {}).get("next_token")
        if not next_token:
            print(f"\n  DONE. Total: {total_tweets} tweets across {pages_fetched} pages")
            break

        # Save checkpoint after each page
        save_checkpoint(start_time, end_time, next_token, pages_fetched, total_tweets)

        # Polite pause between requests
        time.sleep(1.2)

    # ── Write CSVs ──────────────────────────────────────────────────────
    if all_parsed:
        # All tweets
        all_csv = PROCESSED_DIR / f"musk_tweets_all_{batch_label}.csv"
        fieldnames = list(all_parsed[0].keys())
        with open(all_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_parsed)
        print(f"  Saved all tweets:      {all_csv}")

        # Eligible only
        eligible_rows = [t for t in all_parsed if t["counts_for_resolution"]]
        elig_csv = PROCESSED_DIR / f"musk_tweets_eligible_{batch_label}.csv"
        with open(elig_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(eligible_rows)
        print(f"  Saved eligible tweets: {elig_csv}")

    clear_checkpoint()

    # ── Summary ─────────────────────────────────────────────────────────
    summary = {
        "date_range": f"{start_date} → {end_date}",
        "total_tweets": total_tweets,
        "pages": pages_fetched,
        "cost": total_tweets * COST_PER_TWEET,
        "by_type": {},
        "eligible": 0,
    }
    for t in all_parsed:
        tt = t["tweet_type"]
        summary["by_type"][tt] = summary["by_type"].get(tt, 0) + 1
        if t["counts_for_resolution"]:
            summary["eligible"] += 1

    print(f"\n  ── Summary ──")
    print(f"  Total tweets:    {summary['total_tweets']}")
    print(f"  Eligible:        {summary['eligible']}")
    print(f"  By type:         {summary['by_type']}")
    print(f"  Cost:            ${summary['cost']:.2f}")

    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Elon Musk tweets from X API v2")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost without fetching")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--archive", action="store_true", help="Use full-archive search endpoint")

    args = parser.parse_args()

    if not args.resume and (not args.start or not args.end):
        parser.error("--start and --end required (unless --resume)")

    token = get_bearer_token()

    result = fetch_tweets(
        start_date=args.start or "",
        end_date=args.end or "",
        bearer_token=token,
        dry_run=args.dry_run,
        resume=args.resume,
        use_archive=args.archive,
    )
