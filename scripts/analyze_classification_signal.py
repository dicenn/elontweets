#!/usr/bin/env python3
"""Analyze whether LLM classification features predict weekly tweet volume."""

import pandas as pd
import numpy as np
from scipy import stats
import json
import sys

# Load data
clf = pd.read_csv("analysis/daily_classification_features.csv", parse_dates=["date"])
feats = pd.read_csv("analysis/daily_features.csv", parse_dates=["date"])

# Merge eligible_count from feats into clf
clf = clf.merge(feats[["date", "eligible_count"]], on="date", how="left")

# Load backtest periods for weekly alignment
with open("analysis/backtest_results.json") as f:
    bt = json.load(f)

periods = bt["periods"]

print("=" * 70)
print("CLASSIFICATION SIGNAL ANALYSIS")
print("=" * 70)

# ─── 1. Daily correlation: classification features vs daily eligible count ───
print("\n### 1. Daily Correlations: Classification Features vs Eligible Count\n")

daily_features = [
    "pct_political_doge", "pct_tech_product", "pct_meme_shitpost",
    "pct_combative_feud", "pct_signal_boost", "pct_personal_philosophical",
    "mean_intensity", "high_intensity_pct", "feud_pct",
    "pct_low_effort", "pct_high_effort", "pct_negative", "pct_positive",
    "reactive_pct", "mode_concentration"
]

valid = clf.dropna(subset=["eligible_count"])
print(f"{'Feature':<35} {'r':>8} {'p-value':>10} {'Direction':>12}")
print("-" * 70)

correlations = []
for feat in daily_features:
    if feat in valid.columns:
        mask = valid[feat].notna()
        if mask.sum() > 10:
            r, p = stats.pearsonr(valid.loc[mask, feat], valid.loc[mask, "eligible_count"])
            direction = "+" if r > 0 else "-"
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            correlations.append((feat, r, p, direction, sig))
            print(f"{feat:<35} {r:>8.3f} {p:>10.4f} {direction:>5} {sig}")

# ─── 2. Weekly aggregation: do classification features predict weekly volume? ───
print("\n\n### 2. Weekly Period Analysis\n")

weekly_rows = []
for period in periods:
    start = pd.Timestamp(period["period"].split(" to ")[0])
    end = pd.Timestamp(period["period"].split(" to ")[1])
    mask = (clf["date"] >= start) & (clf["date"] <= end)
    week_data = clf[mask]

    if len(week_data) == 0:
        continue

    row = {
        "period": period["period"],
        "actual_count": period["actual_count"],
        "actual_bucket": period["actual_bucket"],
        "n_days": len(week_data),
    }

    for feat in ["pct_political_doge", "pct_tech_product", "pct_meme_shitpost",
                  "pct_combative_feud", "pct_signal_boost", "pct_personal_philosophical",
                  "mean_intensity", "feud_pct", "pct_negative", "pct_positive",
                  "reactive_pct", "pct_low_effort", "mode_concentration"]:
        if feat in week_data.columns:
            row[f"mean_{feat}"] = week_data[feat].mean()

    weekly_rows.append(row)

weekly = pd.DataFrame(weekly_rows)

print(f"{'Feature':<40} {'r':>8} {'p-value':>10} {'Sig':>5}")
print("-" * 65)

weekly_feats = [c for c in weekly.columns if c.startswith("mean_")]
weekly_corrs = []
for feat in weekly_feats:
    mask = weekly[feat].notna()
    if mask.sum() > 5:
        r, p = stats.pearsonr(weekly.loc[mask, feat], weekly.loc[mask, "actual_count"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        weekly_corrs.append((feat, r, p, sig))
        print(f"{feat:<40} {r:>8.3f} {p:>10.4f} {sig:>5}")

# ─── 3. Leading indicator: does prior week's classification predict next week? ───
print("\n\n### 3. Leading Indicators: Prior Week → Next Week Volume\n")

weekly["next_count"] = weekly["actual_count"].shift(-1)
weekly["count_change"] = weekly["actual_count"].diff()
weekly["next_change"] = weekly["count_change"].shift(-1)

print(f"{'Feature (this week)':<40} {'r vs next_count':>15} {'p':>10}")
print("-" * 65)

for feat in weekly_feats:
    mask = weekly[feat].notna() & weekly["next_count"].notna()
    if mask.sum() > 5:
        r, p = stats.pearsonr(weekly.loc[mask, feat], weekly.loc[mask, "next_count"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if abs(r) > 0.3 or p < 0.1:
            print(f"{feat:<40} {r:>8.3f}        {p:>10.4f} {sig}")

# ─── 4. Regime shift detection: does mode composition predict volume spikes? ───
print("\n\n### 4. Regime Shift Analysis\n")

# Define high/low volume weeks
median_count = weekly["actual_count"].median()
weekly["high_vol"] = weekly["actual_count"] > median_count

print(f"Median weekly count: {median_count}")
print(f"High-volume weeks: {weekly['high_vol'].sum()}, Low-volume: {(~weekly['high_vol']).sum()}\n")

print(f"{'Feature':<40} {'Low-vol mean':>12} {'High-vol mean':>13} {'Diff':>8} {'t-stat':>8} {'p':>8}")
print("-" * 85)

for feat in weekly_feats:
    low = weekly.loc[~weekly["high_vol"], feat].dropna()
    high = weekly.loc[weekly["high_vol"], feat].dropna()
    if len(low) > 2 and len(high) > 2:
        t, p = stats.ttest_ind(low, high)
        diff = high.mean() - low.mean()
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if p < 0.15:
            print(f"{feat:<40} {low.mean():>12.4f} {high.mean():>13.4f} {diff:>+8.4f} {t:>8.2f} {p:>8.4f} {sig}")

# ─── 5. Day-over-day: does mode composition predict TOMORROW's volume? ───
print("\n\n### 5. Daily Lag-1: Today's Classification → Tomorrow's Volume\n")

valid = clf.copy()
valid["next_day_count"] = valid["eligible_count"].shift(-1)
valid = valid.dropna(subset=["next_day_count"])

print(f"{'Feature (today)':<35} {'r vs tomorrow':>13} {'p':>10} {'Sig':>5}")
print("-" * 65)

for feat in daily_features:
    if feat in valid.columns:
        mask = valid[feat].notna()
        if mask.sum() > 10:
            r, p = stats.pearsonr(valid.loc[mask, feat], valid.loc[mask, "next_day_count"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if abs(r) > 0.15 or p < 0.1:
                print(f"{feat:<35} {r:>8.3f}     {p:>10.4f} {sig}")

# ─── 6. Rolling features signal ───
print("\n\n### 6. Rolling Features (3d/7d) vs Current Day Volume\n")

rolling_feats = [c for c in clf.columns if "rolling" in c]
valid2 = clf.dropna(subset=["eligible_count"])

print(f"{'Rolling Feature':<45} {'r':>8} {'p':>10} {'Sig':>5}")
print("-" * 70)

for feat in rolling_feats:
    mask = valid2[feat].notna()
    if mask.sum() > 10:
        r, p = stats.pearsonr(valid2.loc[mask, feat], valid2.loc[mask, "eligible_count"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if abs(r) > 0.15 or p < 0.05:
            print(f"{feat:<45} {r:>8.3f} {p:>10.4f} {sig}")

# ─── 7. Dominant mode as categorical predictor ───
print("\n\n### 7. Volume by Dominant Mode\n")

valid3 = clf.dropna(subset=["eligible_count"])
mode_stats = valid3.groupby("dominant_mode")["eligible_count"].agg(["mean", "std", "count"])
mode_stats = mode_stats.sort_values("mean", ascending=False)

print(f"{'Dominant Mode':<30} {'Mean':>8} {'Std':>8} {'Days':>6}")
print("-" * 55)
for mode, row in mode_stats.iterrows():
    print(f"{mode:<30} {row['mean']:>8.1f} {row['std']:>8.1f} {int(row['count']):>6}")

# ANOVA
groups = [g["eligible_count"].values for _, g in valid3.groupby("dominant_mode")]
f_stat, p_val = stats.f_oneway(*groups)
print(f"\nANOVA F={f_stat:.2f}, p={p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

# ─── 8. Intensity as volume predictor ───
print("\n\n### 8. Intensity Terciles vs Volume\n")

valid3["intensity_tercile"] = pd.qcut(valid3["mean_intensity"], 3, labels=["low", "mid", "high"])
tercile_stats = valid3.groupby("intensity_tercile")["eligible_count"].agg(["mean", "std", "count"])
print(f"{'Tercile':<12} {'Mean':>8} {'Std':>8} {'Days':>6}")
print("-" * 36)
for t, row in tercile_stats.iterrows():
    print(f"{t:<12} {row['mean']:>8.1f} {row['std']:>8.1f} {int(row['count']):>6}")

# ─── 9. Combative/feud as spike predictor ───
print("\n\n### 9. Feud Activity vs Volume Spikes\n")

valid3["has_feud"] = valid3["feud_pct"] > 0.05
feud_yes = valid3.loc[valid3["has_feud"], "eligible_count"]
feud_no = valid3.loc[~valid3["has_feud"], "eligible_count"]
t_stat, p = stats.ttest_ind(feud_yes, feud_no)
print(f"Feud active (>5% tweets):   mean={feud_yes.mean():.1f}, n={len(feud_yes)}")
print(f"No feud:                     mean={feud_no.mean():.1f}, n={len(feud_no)}")
print(f"t={t_stat:.2f}, p={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")

# ─── 10. Summary ───
print("\n\n" + "=" * 70)
print("SUMMARY: STRONGEST SIGNALS")
print("=" * 70)

# Collect all significant daily correlations
print("\nDaily features correlated with eligible_count (p < 0.05):")
sig_daily = [(f, r, p, d, s) for f, r, p, d, s in correlations if p < 0.05]
sig_daily.sort(key=lambda x: abs(x[1]), reverse=True)
for f, r, p, d, s in sig_daily:
    print(f"  {d} {f}: r={r:.3f} (p={p:.4f}) {s}")

print("\nWeekly features correlated with actual_count (p < 0.1):")
sig_weekly = [(f, r, p, s) for f, r, p, s in weekly_corrs if p < 0.1]
sig_weekly.sort(key=lambda x: abs(x[1]), reverse=True)
for f, r, p, s in sig_weekly:
    print(f"  {f}: r={r:.3f} (p={p:.4f}) {s}")
