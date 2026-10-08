"""
Phase 2 — Data Cleaning & Feature Engineering
Reads data/raw/mls_matches_raw.csv → writes data/processed/mls_matches_clean.csv
Also produces data/processed/model_features.csv and data/processed/data_dictionary.md
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import (
    build_rivalry_flag,
    build_season_stage,
    build_rolling_form,
    build_away_team_quality,
    build_days_since_last_home,
    build_market_size_tier,
    build_weather_flags,
    build_away_team_draw,
    build_alternate_venue_flag,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "mls_matches_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "mls_matches_clean.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "model_features.csv"
FLAGGED_PATH = ROOT / "data" / "processed" / "flagged_rows.csv"
DICT_PATH = ROOT / "data" / "processed" / "data_dictionary.md"
(ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
flagged_log = []


def _flag(df_subset: pd.DataFrame, reason: str):
    sub = df_subset.copy()
    sub["removal_reason"] = reason
    flagged_log.append(sub)


def log(msg: str):
    print(f"  {msg}")


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------
def clean():
    print("=" * 60)
    print("MLS Attendance Forecasting — Data Cleaning")
    print("=" * 60)

    df = pd.read_csv(RAW_PATH, low_memory=False)
    shape_before = df.shape
    log(f"Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ------------------------------------------------------------------ #
    # Step 1: Deduplication
    # ------------------------------------------------------------------ #
    dup_mask = df.duplicated(subset=["match_id"], keep="first")
    n_dups = dup_mask.sum()
    if n_dups:
        _flag(df[dup_mask], "duplicate_match_id")
        df = df[~dup_mask].copy()
        log(f"Step 1 — Removed {n_dups} duplicate match_id rows")
    else:
        log("Step 1 — No duplicate match_ids found")

    # ------------------------------------------------------------------ #
    # Step 2: COVID handling (CRITICAL — before any other filtering)
    # ------------------------------------------------------------------ #
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    # Remove 2020 entirely (COVID bubble, no fans)
    mask_2020 = df["season"] == 2020
    n_2020 = mask_2020.sum()
    _flag(df[mask_2020], "covid_bubble_2020")
    df = df[~mask_2020].copy()
    log(f"Step 2a — Removed {n_2020} rows for 2020 COVID bubble season")

    # Flag 2021 as capacity-restricted (keep records)
    df["is_capacity_restricted"] = (df["season"] == 2021).astype(int)
    log(f"Step 2b — Flagged {(df['is_capacity_restricted']==1).sum()} rows as is_capacity_restricted=1 (2021 season)")

    # Attendance must be numeric
    df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")

    # Remove matches with no reported attendance (the target cannot be imputed)
    mask_missing = df["attendance"].isna()
    n_missing = mask_missing.sum()
    if n_missing:
        _flag(df[mask_missing], "missing_attendance")
        df = df[~mask_missing].copy()
        log(f"Step 2c — Removed {n_missing} rows with no reported attendance")

    # Remove suspected data errors (attendance < 2,000)
    mask_low = df["attendance"] < 2000
    n_low = mask_low.sum()
    if n_low:
        _flag(df[mask_low], "suspected_data_error")
        df = df[~mask_low].copy()
        log(f"Step 2c — Removed {n_low} rows with attendance < 2,000 (suspected data errors)")

    # Attendance above the listed capacity is NOT an error and is kept as reported:
    # listed capacities are often the reduced soccer configuration of a larger venue
    # (e.g. Lumen Field, Mercedes-Benz Stadium), which clubs open up for big matches.
    if "stadium_capacity" in df.columns:
        df["stadium_capacity"] = pd.to_numeric(df["stadium_capacity"], errors="coerce").fillna(25000)
        n_over = int((df["attendance"] > df["stadium_capacity"] * 1.05).sum())
        log(f"Step 2d — {n_over} rows have attendance > 105% of listed capacity (kept as reported)")

    # Check expansion team debut seasons
    if "mls_debut_season" in df.columns and "home_team" in df.columns:
        df["mls_debut_season"] = pd.to_numeric(df["mls_debut_season"], errors="coerce")
        bad_expansion = df[df["season"] < df["mls_debut_season"]]
        if len(bad_expansion):
            log(f"  [WARN] {len(bad_expansion)} records found before team's debut season — possible join error. Removing.")
            _flag(bad_expansion, "before_debut_season")
            df = df[df["season"] >= df["mls_debut_season"]].copy()

    # ------------------------------------------------------------------ #
    # Step 3: Date parsing & derived date features
    # ------------------------------------------------------------------ #
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon, 6=Sun
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # days_since_season_start (per season)
    season_starts = df.groupby("season")["date"].transform("min")
    df["days_since_season_start"] = (df["date"] - season_starts).dt.days

    log("Step 3 — Parsed dates; added day_of_week, month, is_weekend, days_since_season_start")

    # ------------------------------------------------------------------ #
    # Step 4: Weather normalization
    # ------------------------------------------------------------------ #
    # Compute monthly city averages for imputation
    if "temperature_max" in df.columns:
        df["temperature_max"] = pd.to_numeric(df["temperature_max"], errors="coerce")
        n_missing_temp = int(df["temperature_max"].isna().sum())
        city_monthly_avg = df.groupby(["home_team", "month"])["temperature_max"].transform("mean")
        monthly_avg = df.groupby("month")["temperature_max"].transform("mean")
        df["temperature_max"] = (
            df["temperature_max"].fillna(city_monthly_avg).fillna(monthly_avg).fillna(20.0).round(1)
        )
        if n_missing_temp:
            log(f"  Imputed {n_missing_temp} missing temperatures with the home city's monthly average")
    else:
        df["temperature_max"] = 20.0

    if "precipitation_sum" in df.columns:
        df["precipitation_sum"] = pd.to_numeric(df["precipitation_sum"], errors="coerce").fillna(0)
    else:
        df["precipitation_sum"] = 0.0

    if "windspeed_max" in df.columns:
        df["windspeed_max"] = pd.to_numeric(df["windspeed_max"], errors="coerce").fillna(0)
    else:
        df["windspeed_max"] = 0.0

    df = build_weather_flags(df)
    log("Step 4 — Normalized weather; added is_rainy, is_extreme_heat, is_cold")

    # ------------------------------------------------------------------ #
    # Step 5: Derived analytical features
    # ------------------------------------------------------------------ #
    # Ensure required columns exist
    if "stadium_capacity" not in df.columns or df["stadium_capacity"].isna().all():
        df["stadium_capacity"] = 25000

    df["attendance_pct"] = (df["attendance"] / df["stadium_capacity"]).clip(0, 1.05).round(4)

    if "avg_ticket_price_usd" not in df.columns:
        df["avg_ticket_price_usd"] = 55.0
    df["avg_ticket_price_usd"] = pd.to_numeric(df["avg_ticket_price_usd"], errors="coerce").fillna(55.0)
    df["estimated_revenue"] = (df["attendance"] * df["avg_ticket_price_usd"]).round(0)

    if "home_goals" not in df.columns:
        df["home_goals"] = 0
    if "away_goals" not in df.columns:
        df["away_goals"] = 0
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").fillna(0).astype(int)
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").fillna(0).astype(int)
    df["goal_diff"] = df["home_goals"] - df["away_goals"]

    # Ensure home_team / away_team columns exist
    if "home_team" not in df.columns:
        df["home_team"] = "Unknown"
    if "away_team" not in df.columns:
        df["away_team"] = "Unknown"

    df["is_rivalry"] = build_rivalry_flag(df)
    df["season_stage"] = build_season_stage(df)

    # Time-based features (require sorting)
    df = df.sort_values(["home_team", "season", "date"]).reset_index(drop=True)
    df["days_since_last_home_match"] = build_days_since_last_home(df)
    df["home_team_form"] = build_rolling_form(df)
    df["away_team_quality"] = build_away_team_quality(df)
    df["away_team_draw"] = build_away_team_draw(df)
    df["is_alternate_venue"] = build_alternate_venue_flag(df)

    # Market size tier
    if "metro_population" in df.columns:
        df["metro_population"] = pd.to_numeric(df["metro_population"], errors="coerce").fillna(1_000_000)
    else:
        df["metro_population"] = 1_000_000
    df["market_size_tier"] = build_market_size_tier(df["metro_population"])

    log("Step 5 — Added attendance_pct, estimated_revenue, goal_diff, rivalry flag, season_stage, "
        "days_since_last_home_match, home_team_form, away_team_quality, market_size_tier")

    # ------------------------------------------------------------------ #
    # Step 6: Categorical encoding
    # ------------------------------------------------------------------ #
    # One-hot: conference, market_size_tier, season_stage
    if "conference" in df.columns:
        conf_dummies = pd.get_dummies(df["conference"], prefix="conf", dtype=int)
        df = pd.concat([df, conf_dummies], axis=1)

    tier_dummies = pd.get_dummies(df["market_size_tier"], prefix="tier", dtype=int)
    stage_dummies = pd.get_dummies(df["season_stage"], prefix="stage", dtype=int)
    df = pd.concat([df, tier_dummies, stage_dummies], axis=1)

    # Label encode home_team and day_of_week
    df["home_team_enc"] = df["home_team"].astype("category").cat.codes
    # Same code per club as home_team_enc (-1 for a club that never hosted)
    df["away_team_enc"] = pd.Categorical(
        df["away_team"], categories=df["home_team"].astype("category").cat.categories
    ).codes
    # day_of_week is already numeric (0–6)

    log("Step 6 — One-hot encoded conference, market_size_tier, season_stage; label-encoded home_team")

    # ------------------------------------------------------------------ #
    # Step 7: Final validation
    # ------------------------------------------------------------------ #
    modeling_cols = [
        "attendance", "attendance_pct", "temperature_max", "precipitation_sum",
        "is_rainy", "is_extreme_heat", "is_cold", "is_weekend",
        "is_rivalry", "is_capacity_restricted", "home_team_form",
        "away_team_quality", "days_since_last_home_match",
    ]
    for col in modeling_cols:
        if col not in df.columns:
            df[col] = 0
        null_count = df[col].isna().sum()
        if null_count:
            log(f"  [WARN] {null_count} nulls in '{col}' — filling with median/0")
            df[col] = df[col].fillna(df[col].median() if df[col].dtype in [float, "float64"] else 0)

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    shape_after = df.shape
    df.to_csv(CLEAN_PATH, index=False)
    log(f"\nShape before: {shape_before[0]:,} rows × {shape_before[1]} cols")
    log(f"Shape after : {shape_after[0]:,} rows × {shape_after[1]} cols")
    log(f"Saved clean data: {CLEAN_PATH}")

    # Save flagged rows
    if flagged_log:
        flagged_df = pd.concat(flagged_log, ignore_index=True)
        flagged_df.to_csv(FLAGGED_PATH, index=False)
        log(f"Saved {len(flagged_df):,} flagged rows: {FLAGGED_PATH}")

    # ------------------------------------------------------------------ #
    # Build model_features.csv
    # ------------------------------------------------------------------ #
    _build_model_features(df)

    # ------------------------------------------------------------------ #
    # Generate data dictionary
    # ------------------------------------------------------------------ #
    _write_data_dictionary(df)

    print("=" * 60)
    print("[OK] Phase 2 complete -- Data cleaned and features engineered")
    print("=" * 60)
    return df


# ---------------------------------------------------------------------------
# Model features export
# ---------------------------------------------------------------------------
def _build_model_features(df: pd.DataFrame):
    """Select and export the final modeling feature table."""
    # Core feature columns for the model
    id_cols = ["match_id", "season", "date", "home_team", "away_team"]
    target_cols = ["attendance", "attendance_pct", "estimated_revenue"]
    numeric_features = [
        "matchweek", "day_of_week", "month", "is_weekend",
        "days_since_season_start", "temperature_max", "precipitation_sum",
        "windspeed_max", "is_rainy", "is_extreme_heat", "is_cold",
        "is_rivalry", "is_playoff", "is_capacity_restricted",
        "home_team_form", "away_team_quality", "days_since_last_home_match",
        "stadium_capacity", "metro_population", "avg_ticket_price_usd",
        "home_team_enc", "away_team_enc", "away_team_draw", "is_alternate_venue", "goal_diff",
    ]
    one_hot_cols = [c for c in df.columns if c.startswith(("conf_", "tier_", "stage_"))]

    keep_cols = id_cols + target_cols + numeric_features + one_hot_cols
    keep_cols = [c for c in keep_cols if c in df.columns]

    features_df = df[keep_cols].copy()
    features_df = features_df.dropna(subset=["attendance"])

    features_df.to_csv(FEATURES_PATH, index=False)
    log(f"Saved model features: {FEATURES_PATH} ({len(features_df):,} rows × {len(features_df.columns)} cols)")


# ---------------------------------------------------------------------------
# Data dictionary
# ---------------------------------------------------------------------------
def _write_data_dictionary(df: pd.DataFrame):
    """Generate a markdown data dictionary for all columns in mls_matches_clean.csv."""
    field_meta = {
        "match_id":                    ("str",   "ASA/synthetic", "Unique match identifier", "asa_00001", ""),
        "game_id":                     ("str",   "ASA API",       "Original ASA game ID", "p6qbYzRw", "May differ from match_id"),
        "season":                      ("int",   "ASA API",       "MLS season year", "2023", "ASA season label"),
        "date":                        ("date",  "ASA API",       "Match date in the home market's local time", "2023-04-15", "Converted from UTC kickoff"),
        "kickoff_utc":                 ("str",   "ASA API",       "Kickoff timestamp in UTC", "2023-04-15 23:30", ""),
        "stadium_name":                ("str",   "ASA API/Reference", "Venue the match was played in", "Mercedes-Benz Stadium", "Club home venue if ASA has no venue"),
        "capacity_source":             ("str",   "Derived",       "Where stadium_capacity came from", "match_venue", "match_venue or club_home_venue"),
        "weather_source":              ("str",   "Derived",       "Where the weather values came from", "open_meteo", "open_meteo, missing (imputed) or synthetic"),
        "home_lat":                    ("float", "ASA API/Reference", "Venue latitude", "33.7554", ""),
        "home_lon":                    ("float", "ASA API/Reference", "Venue longitude", "-84.4009", ""),
        "home_team":                   ("str",   "ASA API",       "Home team full name", "Atlanta United FC", ""),
        "away_team":                   ("str",   "ASA API",       "Away team full name", "LA Galaxy", ""),
        "home_goals":                  ("int",   "ASA API",       "Goals scored by home team", "2", ""),
        "away_goals":                  ("int",   "ASA API",       "Goals scored by away team", "1", ""),
        "attendance":                  ("int",   "ASA API",       "Official announced attendance figure", "42350", "Target variable"),
        "matchweek":                   ("int",   "ASA API",       "MLS matchday number within the season", "12", ""),
        "is_playoff":                  ("int",   "ASA API",       "1 if match is a playoff/knockout fixture, 0 otherwise", "0", "From ASA knockout_game"),
        "stadium_capacity":            ("int",   "ASA API/Wikipedia/hardcoded", "Listed capacity of the venue the match was played in", "42500", "Often the reduced soccer configuration; attendance can exceed it"),
        "metro_population":            ("int",   "Census Bureau/hardcoded", "2023 metro area population estimate", "6144050", "MSA or CMA estimate"),
        "avg_ticket_price_usd":        ("float", "Estimated (secondary market)", "Estimated average ticket price in USD", "52.0", "Secondary market estimate — not face value"),
        "conference":                  ("str",   "Reference",     "MLS conference (East/West)", "East", ""),
        "mls_debut_season":            ("int",   "Reference",     "First season the club competed in MLS", "2017", ""),
        "temperature_max":             ("float", "Open-Meteo", "Daily maximum temperature in °C at the venue", "28.5", "See weather_source"),
        "precipitation_sum":           ("float", "Open-Meteo", "Daily total precipitation in mm at the venue", "0.0", "See weather_source"),
        "windspeed_max":               ("float", "Open-Meteo", "Daily maximum windspeed in km/h at the venue", "12.3", "See weather_source"),
        "day_of_week":                 ("int",   "Derived",       "Day of week (0=Mon, 6=Sun)", "5", ""),
        "month":                       ("int",   "Derived",       "Calendar month (1–12)", "7", ""),
        "is_weekend":                  ("int",   "Derived",       "1 if match is on Saturday or Sunday", "1", ""),
        "days_since_season_start":     ("int",   "Derived",       "Days elapsed since first match of the season", "85", ""),
        "is_rainy":                    ("int",   "Derived",       "1 if precipitation_sum > 5 mm", "0", ""),
        "is_extreme_heat":             ("int",   "Derived",       "1 if temperature_max > 35°C (95°F)", "0", ""),
        "is_cold":                     ("int",   "Derived",       "1 if temperature_max < 4°C (40°F)", "0", ""),
        "attendance_pct":              ("float", "Derived",       "attendance / stadium_capacity, capped at 1.05", "0.87", "Alternative target variable"),
        "estimated_revenue":           ("float", "Derived",       "attendance × avg_ticket_price_usd (estimated)", "2202200.0", "Estimated — not from club financials"),
        "goal_diff":                   ("int",   "Derived",       "home_goals − away_goals", "1", ""),
        "is_rivalry":                  ("int",   "Derived",       "1 if match is a known rivalry fixture", "0", "Based on hardcoded rivalry pair list"),
        "is_capacity_restricted":      ("int",   "Derived",       "1 for all 2021 matches (COVID attendance caps)", "0", "Binary flag — actual per-game caps varied by city"),
        "season_stage":                ("str",   "Derived",       "Season phase (Early/Mid/Late/Playoff)", "Mid", "Based on matchweek"),
        "days_since_last_home_match":  ("float", "Derived",       "Days since home team's previous home match (same season)", "14.0", "First match of season gets prior of 14 days"),
        "home_team_form":              ("float", "Derived",       "Rolling 5-match points total for home team (max 15)", "9.0", "3=W, 1=D, 0=L"),
        "away_team_quality":           ("float", "Derived",       "Away team season win% up to this match (proxy for opponent draw)", "0.48", ""),
        "market_size_tier":            ("str",   "Derived",       "Large (>5M pop), Medium (1M–5M), Small (<1M)", "Large", ""),
        "home_team_enc":               ("int",   "Derived",       "Label-encoded home_team (for tree models)", "3", "Ordinal — not meaningful beyond tree models"),
        "away_team_enc":               ("int",   "Derived",       "Label-encoded away_team, same codes as home_team_enc", "12", "Ordinal — not meaningful beyond tree models"),
        "away_team_draw":              ("float", "Derived",       "Visiting club's recent road draw: mean of attendance / host's prior home average over its previous 10 away matches", "1.08", "1.0 = hosts drew their usual crowd; uses earlier matches only"),
        "is_alternate_venue":          ("int",   "Derived",       "1 if played outside the home club's usual venue that season", "0", "Known in advance from the schedule"),
    }

    lines = [
        "# Data Dictionary — mls_matches_clean.csv\n",
        "Generated automatically by `src/clean_data.py`.\n",
        "| field_name | data_type | source | description | example_value | notes |",
        "|---|---|---|---|---|---|",
    ]
    for col in df.columns:
        meta = field_meta.get(col)
        if meta:
            dtype, source, desc, example, notes = meta
        else:
            dtype = str(df[col].dtype)
            source = "Unknown"
            desc = col.replace("_", " ").title()
            example = str(df[col].iloc[0]) if len(df) else ""
            notes = "Auto-generated column"
        lines.append(f"| `{col}` | {dtype} | {source} | {desc} | `{example}` | {notes} |")

    with open(DICT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Saved data dictionary: {DICT_PATH}")


if __name__ == "__main__":
    clean()
