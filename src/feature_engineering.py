"""
Shared feature construction functions used by both clean_data.py (training)
and model.py (inference). Keeping logic here prevents train/serve skew.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Rivalry pairs (home_team, away_team) — order-insensitive.
# Names must match the ASA team names used in the match data exactly.
# ---------------------------------------------------------------------------
RIVALRY_PAIRS = {
    frozenset({"LA Galaxy", "Los Angeles FC"}),         # El Tráfico
    frozenset({"Seattle Sounders FC", "Portland Timbers FC"}),  # Cascadia Cup
    frozenset({"Seattle Sounders FC", "Vancouver Whitecaps FC"}),  # Cascadia
    frozenset({"Portland Timbers FC", "Vancouver Whitecaps FC"}),  # Cascadia
    frozenset({"New York Red Bulls", "New York City FC"}),  # Hudson River Derby
    frozenset({"D.C. United", "Philadelphia Union"}),       # Atlantic Cup
    frozenset({"Atlanta United FC", "Charlotte FC"}),       # I-85 Derby
    frozenset({"Colorado Rapids", "Real Salt Lake"}),       # Rocky Mountain Cup
    frozenset({"Chicago Fire FC", "FC Cincinnati"}),
    frozenset({"LA Galaxy", "San Jose Earthquakes"}),       # California Clásico
    frozenset({"Toronto FC", "CF Montréal"}),               # Canadian Clásico
    frozenset({"Columbus Crew", "FC Cincinnati"}),          # Ohio Derby
}


def build_rivalry_flag(df: pd.DataFrame) -> pd.Series:
    """Return 1 if the match is a known rivalry, 0 otherwise."""
    return df.apply(
        lambda r: int(frozenset({r["home_team"], r["away_team"]}) in RIVALRY_PAIRS),
        axis=1,
    )


def build_season_stage(df: pd.DataFrame) -> pd.Series:
    """
    Categorise each match into a season stage based on matchweek.
    Returns: 'Early', 'Mid', 'Late', or 'Playoff'
    """
    def _stage(row):
        if row.get("is_playoff", 0) == 1:
            return "Playoff"
        mw = row.get("matchweek", 0)
        if mw is None or pd.isna(mw):
            return "Mid"
        mw = int(mw)
        if mw <= 5:
            return "Early"
        elif mw <= 25:
            return "Mid"
        else:
            return "Late"

    return df.apply(_stage, axis=1)


def build_rolling_form(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Rolling points total for the home team over the last `window` matches
    (3 = win, 1 = draw, 0 = loss). Computed per team, in date order.
    Returns a Series aligned to df.index.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"])

    def _points(row):
        hg = row.get("home_goals", 0)
        ag = row.get("away_goals", 0)
        if hg > ag:
            return 3
        elif hg == ag:
            return 1
        return 0

    df["_pts"] = df.apply(_points, axis=1)
    df = df.sort_values(["home_team", "_date"])

    rolling = (
        df.groupby("home_team")["_pts"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    )
    return rolling.fillna(0)


def build_away_team_quality(df: pd.DataFrame) -> pd.Series:
    """
    Away team's season win percentage up to (but not including) this match.
    Proxy for opponent draw quality.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["away_team", "season", "_date"])

    def _away_win(row):
        ag = row.get("away_goals", 0)
        hg = row.get("home_goals", 0)
        return int(ag > hg)

    df["_away_win"] = df.apply(_away_win, axis=1)

    quality = (
        df.groupby(["away_team", "season"])["_away_win"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    return quality.fillna(0.5)  # prior = 50% for first match


def build_days_since_last_home(df: pd.DataFrame) -> pd.Series:
    """
    Days elapsed since the home team's previous home match (within the same season).
    First match of the season gets 14 days as a neutral prior.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["home_team", "season", "_date"])

    days = (
        df.groupby(["home_team", "season"])["_date"]
        .transform(lambda s: s.diff().dt.days)
    )
    return days.fillna(14)


def build_market_size_tier(metro_population: pd.Series) -> pd.Series:
    """Bin metro population into Large / Medium / Small tiers."""
    def _tier(pop):
        if pd.isna(pop) or pop == 0:
            return "Medium"
        if pop >= 5_000_000:
            return "Large"
        elif pop >= 1_000_000:
            return "Medium"
        return "Small"

    return metro_population.apply(_tier)


def build_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean weather flag columns to df:
      - is_rainy       : precipitation_sum > 5 mm
      - is_extreme_heat: temperature_max > 35 °C (95 °F)
      - is_cold        : temperature_max < 4 °C (40 °F)
    Returns df with new columns added.
    """
    df = df.copy()
    prec = df.get("precipitation_sum", pd.Series(0, index=df.index))
    temp = df.get("temperature_max", pd.Series(20, index=df.index))

    df["is_rainy"] = (prec.fillna(0) > 5).astype(int)
    df["is_extreme_heat"] = (temp.fillna(20) > 35).astype(int)
    df["is_cold"] = (temp.fillna(20) < 4).astype(int)
    return df


def build_away_team_draw(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    How much the visiting club has recently moved crowds on the road.

    For every match, attendance is first expressed relative to what the host had
    averaged at home *before* that match. The feature is the visiting club's mean
    of that ratio over its previous `window` away matches (1.0 = hosts drew their
    usual crowd). Only earlier matches are used, so nothing leaks from the match
    being predicted. Clubs with no away history yet get the neutral value 1.0.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["_date", "home_team"])

    host_prior_avg = (
        df.groupby("home_team")["attendance"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    df["_rel"] = (df["attendance"] / host_prior_avg).clip(upper=4.0)

    draw = (
        df.groupby("away_team")["_rel"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
    )
    return draw.fillna(1.0).round(4)


def build_alternate_venue_flag(df: pd.DataFrame) -> pd.Series:
    """
    1 if the match is played somewhere other than the home club's usual venue
    that season (e.g. a marquee fixture moved to an NFL stadium), 0 otherwise.
    The venue is announced with the schedule, so this is known before kickoff.
    """
    if "stadium_name" not in df.columns:
        return pd.Series(0, index=df.index)
    usual = df.groupby(["home_team", "season"])["stadium_name"].transform(
        lambda s: s.mode().iloc[0] if s.notna().any() else None
    )
    return ((df["stadium_name"] != usual) & df["stadium_name"].notna()).astype(int)
