"""
Phase 1 — Data Collection
Collects MLS match data, stadium/market reference data, and weather data.
Falls back gracefully to synthetic data if any external source is unavailable.
"""

import json
import os
import sys
import threading
import time
import difflib
import random
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

WEATHER_CACHE_PATH = DATA_RAW / "weather_cache.json"
STADIUM_MARKET_CSV = DATA_RAW / "stadium_market.csv"
PROVENANCE_PATH = DATA_RAW / "stadium_market_provenance.json"
OUTPUT_PATH = DATA_RAW / "mls_matches_raw.csv"

# ---------------------------------------------------------------------------
# Hardcoded reference data (Sub-source 2c)
# Intentionally hardcoded: these values are stable facts that do not require
# automated collection.
# ---------------------------------------------------------------------------
TEAM_REFERENCE = [
    # abbr, name, stadium, city, state, founding, conference, lat, lon, debut, ticket_price
    ("ATL",  "Atlanta United FC",       "Mercedes-Benz Stadium",       "Atlanta",       "GA", 2014, "East",  33.7554, -84.4009,  2017, 52),
    ("ATX",  "Austin FC",               "Q2 Stadium",                  "Austin",        "TX", 2018, "West",  30.3870, -97.7191,  2021, 61),
    ("CHI",  "Chicago Fire FC",         "Soldier Field",               "Chicago",       "IL", 1997, "East",  41.8623, -87.6167,  1998, 48),
    ("CLB",  "Columbus Crew",           "Lower.com Field",             "Columbus",      "OH", 1995, "East",  39.9690, -83.0175,  1996, 42),
    ("CLT",  "Charlotte FC",            "Bank of America Stadium",     "Charlotte",     "NC", 2019, "East",  35.2258, -80.8528,  2022, 55),
    ("COL",  "Colorado Rapids",         "Dick's Sporting Goods Park",  "Commerce City", "CO", 1995, "West",  39.8054, -104.8919, 1996, 40),
    ("DAL",  "FC Dallas",               "Toyota Stadium",              "Frisco",        "TX", 1995, "West",  33.1548, -96.8354,  1996, 43),
    ("DC",   "D.C. United",             "Audi Field",                  "Washington",    "DC", 1995, "East",  38.8683, -77.0122,  1996, 50),
    ("HOU",  "Houston Dynamo FC",       "Shell Energy Stadium",        "Houston",       "TX", 2005, "West",  29.7520, -95.3510,  2006, 41),
    ("LA",   "LA Galaxy",               "Dignity Health Sports Park",  "Carson",        "CA", 1995, "West",  33.8644, -118.2611, 1996, 65),
    ("LAFC", "Los Angeles FC",          "BMO Stadium",                 "Los Angeles",   "CA", 2014, "West",  34.0126, -118.2849, 2018, 78),
    ("MIA",  "Inter Miami CF",          "Chase Stadium",               "Fort Lauderdale","FL",2018, "East",  26.1904, -80.1576,  2020, 88),
    ("MIN",  "Minnesota United FC",     "Allianz Field",               "Saint Paul",    "MN", 2010, "West",  44.9531, -93.1650,  2017, 49),
    ("MTL",  "CF Montréal",             "Stade Saputo",                "Montreal",      "QC", 1993, "East",  45.5638, -73.5510,  2012, 46),
    ("NE",   "New England Revolution",  "Gillette Stadium",            "Foxborough",    "MA", 1995, "East",  42.0909, -71.2643,  1996, 53),
    ("NSH",  "Nashville SC",            "GEODIS Park",                 "Nashville",     "TN", 2017, "East",  36.1303, -86.7714,  2020, 58),
    ("NY",   "New York Red Bulls",      "Red Bull Arena",              "Harrison",      "NJ", 1994, "East",  40.7370, -74.1502,  1996, 67),
    ("NYC",  "New York City FC",        "Yankee Stadium",              "New York",      "NY", 2013, "East",  40.8296, -73.9262,  2015, 82),
    ("ORL",  "Orlando City SC",         "Inter&Co Stadium",            "Orlando",       "FL", 2010, "East",  28.5392, -81.3892,  2015, 44),
    ("PHI",  "Philadelphia Union",      "Subaru Park",                 "Chester",       "PA", 2008, "East",  39.8317, -75.3797,  2010, 46),
    ("POR",  "Portland Timbers FC",     "Providence Park",             "Portland",      "OR", 1975, "West",  45.5213, -122.6916, 2011, 58),
    ("RSL",  "Real Salt Lake",          "America First Field",         "Sandy",         "UT", 2004, "West",  40.5830, -111.8930, 2005, 44),
    ("SD",   "San Diego FC",            "Snapdragon Stadium",          "San Diego",     "CA", 2023, "West",  32.7831, -117.1188, 2025, 58),
    ("SEA",  "Seattle Sounders FC",     "Lumen Field",                 "Seattle",       "WA", 1994, "West",  47.5952, -122.3316, 2009, 71),
    ("SJ",   "San Jose Earthquakes",    "PayPal Park",                 "San Jose",      "CA", 1995, "West",  37.3516, -121.9250, 1996, 47),
    ("SKC",  "Sporting Kansas City",    "Children's Mercy Park",       "Kansas City",   "KS", 1995, "West",  39.1224, -94.8232,  1996, 51),
    ("STL",  "St. Louis City SC",       "CITYPARK",                    "St. Louis",     "MO", 2019, "West",  38.6332, -90.2138,  2023, 54),
    ("TOR",  "Toronto FC",              "BMO Field",                   "Toronto",       "ON", 2005, "East",  43.6334, -79.4179,  2007, 69),
    ("VAN",  "Vancouver Whitecaps FC",  "BC Place",                    "Vancouver",     "BC", 1986, "West",  49.2769, -123.1121, 2011, 52),
    # Ticket prices for the next two clubs are placeholders (league-wide default), not researched estimates
    ("CIN",  "FC Cincinnati",           "TQL Stadium",                 "Cincinnati",    "OH", 2015, "East",  39.1114, -84.5222,  2019, 55),
    ("CHV",  "Chivas USA",              "Dignity Health Sports Park",  "Carson",        "CA", 2004, "West",  33.8644, -118.2611, 2005, 55),  # defunct after 2014
]

# IANA timezone of each club's home market — ASA kickoff times are UTC, and an
# evening kickoff in North America falls on the *next* UTC calendar day.
TEAM_TIMEZONES = {
    "ATL": "America/New_York", "ATX": "America/Chicago", "CHI": "America/Chicago",
    "CLB": "America/New_York", "CLT": "America/New_York", "COL": "America/Denver",
    "DAL": "America/Chicago", "DC": "America/New_York", "HOU": "America/Chicago",
    "LA": "America/Los_Angeles", "LAFC": "America/Los_Angeles", "MIA": "America/New_York",
    "MIN": "America/Chicago", "MTL": "America/Toronto", "NE": "America/New_York",
    "NSH": "America/Chicago", "NY": "America/New_York", "NYC": "America/New_York",
    "ORL": "America/New_York", "PHI": "America/New_York", "POR": "America/Los_Angeles",
    "RSL": "America/Denver", "SD": "America/Los_Angeles", "SEA": "America/Los_Angeles",
    "SJ": "America/Los_Angeles", "SKC": "America/Chicago", "STL": "America/Chicago",
    "TOR": "America/Toronto", "VAN": "America/Vancouver", "CIN": "America/New_York",
    "CHV": "America/Los_Angeles",
}

# Venues that ASA's stadium table lists without a capacity and/or coordinates:
# name -> (capacity, latitude, longitude). Only used to fill gaps.
VENUE_REFERENCE = {
    "Bank of America Stadium":        (74867, 35.2258, -80.8528),
    "Snapdragon Stadium":             (35000, 32.7831, -117.1188),
    "Miami Freedom Park":             (26700, 25.7890, -80.2590),
    "Empower Field at Mile High":     (76125, 39.7439, -105.0201),
    "FirstEnergy Stadium":            (67431, 41.5061, -81.6995),
    "M&T Bank Stadium":               (70745, 39.2780, -76.6227),
    "Rose Bowl":                      (89702, 34.1613, -118.1676),
    "Los Angeles Memorial Coliseum":  (77500, 34.0141, -118.2879),
    "Arrowhead Stadium":              (76416, 39.0489, -94.4839),
}

# ASA capacity values that are clearly wrong (attendance there routinely exceeded them 2-3x)
CAPACITY_CORRECTIONS = {
    "Camping World Stadium": 60219,
}

# Hardcoded stadium capacities (fallback)
CAPACITY_FALLBACK = {
    "ATL": 42500, "ATX": 20738, "CHI": 61500, "CLB": 20371, "CLT": 74867,
    "COL": 18061, "DAL": 19096, "DC": 20000, "HOU": 22039, "LA": 27000,
    "LAFC": 22000, "MIA": 21550, "MIN": 19400, "MTL": 19619, "NE": 65878,
    "NSH": 30000, "NY": 25000, "NYC": 54251, "ORL": 25500, "PHI": 18500,
    "POR": 25218, "RSL": 20213, "SD": 35000, "SEA": 68740, "SJ": 18000,
    "SKC": 18467, "STL": 22500, "TOR": 30000, "VAN": 54320,
    "CIN": 26000, "CHV": 27000,
}

# Hardcoded metro populations (fallback, incl. Canadian CMAs)
POPULATION_FALLBACK = {
    "ATL": 6_144_050, "ATX": 2_352_426, "CHI": 9_478_801, "CLB": 2_138_926,
    "CLT": 2_701_487, "COL": 2_963_821, "DAL": 7_759_615, "DC": 6_394_598,
    "HOU": 7_386_941, "LA": 13_200_998, "LAFC": 13_200_998, "MIA": 6_183_199,
    "MIN": 3_712_020, "MTL": 4_292_000, "NE": 4_941_632, "NSH": 2_107_533,
    "NY": 19_979_477, "NYC": 19_979_477, "ORL": 3_238_723, "PHI": 6_245_051,
    "POR": 2_508_050, "RSL": 1_257_936, "SD": 3_286_069, "SEA": 4_044_837,
    "SJ": 1_999_107, "SKC": 2_238_279, "STL": 2_820_253, "TOR": 6_712_000,
    "VAN": 2_906_000, "CIN": 2_271_479, "CHV": 13_200_998,
}

# Team coords for weather API
TEAM_COORDS = {row[0]: (row[7], row[8]) for row in TEAM_REFERENCE}

# Wikipedia stadium article titles (unencoded — scrape_stadium_capacity() URL-encodes them)
STADIUM_WIKI = {
    "ATL":  "Mercedes-Benz_Stadium",
    "ATX":  "Q2_Stadium",
    "CHI":  "Soldier_Field",
    "CLB":  "Lower.com_Field",
    "CLT":  "Bank_of_America_Stadium",
    "COL":  "Dick's_Sporting_Goods_Park",
    "DAL":  "Toyota_Stadium_(Texas)",
    "DC":   "Audi_Field",
    "HOU":  "Shell_Energy_Stadium",
    "LA":   "Dignity_Health_Sports_Park",
    "LAFC": "BMO_Stadium",
    "MIA":  "Chase_Stadium",
    "MIN":  "Allianz_Field",
    "MTL":  "Stade_Saputo",
    "NE":   "Gillette_Stadium",
    "NSH":  "GEODIS_Park",
    "NY":   "Red_Bull_Arena_(New_Jersey)",
    "NYC":  "Yankee_Stadium",
    "ORL":  "Inter&Co_Stadium",
    "PHI":  "Subaru_Park",
    "POR":  "Providence_Park",
    "RSL":  "America_First_Field",
    "SD":   "Snapdragon_Stadium",
    "SEA":  "Lumen_Field",
    "SJ":   "PayPal_Park",
    "SKC":  "Children's_Mercy_Park",
    "STL":  "CITYPARK",
    "TOR":  "BMO_Field",
    "VAN":  "BC_Place",
    "CIN":  "TQL_Stadium",
    "CHV":  "Dignity_Health_Sports_Park",
}


# ---------------------------------------------------------------------------
# Sub-source 2a — Wikipedia stadium capacity scraper
# ---------------------------------------------------------------------------
def scrape_stadium_capacity(abbr: str, wiki_title: str, provenance: dict) -> int:
    """Scrape stadium capacity from Wikipedia infobox. Falls back to CAPACITY_FALLBACK."""
    try:
        from bs4 import BeautifulSoup
        encoded = urllib.parse.quote(wiki_title, safe="_()")
        url = f"https://en.wikipedia.org/wiki/{encoded}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MLSResearchBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        infobox = soup.find("table", {"class": "infobox"})
        if infobox is None:
            raise ValueError("No infobox found")

        for row in infobox.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if th and td and "capacity" in th.get_text(strip=True).lower():
                raw = td.get_text(separator=" ", strip=True)
                # Remove citations like [1], [a], footnotes
                import re
                raw = re.sub(r"\[.*?\]", "", raw)
                raw = re.sub(r"\(.*?\)", "", raw)
                capacity = int("".join(filter(str.isdigit, raw.split()[0].replace(",", ""))))
                # Infoboxes sometimes list a historical or partial figure first
                reference = CAPACITY_FALLBACK.get(abbr)
                if reference and abs(capacity - reference) / reference > 0.25:
                    raise ValueError(f"Scraped capacity {capacity:,} implausible vs reference {reference:,}")
                provenance[abbr]["capacity_source"] = "wikipedia_scraped"
                return capacity

        raise ValueError("Capacity row not found in infobox")

    except Exception as exc:
        fallback = CAPACITY_FALLBACK.get(abbr, 25000)
        provenance[abbr]["capacity_source"] = "hardcoded_fallback"
        provenance[abbr]["capacity_scrape_error"] = str(exc)
        return fallback


# ---------------------------------------------------------------------------
# Sub-source 2b — Census Bureau metro population
# ---------------------------------------------------------------------------
def fetch_census_population(provenance: dict) -> dict:
    """
    Fetch 2023 MSA population from Census Bureau API.
    Returns dict {abbr: population}.
    """
    # MSA name fragments to match each team
    MSA_HINTS = {
        "ATL": "Atlanta", "ATX": "Austin", "CHI": "Chicago",
        "CLB": "Columbus", "CLT": "Charlotte", "COL": "Denver",
        "DAL": "Dallas", "DC": "Washington", "HOU": "Houston",
        "LA": "Los Angeles", "LAFC": "Los Angeles", "MIA": "Miami",
        "MIN": "Minneapolis", "NE": "Boston", "NSH": "Nashville",
        "NY": "New York", "NYC": "New York", "ORL": "Orlando",
        "PHI": "Philadelphia", "POR": "Portland", "RSL": "Salt Lake",
        "SD": "San Diego", "SEA": "Seattle", "SJ": "San Jose",
        "SKC": "Kansas City", "STL": "St. Louis",
        "CIN": "Cincinnati", "CHV": "Los Angeles",
    }

    try:
        url = (
            "https://api.census.gov/data/2023/pep/population"
            "?get=NAME,POP_2023&for=metropolitan%20statistical%20area/"
            "micropolitan%20statistical%20area:*"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        headers = data[0]
        rows = data[1:]
        census_df = pd.DataFrame(rows, columns=headers)
        census_df["POP_2023"] = pd.to_numeric(census_df["POP_2023"], errors="coerce")

        result = {}
        for abbr, hint in MSA_HINTS.items():
            matches = census_df[census_df["NAME"].str.contains(hint, case=False, na=False)]
            if not matches.empty:
                pop = int(matches.sort_values("POP_2023", ascending=False).iloc[0]["POP_2023"])
                result[abbr] = pop
                provenance[abbr]["population_source"] = "census_api"
            else:
                result[abbr] = POPULATION_FALLBACK.get(abbr, 1_000_000)
                provenance[abbr]["population_source"] = "hardcoded_fallback"

        # Canadian teams — Statistics Canada 2023 estimates
        for abbr, pop in [("MTL", 4_292_000), ("TOR", 6_712_000), ("VAN", 2_906_000)]:
            result[abbr] = pop
            provenance[abbr]["population_source"] = "statscan_hardcoded"

        return result

    except Exception as exc:
        print(f"  [WARN] Census API unavailable ({exc}); using hardcoded population fallback.")
        for abbr in provenance:
            provenance[abbr]["population_source"] = "hardcoded_fallback"
        return dict(POPULATION_FALLBACK)


# ---------------------------------------------------------------------------
# Main: collect_stadium_market_data()
# ---------------------------------------------------------------------------
def collect_stadium_market_data() -> pd.DataFrame:
    """
    Build stadium/market reference table from three sub-sources.
    Saves stadium_market.csv and stadium_market_provenance.json.
    """
    print("\n=== Collecting Stadium & Market Data ===")
    provenance = {row[0]: {} for row in TEAM_REFERENCE}

    # ----- Base table (Sub-source 2c) -----
    base_rows = []
    for row in TEAM_REFERENCE:
        abbr, name, stadium, city, state, founding, conf, lat, lon, debut, ticket = row
        base_rows.append({
            "team_abbreviation": abbr,
            "team_name": name,
            "stadium_name": stadium,
            "city": city,
            "state": state,
            "founding_year": founding,
            "conference": conf,
            "latitude": lat,
            "longitude": lon,
            "mls_debut_season": debut,
            "avg_ticket_price_usd": ticket,
        })
    df = pd.DataFrame(base_rows)

    # ----- Sub-source 2a: Wikipedia capacity -----
    print("  Scraping stadium capacities from Wikipedia...")
    capacities = {}
    scraped_count = 0
    fallback_count = 0
    for abbr, wiki_title in STADIUM_WIKI.items():
        cap = scrape_stadium_capacity(abbr, wiki_title, provenance)
        capacities[abbr] = cap
        src = provenance.get(abbr, {}).get("capacity_source", "hardcoded_fallback")
        if src == "wikipedia_scraped":
            scraped_count += 1
        else:
            fallback_count += 1
        time.sleep(1.5)

    df["stadium_capacity"] = df["team_abbreviation"].map(capacities).fillna(25000).astype(int)

    # ----- Sub-source 2b: Census population -----
    print("  Fetching metro population from Census Bureau API...")
    populations = fetch_census_population(provenance)
    df["metro_population"] = df["team_abbreviation"].map(populations).fillna(1_000_000).astype(int)

    # ----- Save -----
    df.to_csv(STADIUM_MARKET_CSV, index=False)
    with open(PROVENANCE_PATH, "w") as f:
        json.dump(provenance, f, indent=2)

    census_count = sum(1 for v in provenance.values() if v.get("population_source") == "census_api")
    pop_fallback_count = sum(1 for v in provenance.values() if v.get("population_source") != "census_api")
    print(f"  Stadium capacities: {scraped_count} scraped, {fallback_count} from hardcoded fallback")
    print(f"  Metro populations:  {census_count} from Census API, {pop_fallback_count} from fallback/hardcoded")
    print(f"  Saved: {STADIUM_MARKET_CSV}")
    return df


# ---------------------------------------------------------------------------
# Source 1 — ASA match data (itscalledsoccer) with synthetic fallback
# ---------------------------------------------------------------------------
def _build_synthetic_matches(current_year: int) -> pd.DataFrame:
    """Generate a realistic synthetic MLS match dataset as fallback."""
    print("  [FALLBACK] Generating synthetic MLS match data...")
    random.seed(42)
    np.random.seed(42)

    # Load stadium reference for capacity-based distributions
    if STADIUM_MARKET_CSV.exists():
        sm = pd.read_csv(STADIUM_MARKET_CSV)
        cap_map = dict(zip(sm["team_abbreviation"], sm["stadium_capacity"]))
        debut_map = dict(zip(sm["team_abbreviation"], sm["mls_debut_season"]))
        name_map = dict(zip(sm["team_abbreviation"], sm["team_name"]))
    else:
        cap_map = dict(CAPACITY_FALLBACK)
        debut_map = {row[0]: row[9] for row in TEAM_REFERENCE}
        name_map = {row[0]: row[1] for row in TEAM_REFERENCE}

    teams = [row[0] for row in TEAM_REFERENCE]
    rivalry_pairs_abbr = [
        ("LA", "LAFC"), ("SEA", "POR"), ("SEA", "VAN"), ("POR", "VAN"),
        ("NY", "NYC"), ("DC", "PHI"), ("ATL", "CLT"), ("COL", "RSL"),
        ("LA", "SJ"), ("TOR", "MTL"), ("CLB", "CHI"),
    ]

    rows = []
    match_id = 1
    for season in range(2012, current_year + 1):
        season_teams = [t for t in teams if debut_map.get(t, 1996) <= season]
        if not season_teams:
            continue
        n_teams = len(season_teams)
        games_per_team = 17
        matchweeks = games_per_team

        for mw in range(1, matchweeks + 1):
            # Pair teams for this matchweek
            random.shuffle(season_teams)
            for i in range(0, n_teams - 1, 2):
                home = season_teams[i]
                away = season_teams[i + 1]
                cap = cap_map.get(home, 25000)

                # Seasonal attendance pattern
                mw_factor = 1.0
                if mw <= 3:
                    mw_factor = 0.88
                elif mw >= matchweeks - 1:
                    mw_factor = 0.91
                elif 8 <= mw <= 20:
                    mw_factor = 1.04

                # Rivalry boost
                is_rival = int(
                    (home, away) in rivalry_pairs_abbr or
                    (away, home) in rivalry_pairs_abbr
                )
                rival_boost = 1 + random.uniform(0.10, 0.18) if is_rival else 1.0

                # SD FC: new expansion team ramp-up
                if home == "SD":
                    base_pct = 0.70 + (mw / matchweeks) * 0.10
                else:
                    base_pct = 0.85

                mean_att = cap * base_pct * mw_factor * rival_boost
                std_att = cap * 0.08
                attendance = int(np.clip(np.random.normal(mean_att, std_att), cap * 0.45, cap * 1.02))

                # Match date (approximate)
                base_date = datetime(season, 3, 1) + timedelta(weeks=mw - 1)
                date = base_date + timedelta(days=random.randint(0, 6))

                # Goals
                home_goals = np.random.poisson(1.5)
                away_goals = np.random.poisson(1.2)

                rows.append({
                    "match_id": f"synth_{match_id:05d}",
                    "season": season,
                    "date": date.strftime("%Y-%m-%d"),
                    "home_team_id": home,
                    "away_team_id": away,
                    "home_team": name_map.get(home, home),
                    "away_team": name_map.get(away, away),
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "attendance": attendance,
                    "matchweek": mw,
                    "is_playoff": 0,
                    "stage": "Regular Season",
                })
                match_id += 1

    df = pd.DataFrame(rows)
    print(f"  Synthetic dataset: {len(df):,} matches, seasons {df['season'].min()}–{df['season'].max()}")
    return df


ASA_API = "https://app.americansocceranalysis.com/api/v1/mls"


def _to_pandas(frame) -> pd.DataFrame:
    """itscalledsoccer >= 1.0 returns polars DataFrames; older releases return pandas."""
    if isinstance(frame, pd.DataFrame):
        return frame
    return pd.DataFrame(frame.to_dicts())


def collect_asa_matches() -> pd.DataFrame:
    """Fetch MLS match data from ASA via itscalledsoccer. Falls back to synthetic."""
    print("\n=== Collecting MLS Match Data (ASA) ===")
    current_year = datetime.today().year

    try:
        from itscalledsoccer.client import AmericanSoccerAnalysis
        print("  Connecting to American Soccer Analysis API...")
        asa = AmericanSoccerAnalysis()
        games_df = _to_pandas(asa.get_games(leagues=["mls"]))
        print("  ASA games columns:", games_df.columns.tolist())

        # Defensive column mapping
        COLUMN_MAP = {
            "game_id": "match_id",
            "date_time_utc": "date",
            "date": "date",
            "home_team_id": "home_team_id",
            "away_team_id": "away_team_id",
            "home_score": "home_goals",
            "away_score": "away_goals",
            "attendance": "attendance",
            "matchday": "matchweek",
            "matchweek": "matchweek",
        }
        games_df = games_df.rename(
            columns={k: v for k, v in COLUMN_MAP.items() if k in games_df.columns}
        )

        # Resolve team IDs to names
        try:
            teams_df = _to_pandas(asa.get_teams(leagues=["mls"]))
            id_col = next((c for c in ["team_id", "id"] if c in teams_df.columns), None)
            name_col = next((c for c in ["team_name", "name"] if c in teams_df.columns), None)
            if id_col and name_col:
                id_to_name = dict(zip(teams_df[id_col], teams_df[name_col]))
                if "home_team_id" in games_df.columns:
                    games_df["home_team"] = games_df["home_team_id"].map(id_to_name).fillna(games_df["home_team_id"])
                if "away_team_id" in games_df.columns:
                    games_df["away_team"] = games_df["away_team_id"].map(id_to_name).fillna(games_df["away_team_id"])
        except Exception as te:
            print(f"  [WARN] Could not resolve team names: {te}")
            if "home_team" not in games_df.columns and "home_team_id" in games_df.columns:
                games_df["home_team"] = games_df["home_team_id"]
            if "away_team" not in games_df.columns and "away_team_id" in games_df.columns:
                games_df["away_team"] = games_df["away_team_id"]

        # Kickoff is reported in UTC; convert to the home market's local calendar date
        # so that date, day-of-week and weather all refer to the day the match was played.
        name_to_tz = {row[1]: TEAM_TIMEZONES.get(row[0]) for row in TEAM_REFERENCE}
        kickoff_utc = pd.to_datetime(games_df["date"], utc=True, errors="coerce")
        zones = games_df["home_team"].map(name_to_tz).fillna("America/New_York")
        games_df["kickoff_utc"] = kickoff_utc.dt.strftime("%Y-%m-%d %H:%M")
        games_df["date"] = [
            ts.tz_convert(zone).strftime("%Y-%m-%d") if pd.notna(ts) else None
            for ts, zone in zip(kickoff_utc, zones)
        ]

        # Season: ASA's own season label (falls back to the calendar year of the match)
        date_year = pd.to_datetime(games_df["date"], errors="coerce").dt.year
        if "season_name" in games_df.columns:
            games_df["season"] = pd.to_numeric(games_df["season_name"], errors="coerce").fillna(date_year)
        else:
            games_df["season"] = date_year

        # Playoff flag: ASA marks post-season fixtures with `knockout_game`
        if "knockout_game" in games_df.columns:
            games_df["is_playoff"] = games_df["knockout_game"].fillna(False).astype(bool).astype(int)
        else:
            print("  [WARN] No knockout_game column found; setting is_playoff=0")
            games_df["is_playoff"] = 0
        games_df["stage"] = np.where(games_df["is_playoff"] == 1, "Playoff", "Regular Season")

        # Ensure matchweek exists
        if "matchweek" not in games_df.columns:
            games_df["matchweek"] = 0

        # Ensure required columns
        for col in ["home_goals", "away_goals"]:
            if col not in games_df.columns:
                games_df[col] = 0

        if "match_id" not in games_df.columns:
            games_df["match_id"] = [f"asa_{i:05d}" for i in range(len(games_df))]

        print(f"  ASA data: {len(games_df):,} rows, attendance non-null: {games_df['attendance'].notna().sum():,}, "
              f"playoff matches: {int(games_df['is_playoff'].sum())}")
        return games_df

    except Exception as exc:
        print("  " + "!" * 56)
        print(f"  [WARN] ASA API unavailable ({exc}); falling back to SYNTHETIC match data.")
        print("  " + "!" * 56)
        return _build_synthetic_matches(current_year)


# ---------------------------------------------------------------------------
# Source 1b — ASA stadium table (per-match venue capacity & coordinates)
# ---------------------------------------------------------------------------
def fetch_asa_stadia() -> pd.DataFrame:
    """
    Venue table from the ASA REST API. Queried directly because the client's
    get_stadia() concatenates every league and fails on mismatched columns.
    """
    resp = requests.get(f"{ASA_API}/stadia", timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def attach_venue_data(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add venue_name / venue_capacity / venue_lat / venue_lon for the stadium each
    match was actually played in (clubs change stadium and play one-off games in
    larger venues). Gaps are filled later from the club's home-venue reference.
    """
    print("\n=== Attaching Venue Data (ASA stadia) ===")
    df = matches_df.copy()
    venue_cols = ["venue_name", "venue_capacity", "venue_lat", "venue_lon"]

    def _without_venues(reason: str) -> pd.DataFrame:
        print(f"  [WARN] {reason}; using each club's home venue for every match.")
        df["venue_name"] = None
        for col in venue_cols[1:]:
            df[col] = np.nan
        return df

    if "stadium_id" not in df.columns:
        return _without_venues("No stadium_id column")
    try:
        stadia = fetch_asa_stadia()
    except Exception as exc:
        return _without_venues(f"ASA stadia unavailable ({exc})")

    # A handful of matches have no stadium_id: assume the club's usual venue that season
    usual = (
        df.dropna(subset=["stadium_id"])
        .groupby(["home_team", "season"])["stadium_id"]
        .agg(lambda s: s.mode().iloc[0])
    )
    missing_id = df["stadium_id"].isna()
    keys = list(zip(df.loc[missing_id, "home_team"], df.loc[missing_id, "season"]))
    df.loc[missing_id, "stadium_id"] = [usual.get(k) for k in keys]

    stadia = stadia.rename(columns={
        "stadium_name": "venue_name", "capacity": "venue_capacity",
        "latitude": "venue_lat", "longitude": "venue_lon",
    })[["stadium_id"] + venue_cols]
    for col in venue_cols[1:]:
        stadia[col] = pd.to_numeric(stadia[col], errors="coerce")

    for name, capacity in CAPACITY_CORRECTIONS.items():
        stadia.loc[stadia["venue_name"] == name, "venue_capacity"] = capacity
    for name, (capacity, lat, lon) in VENUE_REFERENCE.items():
        row = stadia["venue_name"] == name
        stadia.loc[row, "venue_capacity"] = stadia.loc[row, "venue_capacity"].fillna(capacity)
        stadia.loc[row, "venue_lat"] = stadia.loc[row, "venue_lat"].fillna(lat)
        stadia.loc[row, "venue_lon"] = stadia.loc[row, "venue_lon"].fillna(lon)

    df = df.merge(stadia, on="stadium_id", how="left")
    print(f"  Venue matched for {df['venue_name'].notna().sum():,} of {len(df):,} matches "
          f"({df['venue_name'].nunique()} distinct venues); "
          f"{int(missing_id.sum())} matches had no stadium_id and use the club's usual venue that season")
    no_cap = df.loc[df["venue_name"].notna() & df["venue_capacity"].isna(), "venue_name"].unique().tolist()
    if no_cap:
        print(f"  [WARN] No capacity for venues {no_cap}; the club's home-venue capacity is used instead")
    return df


# ---------------------------------------------------------------------------
# Source 3 — Weather (Open-Meteo historical archive)
# ---------------------------------------------------------------------------
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_WINDOW_DAYS = 14        # Open-Meteo counts a request of <= 2 weeks as one API call
WEATHER_WORKERS = 4
WEATHER_MIN_INTERVAL = 0.5      # seconds between requests per worker (~480/min, limit is 600/min)
WEATHER_MAX_CONSECUTIVE_FAILURES = 8


def _weather_key(lat: float, lon: float, date_str: str) -> str:
    return f"{lat:.3f},{lon:.3f},{date_str}"


def _plan_weather_windows(dates: list[str]) -> list[tuple[str, str]]:
    """Group match dates at one venue into windows of at most WEATHER_WINDOW_DAYS."""
    windows = []
    start = end = None
    for d in sorted(set(dates)):
        day = datetime.strptime(d, "%Y-%m-%d")
        if start is None:
            start = end = day
        elif (day - start).days < WEATHER_WINDOW_DAYS:
            end = day
        else:
            windows.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
            start = end = day
    if start is not None:
        windows.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return windows


def fetch_weather_window(lat: float, lon: float, start: str, end: str, cache: dict) -> str | None:
    """
    Fetch daily weather for one location and date window into `cache`.
    Returns None on success, or the last error message after retries.
    """
    params = {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max",
        "timezone": "auto",
    }
    error = None
    for attempt in range(3):
        try:
            resp = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=30)
            if resp.status_code == 429:
                error = f"429 rate limited: {resp.text[:120]}"
                time.sleep(20 * (attempt + 1))
                continue
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            for i, day in enumerate(daily.get("time", [])):
                temp = daily["temperature_2m_max"][i]
                if temp is None:      # archive not yet available for very recent days
                    continue
                cache[_weather_key(lat, lon, day)] = {
                    "temperature_max": temp,
                    "precipitation_sum": daily["precipitation_sum"][i] or 0.0,
                    "windspeed_max": daily["wind_speed_10m_max"][i] or 0.0,
                }
            return None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * (attempt + 1))
    return error


def _synthetic_weather(lat: float, month: int) -> dict:
    """Generate realistic synthetic weather based on latitude and month."""
    # Temperature model: sinusoidal by latitude
    base_temp = 20 - abs(lat - 25) * 0.3
    seasonal_offset = 8 * np.sin((month - 1) / 12 * 2 * np.pi)
    temp_c = base_temp + seasonal_offset + np.random.normal(0, 3)

    # Precipitation: higher probability in spring/fall, higher at high latitudes
    rain_prob = 0.15 + 0.1 * (abs(lat) > 40) + 0.05 * (month in [3, 4, 5, 9, 10, 11])
    precip = np.random.exponential(5) if np.random.random() < rain_prob else 0.0

    return {
        "temperature_max": round(temp_c, 1),
        "precipitation_sum": round(precip, 1),
        "windspeed_max": round(abs(np.random.normal(15, 5)), 1),
    }


def collect_weather(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add match-day weather at the venue (home_lat / home_lon / local date).
    Adds a `weather_source` column: 'open_meteo', 'missing' (left null for the
    cleaning step to impute) or 'synthetic' (only when the API is unreachable).
    """
    print("\n=== Collecting Weather Data (Open-Meteo) ===")

    cache = {}
    if WEATHER_CACHE_PATH.exists():
        try:
            with open(WEATHER_CACHE_PATH) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    df = matches_df
    for col in ["home_lat", "home_lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    date_str = df["date"].astype(str).str[:10]
    has_coords = df["home_lat"].notna() & df["home_lon"].notna() & df["date"].notna()
    keys = pd.Series(
        [_weather_key(la, lo, d) if ok else None
         for la, lo, d, ok in zip(df["home_lat"], df["home_lon"], date_str, has_coords)],
        index=df.index,
    )

    # Plan one request per venue per <=14-day window, skipping days already cached
    today = datetime.today().strftime("%Y-%m-%d")
    needed = {}
    for la, lo, d, k in zip(df["home_lat"], df["home_lon"], date_str, keys):
        if k is not None and k not in cache and d < today:
            needed.setdefault((round(la, 3), round(lo, 3)), []).append(d)
    jobs = [(lat, lon, start, end)
            for (lat, lon), dates in needed.items()
            for start, end in _plan_weather_windows(dates)]
    print(f"  {len(cache):,} days cached; {len(jobs):,} API requests needed "
          f"for {sum(len(set(v)) for v in needed.values()):,} uncached venue-days")

    abort = threading.Event()
    lock = threading.Lock()
    state = {"consecutive_failures": 0, "failed": 0, "done": 0, "first_error": None}

    def _run(job):
        if abort.is_set():
            return
        error = fetch_weather_window(*job, cache)
        with lock:
            state["done"] += 1
            if error:
                state["failed"] += 1
                state["consecutive_failures"] += 1
                if state["first_error"] is None:
                    state["first_error"] = error
                    print(f"  [WARN] First weather request failure: {error}")
                if state["consecutive_failures"] >= WEATHER_MAX_CONSECUTIVE_FAILURES:
                    abort.set()
            else:
                state["consecutive_failures"] = 0
            if state["done"] % 500 == 0:
                print(f"    ... {state['done']:,}/{len(jobs):,} requests", flush=True)
                with open(WEATHER_CACHE_PATH, "w") as f:
                    json.dump(dict(cache), f)
        time.sleep(WEATHER_MIN_INTERVAL)

    if jobs:
        with ThreadPoolExecutor(max_workers=WEATHER_WORKERS) as pool:
            list(pool.map(_run, jobs))
    if abort.is_set():
        print(f"  [WARN] Stopped after {WEATHER_MAX_CONSECUTIVE_FAILURES} consecutive failures — "
              "Open-Meteo appears to be unreachable or rate-limiting.")

    # Keep only match days (each request also returns the other days in its window)
    match_keys = set(keys.dropna())
    cache = {k: v for k, v in cache.items() if k in match_keys}
    with open(WEATHER_CACHE_PATH, "w") as f:
        json.dump(cache, f)

    found =keys.map(lambda k: k is not None and k in cache).astype(bool)
    for col in ["temperature_max", "precipitation_sum", "windspeed_max"]:
        df[col] = [cache[k][col] if ok else np.nan for k, ok in zip(keys, found)]
    df["weather_source"] = np.where(found, "open_meteo", "missing")

    n_real, n_total = int(found.sum()), len(df)
    if n_real < 0.5 * n_total:
        # API effectively unavailable: keep the pipeline runnable, but say so loudly
        print("  " + "!" * 56)
        print("  [WARN] Real weather unavailable for most matches — filling with SYNTHETIC values.")
        print("  [WARN] Do not draw weather conclusions from this run.")
        print("  " + "!" * 56)
        months = pd.to_datetime(date_str, errors="coerce").dt.month.fillna(7).astype(int)
        for idx in df.index[~found]:
            lat = df.at[idx, "home_lat"]
            w = _synthetic_weather(lat if pd.notna(lat) else 40.0, int(months.at[idx]))
            for col, val in w.items():
                df.at[idx, col] = val
        df.loc[~found, "weather_source"] = "synthetic"

    counts = df["weather_source"].value_counts().to_dict()
    print(f"  Weather records: {n_real:,} of {n_total:,} from Open-Meteo ({n_real / max(n_total, 1):.1%}); "
          f"by source: {counts}; failed requests: {state['failed']}")
    return df


# ---------------------------------------------------------------------------
# Main collection pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("MLS Attendance Forecasting — Data Collection")
    print("=" * 60)

    # Phase 2a/2b/2c: Stadium & market reference
    stadium_df = collect_stadium_market_data()

    # Phase 1: ASA match data + the venue each match was played in
    matches_df = collect_asa_matches()
    matches_df = attach_venue_data(matches_df)

    # Merge stadium/market onto matches
    merge_cols = ["team_abbreviation", "team_name", "stadium_name", "stadium_capacity",
                  "metro_population", "avg_ticket_price_usd", "conference",
                  "latitude", "longitude", "mls_debut_season"]
    stadium_sub = stadium_df[merge_cols].copy()
    stadium_sub = stadium_sub.rename(columns={
        "team_name": "home_team_name_ref",
        "latitude": "home_lat",
        "longitude": "home_lon",
    })

    # Build name→abbr map for join
    name_to_abbr = dict(zip(stadium_df["team_name"], stadium_df["team_abbreviation"]))
    if "home_team" in matches_df.columns:
        matches_df["home_team_abbr"] = matches_df["home_team"].map(name_to_abbr)
        unmatched = sorted(matches_df.loc[matches_df["home_team_abbr"].isna(), "home_team"].dropna().unique())
        if unmatched:
            print(f"\n  [WARN] Home teams missing from TEAM_REFERENCE (default market data will be used): {unmatched}")
    else:
        matches_df["home_team_abbr"] = None

    merged = matches_df.merge(
        stadium_sub,
        left_on="home_team_abbr",
        right_on="team_abbreviation",
        how="left",
    )

    # Prefer the venue the match was actually played in; fall back to the club's home venue
    merged["capacity_source"] = np.where(merged["venue_capacity"].notna(), "match_venue", "club_home_venue")
    merged["stadium_name"] = merged["venue_name"].fillna(merged["stadium_name"])
    merged["stadium_capacity"] = merged["venue_capacity"].fillna(merged["stadium_capacity"])
    merged["home_lat"] = merged["venue_lat"].fillna(merged["home_lat"])
    merged["home_lon"] = merged["venue_lon"].fillna(merged["home_lon"])
    merged = merged.drop(columns=["venue_name", "venue_capacity", "venue_lat", "venue_lon"])

    # Phase 3: Weather
    merged = collect_weather(merged)

    # Dynamically determine date range
    dates = pd.to_datetime(merged["date"], errors="coerce")
    data_start = dates.dropna().dt.year.min()
    data_end = dates.dropna().dt.year.max()

    # Save
    merged.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)
    print(f"  Total rows collected : {len(merged):,}")
    print(f"  Date range           : {data_start} – {data_end}")
    print(f"  Columns              : {len(merged.columns)}")
    print(f"  Attendance non-null  : {merged['attendance'].notna().sum():,}")
    print(f"  Output saved to      : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
