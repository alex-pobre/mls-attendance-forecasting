-- =============================================================
-- MLS Attendance Forecasting — Schema Definition (SQLite)
-- =============================================================

DROP TABLE IF EXISTS matches;
DROP TABLE IF EXISTS teams;

-- -------------------------------------------------------------
-- matches: cleaned match-level records
-- -------------------------------------------------------------
CREATE TABLE matches (
    match_id                    TEXT PRIMARY KEY,
    season                      INTEGER,
    date                        TEXT,
    home_team                   TEXT,
    away_team                   TEXT,
    home_goals                  INTEGER,
    away_goals                  INTEGER,
    attendance                  INTEGER,
    attendance_pct              REAL,
    estimated_revenue           REAL,
    matchweek                   INTEGER,
    is_playoff                  INTEGER,
    is_capacity_restricted      INTEGER,
    is_rivalry                  INTEGER,
    is_weekend                  INTEGER,
    day_of_week                 INTEGER,
    month                       INTEGER,
    days_since_season_start     INTEGER,
    season_stage                TEXT,
    home_team_form              REAL,
    away_team_quality           REAL,
    days_since_last_home_match  REAL,
    temperature_max             REAL,
    precipitation_sum           REAL,
    windspeed_max               REAL,
    is_rainy                    INTEGER,
    is_extreme_heat             INTEGER,
    is_cold                     INTEGER,
    stadium_capacity            INTEGER,
    metro_population            INTEGER,
    avg_ticket_price_usd        REAL,
    conference                  TEXT,
    market_size_tier            TEXT,
    goal_diff                   INTEGER,
    home_team_enc               INTEGER,
    away_team_enc               INTEGER,
    away_team_draw              REAL,
    is_alternate_venue          INTEGER
);

-- -------------------------------------------------------------
-- teams: stadium and market reference data
-- -------------------------------------------------------------
CREATE TABLE teams (
    team_abbreviation   TEXT PRIMARY KEY,
    team_name           TEXT,
    stadium_name        TEXT,
    city                TEXT,
    state               TEXT,
    founding_year       INTEGER,
    conference          TEXT,
    latitude            REAL,
    longitude           REAL,
    mls_debut_season    INTEGER,
    avg_ticket_price_usd REAL,
    stadium_capacity    INTEGER,
    metro_population    INTEGER
);
