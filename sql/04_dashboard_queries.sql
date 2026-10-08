-- =============================================================
-- MLS Attendance Forecasting — Dashboard & EDA Summary Queries
-- =============================================================
-- Each named block is exported to data/processed/sql_results/<name>.csv
-- by src/sql_loader.py and consumed by notebooks/eda.ipynb and
-- src/dashboard.py.

-- QUERY: attendance_by_team
-- Average home attendance per team (all matches), sorted descending
SELECT
    home_team,
    COUNT(*)                                AS games_played,
    MIN(season)                             AS first_season,
    MAX(season)                             AS last_season,
    ROUND(AVG(attendance), 0)               AS avg_attendance,
    ROUND(MAX(attendance), 0)               AS max_attendance,
    ROUND(AVG(estimated_revenue), 0)        AS avg_estimated_revenue
FROM matches
GROUP BY home_team
ORDER BY avg_attendance DESC;

-- QUERY: attendance_by_season
-- Average attendance per season; 2021 flagged as capacity-restricted
SELECT
    season,
    COUNT(*)                                AS games_played,
    COUNT(DISTINCT home_team)               AS teams,
    ROUND(AVG(attendance), 0)               AS avg_attendance,
    ROUND(SUM(attendance), 0)               AS total_attendance,
    MAX(is_capacity_restricted)             AS is_restricted_season
FROM matches
GROUP BY season
ORDER BY season;

-- QUERY: attendance_by_month
-- Average attendance by calendar month
SELECT
    month,
    COUNT(*)                                AS games_played,
    ROUND(AVG(attendance), 0)               AS avg_attendance,
    ROUND(AVG(attendance_pct) * 100, 1)     AS avg_utilization_pct
FROM matches
GROUP BY month
ORDER BY month;

-- QUERY: top_rivalries
-- Top 10 home/away matchups by average attendance (min. 5 meetings at that venue),
-- with the lift over the home team's average across all of its home games
SELECT
    mu.home_team || ' vs ' || mu.away_team                      AS matchup,
    mu.home_team,
    mu.away_team,
    mu.games_played,
    ROUND(mu.avg_attendance, 0)                                 AS avg_attendance,
    ROUND(mu.max_attendance, 0)                                 AS max_attendance,
    ROUND((mu.avg_attendance - h.home_avg_attendance)
          / h.home_avg_attendance * 100, 1)                     AS lift_vs_home_avg_pct,
    mu.is_rivalry
FROM (
    SELECT
        home_team,
        away_team,
        COUNT(*)            AS games_played,
        AVG(attendance)     AS avg_attendance,
        MAX(attendance)     AS max_attendance,
        MAX(is_rivalry)     AS is_rivalry
    FROM matches
    GROUP BY home_team, away_team
    HAVING COUNT(*) >= 5
) mu
JOIN (
    SELECT home_team, AVG(attendance) AS home_avg_attendance
    FROM matches
    GROUP BY home_team
) h ON mu.home_team = h.home_team
ORDER BY mu.avg_attendance DESC
LIMIT 10;

-- QUERY: temperature_bucket_attendance
-- Average attendance by match-day maximum temperature bucket (°C)
SELECT
    CASE
        WHEN temperature_max < 10 THEN '1. Cold (<10C / <50F)'
        WHEN temperature_max < 20 THEN '2. Mild (10-20C / 50-68F)'
        WHEN temperature_max < 30 THEN '3. Warm (20-30C / 68-86F)'
        ELSE                           '4. Hot (>=30C / >=86F)'
    END                                     AS temperature_bucket,
    COUNT(*)                                AS games_played,
    ROUND(AVG(attendance), 0)               AS avg_attendance,
    ROUND(AVG(attendance_pct) * 100, 1)     AS avg_utilization_pct
FROM matches
GROUP BY temperature_bucket
ORDER BY temperature_bucket;

-- QUERY: playoff_vs_regular
-- Average attendance: playoff vs regular season
SELECT
    CASE WHEN is_playoff = 1 THEN 'Playoff' ELSE 'Regular Season' END AS match_type,
    COUNT(*)                                AS games_played,
    ROUND(AVG(attendance), 0)               AS avg_attendance,
    ROUND(AVG(attendance_pct) * 100, 1)     AS avg_utilization_pct
FROM matches
GROUP BY match_type
ORDER BY match_type DESC;

-- QUERY: playoff_premium_matched
-- Like-for-like playoff premium: each team-season's playoff home average vs the
-- same team's regular-season home average in that season (removes the bias from
-- playoff hosts being a different mix of clubs than the league as a whole)
SELECT
    COUNT(*)                                                    AS team_seasons_with_playoff_home_game,
    SUM(playoff_games)                                          AS playoff_games,
    ROUND(AVG(regular_avg), 0)                                  AS avg_regular_season_attendance,
    ROUND(AVG(playoff_avg), 0)                                  AS avg_playoff_attendance,
    ROUND(AVG((playoff_avg - regular_avg) / regular_avg) * 100, 1) AS avg_playoff_premium_pct
FROM (
    SELECT
        home_team,
        season,
        AVG(CASE WHEN is_playoff = 1 THEN attendance END)    AS playoff_avg,
        AVG(CASE WHEN is_playoff = 0 THEN attendance END)    AS regular_avg,
        SUM(CASE WHEN is_playoff = 1 THEN 1 ELSE 0 END)      AS playoff_games
    FROM matches
    GROUP BY home_team, season
) team_season
WHERE playoff_games > 0 AND regular_avg IS NOT NULL;

-- QUERY: weekend_vs_weekday
-- Average attendance: weekend (Sat/Sun) vs weekday
SELECT
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END  AS day_type,
    COUNT(*)                                AS games_played,
    ROUND(AVG(attendance), 0)               AS avg_attendance,
    ROUND(AVG(attendance_pct) * 100, 1)     AS avg_utilization_pct
FROM matches
GROUP BY day_type
ORDER BY day_type DESC;

-- QUERY: capacity_utilization
-- Average % of listed venue capacity filled per home team (per match, capped at 105%)
SELECT
    home_team,
    COUNT(DISTINCT stadium_name)                                AS venues_used,
    COUNT(*)                                                    AS games_played,
    ROUND(AVG(attendance), 0)                                   AS avg_attendance,
    ROUND(AVG(attendance_pct) * 100, 1)                         AS avg_utilization_pct,
    ROUND(AVG(CASE WHEN attendance_pct >= 0.98 THEN 1.0 ELSE 0.0 END) * 100, 1) AS near_sellout_pct
FROM matches
GROUP BY home_team
ORDER BY avg_utilization_pct DESC;
