-- =============================================================
-- MLS Attendance Forecasting — Exploratory SQL Queries
-- =============================================================

-- QUERY: avg_attendance_by_team
-- Average attendance, capacity utilization %, and estimated revenue by home team
SELECT
    m.home_team,
    t.conference,
    t.stadium_name,
    t.stadium_capacity,
    COUNT(*)                                        AS games_played,
    ROUND(AVG(m.attendance), 0)                     AS avg_attendance,
    ROUND(AVG(m.attendance_pct) * 100, 1)           AS avg_utilization_pct,
    ROUND(AVG(m.estimated_revenue), 0)              AS avg_estimated_revenue,
    ROUND(MAX(m.attendance), 0)                     AS max_attendance
FROM matches m
LEFT JOIN teams t ON m.home_team = t.team_name
WHERE m.is_playoff = 0
GROUP BY m.home_team, t.conference, t.stadium_name, t.stadium_capacity
ORDER BY avg_attendance DESC;

-- QUERY: attendance_trend_by_year
-- Average attendance per season year with YoY change; 2021 flagged as restricted
SELECT
    season,
    COUNT(*)                                        AS games_played,
    ROUND(AVG(attendance), 0)                       AS avg_attendance,
    ROUND(AVG(attendance_pct) * 100, 1)             AS avg_utilization_pct,
    MAX(is_capacity_restricted)                     AS is_restricted_season,
    ROUND(
        (AVG(attendance) - LAG(AVG(attendance)) OVER (ORDER BY season))
        / LAG(AVG(attendance)) OVER (ORDER BY season) * 100,
        1
    )                                               AS yoy_change_pct
FROM matches
WHERE is_playoff = 0
GROUP BY season
ORDER BY season;

-- QUERY: day_of_week_effect
-- Average attendance and utilization by day of week
SELECT
    CASE day_of_week
        WHEN 0 THEN 'Monday'
        WHEN 1 THEN 'Tuesday'
        WHEN 2 THEN 'Wednesday'
        WHEN 3 THEN 'Thursday'
        WHEN 4 THEN 'Friday'
        WHEN 5 THEN 'Saturday'
        WHEN 6 THEN 'Sunday'
    END                                             AS day_name,
    day_of_week,
    COUNT(*)                                        AS sample_count,
    ROUND(AVG(attendance), 0)                       AS avg_attendance,
    ROUND(AVG(attendance_pct) * 100, 1)             AS avg_utilization_pct
FROM matches
WHERE is_playoff = 0
GROUP BY day_of_week
ORDER BY day_of_week;

-- QUERY: weather_impact
-- Average attendance for different weather conditions
SELECT
    'All Games'                                     AS condition_type,
    'All'                                           AS condition_value,
    COUNT(*)                                        AS sample_count,
    ROUND(AVG(attendance), 0)                       AS avg_attendance
FROM matches WHERE is_playoff = 0

UNION ALL

SELECT 'Rain', 'Rainy', COUNT(*), ROUND(AVG(attendance), 0)
FROM matches WHERE is_rainy = 1 AND is_playoff = 0

UNION ALL

SELECT 'Rain', 'Dry', COUNT(*), ROUND(AVG(attendance), 0)
FROM matches WHERE is_rainy = 0 AND is_playoff = 0

UNION ALL

SELECT 'Temperature', 'Extreme Heat (>35C)', COUNT(*), ROUND(AVG(attendance), 0)
FROM matches WHERE is_extreme_heat = 1 AND is_playoff = 0

UNION ALL

SELECT 'Temperature', 'Normal', COUNT(*), ROUND(AVG(attendance), 0)
FROM matches WHERE is_extreme_heat = 0 AND is_cold = 0 AND is_playoff = 0

UNION ALL

SELECT 'Temperature', 'Cold (<4C)', COUNT(*), ROUND(AVG(attendance), 0)
FROM matches WHERE is_cold = 1 AND is_playoff = 0

ORDER BY condition_type, condition_value;

-- QUERY: rivalry_premium
-- Average attendance for rivalry vs non-rivalry matches by team
SELECT
    home_team,
    COUNT(CASE WHEN is_rivalry = 1 THEN 1 END)              AS rivalry_games,
    ROUND(AVG(CASE WHEN is_rivalry = 1 THEN attendance END), 0) AS avg_rivalry_attendance,
    COUNT(CASE WHEN is_rivalry = 0 THEN 1 END)              AS non_rivalry_games,
    ROUND(AVG(CASE WHEN is_rivalry = 0 THEN attendance END), 0) AS avg_non_rivalry_attendance,
    ROUND(
        (AVG(CASE WHEN is_rivalry = 1 THEN attendance END)
         - AVG(CASE WHEN is_rivalry = 0 THEN attendance END))
        / AVG(CASE WHEN is_rivalry = 0 THEN attendance END) * 100,
        1
    )                                                        AS rivalry_lift_pct
FROM matches
WHERE is_playoff = 0
GROUP BY home_team
HAVING rivalry_games > 0
ORDER BY rivalry_lift_pct DESC;

-- QUERY: top_10_attended_matches
-- Top 10 all-time attendance records
SELECT
    m.date,
    m.home_team,
    m.away_team,
    m.season,
    m.matchweek,
    m.attendance,
    ROUND(m.attendance_pct * 100, 1)    AS utilization_pct,
    m.is_rivalry,
    m.is_playoff,
    ROUND(m.temperature_max, 1)         AS temp_c,
    m.is_rainy,
    ROUND(m.estimated_revenue, 0)       AS estimated_revenue
FROM matches m
ORDER BY attendance DESC
LIMIT 10;

-- QUERY: market_size_attendance
-- Average attendance grouped by market size tier
SELECT
    market_size_tier,
    COUNT(DISTINCT home_team)               AS teams_in_tier,
    COUNT(*)                                AS total_games,
    ROUND(AVG(attendance), 0)               AS avg_attendance,
    ROUND(AVG(attendance_pct) * 100, 1)     AS avg_utilization_pct,
    ROUND(AVG(avg_ticket_price_usd), 2)     AS avg_ticket_price,
    ROUND(AVG(estimated_revenue), 0)        AS avg_estimated_revenue
FROM matches
WHERE is_playoff = 0
GROUP BY market_size_tier
ORDER BY avg_attendance DESC;

-- QUERY: form_vs_attendance
-- Average attendance binned by home team's recent form (rolling 5-match points)
SELECT
    CASE
        WHEN home_team_form BETWEEN 0  AND 3  THEN '0-3 pts (Poor)'
        WHEN home_team_form BETWEEN 4  AND 6  THEN '4-6 pts (Average)'
        WHEN home_team_form BETWEEN 7  AND 9  THEN '7-9 pts (Good)'
        WHEN home_team_form BETWEEN 10 AND 15 THEN '10-15 pts (Excellent)'
        ELSE 'Unknown'
    END                                 AS form_bin,
    COUNT(*)                            AS sample_count,
    ROUND(AVG(attendance), 0)           AS avg_attendance,
    ROUND(AVG(attendance_pct) * 100, 1) AS avg_utilization_pct
FROM matches
WHERE is_playoff = 0
GROUP BY form_bin
ORDER BY avg_attendance DESC;
