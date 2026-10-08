-- =============================================================
-- MLS Attendance Forecasting — Feature Export Query
-- =============================================================
-- Produces the final modeling feature table: a clean join of all
-- engineered features used for prediction with attendance targets.
-- QUERY: model_features

SELECT
    m.match_id,
    m.season,
    m.date,
    m.home_team,
    m.away_team,

    -- Targets
    m.attendance,
    m.attendance_pct,
    m.estimated_revenue,

    -- Match context
    m.matchweek,
    m.is_playoff,
    m.is_rivalry,
    m.is_capacity_restricted,
    m.season_stage,

    -- Date features
    m.day_of_week,
    m.month,
    m.is_weekend,
    m.days_since_season_start,

    -- Weather features
    m.temperature_max,
    m.precipitation_sum,
    m.windspeed_max,
    m.is_rainy,
    m.is_extreme_heat,
    m.is_cold,

    -- Team / form features
    m.home_team_form,
    m.away_team_quality,
    m.days_since_last_home_match,
    m.home_team_enc,
    m.away_team_enc,
    m.away_team_draw,
    m.is_alternate_venue,

    -- Market / stadium features
    m.stadium_capacity,
    m.metro_population,
    m.avg_ticket_price_usd,
    m.conference,
    m.market_size_tier

FROM matches m
WHERE m.attendance IS NOT NULL
ORDER BY m.season, m.date, m.home_team;
