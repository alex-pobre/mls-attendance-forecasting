# Data Dictionary — mls_matches_clean.csv

Generated automatically by `src/clean_data.py`.

| field_name | data_type | source | description | example_value | notes |
|---|---|---|---|---|---|
| `away_manager_id` | str | Unknown | Away Manager Id | `Xj5Yre0qbd` | Auto-generated column |
| `away_team_id` | str | Unknown | Away Team Id | `a2lqRX2Mr0` | Auto-generated column |
| `knockout_game` | bool | Unknown | Knockout Game | `False` | Auto-generated column |
| `penalties` | object | Unknown | Penalties | `nan` | Auto-generated column |
| `attendance` | int | ASA API | Official announced attendance figure | `42350` | Target variable |
| `home_team_id` | str | Unknown | Home Team Id | `KAqBN0Vqbg` | Auto-generated column |
| `match_id` | str | ASA/synthetic | Unique match identifier | `asa_00001` |  |
| `expanded_minutes` | int64 | Unknown | Expanded Minutes | `97` | Auto-generated column |
| `status` | str | Unknown | Status | `FullTime` | Auto-generated column |
| `last_updated_utc` | str | Unknown | Last Updated Utc | `2017-12-04 15:31:45 UTC` | Auto-generated column |
| `stadium_id` | str | Unknown | Stadium Id | `gOMnY6mQwN` | Auto-generated column |
| `home_manager_id` | str | Unknown | Home Manager Id | `7vQ7z0dqD1` | Auto-generated column |
| `away_goals` | int | ASA API | Goals scored by away team | `1` |  |
| `referee_id` | str | Unknown | Referee Id | `a35r6KG5L6` | Auto-generated column |
| `matchweek` | int | ASA API | MLS matchday number within the season | `12` |  |
| `season_name` | int64 | Unknown | Season Name | `2017` | Auto-generated column |
| `away_penalties` | float64 | Unknown | Away Penalties | `nan` | Auto-generated column |
| `extra_time` | object | Unknown | Extra Time | `nan` | Auto-generated column |
| `home_goals` | int | ASA API | Goals scored by home team | `2` |  |
| `home_penalties` | float64 | Unknown | Home Penalties | `nan` | Auto-generated column |
| `date` | date | ASA API | Match date in the home market's local time | `2023-04-15` | Converted from UTC kickoff |
| `home_team` | str | ASA API | Home team full name | `Atlanta United FC` |  |
| `away_team` | str | ASA API | Away team full name | `LA Galaxy` |  |
| `kickoff_utc` | str | ASA API | Kickoff timestamp in UTC | `2023-04-15 23:30` |  |
| `season` | int | ASA API | MLS season year | `2023` | ASA season label |
| `is_playoff` | int | ASA API | 1 if match is a playoff/knockout fixture, 0 otherwise | `0` | From ASA knockout_game |
| `stage` | str | Unknown | Stage | `Regular Season` | Auto-generated column |
| `home_team_abbr` | str | Unknown | Home Team Abbr | `ATL` | Auto-generated column |
| `team_abbreviation` | str | Unknown | Team Abbreviation | `ATL` | Auto-generated column |
| `home_team_name_ref` | str | Unknown | Home Team Name Ref | `Atlanta United FC` | Auto-generated column |
| `stadium_name` | str | ASA API/Reference | Venue the match was played in | `Mercedes-Benz Stadium` | Club home venue if ASA has no venue |
| `stadium_capacity` | int | ASA API/Wikipedia/hardcoded | Listed capacity of the venue the match was played in | `42500` | Often the reduced soccer configuration; attendance can exceed it |
| `metro_population` | int | Census Bureau/hardcoded | 2023 metro area population estimate | `6144050` | MSA or CMA estimate |
| `avg_ticket_price_usd` | float | Estimated (secondary market) | Estimated average ticket price in USD | `52.0` | Secondary market estimate — not face value |
| `conference` | str | Reference | MLS conference (East/West) | `East` |  |
| `home_lat` | float | ASA API/Reference | Venue latitude | `33.7554` |  |
| `home_lon` | float | ASA API/Reference | Venue longitude | `-84.4009` |  |
| `mls_debut_season` | int | Reference | First season the club competed in MLS | `2017` |  |
| `capacity_source` | str | Derived | Where stadium_capacity came from | `match_venue` | match_venue or club_home_venue |
| `temperature_max` | float | Open-Meteo | Daily maximum temperature in °C at the venue | `28.5` | See weather_source |
| `precipitation_sum` | float | Open-Meteo | Daily total precipitation in mm at the venue | `0.0` | See weather_source |
| `windspeed_max` | float | Open-Meteo | Daily maximum windspeed in km/h at the venue | `12.3` | See weather_source |
| `weather_source` | str | Derived | Where the weather values came from | `open_meteo` | open_meteo, missing (imputed) or synthetic |
| `is_capacity_restricted` | int | Derived | 1 for all 2021 matches (COVID attendance caps) | `0` | Binary flag — actual per-game caps varied by city |
| `day_of_week` | int | Derived | Day of week (0=Mon, 6=Sun) | `5` |  |
| `month` | int | Derived | Calendar month (1–12) | `7` |  |
| `is_weekend` | int | Derived | 1 if match is on Saturday or Sunday | `1` |  |
| `days_since_season_start` | int | Derived | Days elapsed since first match of the season | `85` |  |
| `is_rainy` | int | Derived | 1 if precipitation_sum > 5 mm | `0` |  |
| `is_extreme_heat` | int | Derived | 1 if temperature_max > 35°C (95°F) | `0` |  |
| `is_cold` | int | Derived | 1 if temperature_max < 4°C (40°F) | `0` |  |
| `attendance_pct` | float | Derived | attendance / stadium_capacity, capped at 1.05 | `0.87` | Alternative target variable |
| `estimated_revenue` | float | Derived | attendance × avg_ticket_price_usd (estimated) | `2202200.0` | Estimated — not from club financials |
| `goal_diff` | int | Derived | home_goals − away_goals | `1` |  |
| `is_rivalry` | int | Derived | 1 if match is a known rivalry fixture | `0` | Based on hardcoded rivalry pair list |
| `season_stage` | str | Derived | Season phase (Early/Mid/Late/Playoff) | `Mid` | Based on matchweek |
| `days_since_last_home_match` | float | Derived | Days since home team's previous home match (same season) | `14.0` | First match of season gets prior of 14 days |
| `home_team_form` | float | Derived | Rolling 5-match points total for home team (max 15) | `9.0` | 3=W, 1=D, 0=L |
| `away_team_quality` | float | Derived | Away team season win% up to this match (proxy for opponent draw) | `0.48` |  |
| `away_team_draw` | float | Derived | Visiting club's recent road draw: mean of attendance / host's prior home average over its previous 10 away matches | `1.08` | 1.0 = hosts drew their usual crowd; uses earlier matches only |
| `is_alternate_venue` | int | Derived | 1 if played outside the home club's usual venue that season | `0` | Known in advance from the schedule |
| `market_size_tier` | str | Derived | Large (>5M pop), Medium (1M–5M), Small (<1M) | `Large` |  |
| `conf_East` | int64 | Unknown | Conf East | `1` | Auto-generated column |
| `conf_West` | int64 | Unknown | Conf West | `0` | Auto-generated column |
| `tier_Large` | int64 | Unknown | Tier Large | `1` | Auto-generated column |
| `tier_Medium` | int64 | Unknown | Tier Medium | `0` | Auto-generated column |
| `stage_Early` | int64 | Unknown | Stage Early | `1` | Auto-generated column |
| `stage_Late` | int64 | Unknown | Stage Late | `0` | Auto-generated column |
| `stage_Mid` | int64 | Unknown | Stage Mid | `0` | Auto-generated column |
| `stage_Playoff` | int64 | Unknown | Stage Playoff | `0` | Auto-generated column |
| `home_team_enc` | int | Derived | Label-encoded home_team (for tree models) | `3` | Ordinal — not meaningful beyond tree models |
| `away_team_enc` | int | Derived | Label-encoded away_team, same codes as home_team_enc | `12` | Ordinal — not meaningful beyond tree models |