# Executive Summary: MLS Attendance & Ticket Revenue Forecasting
**Prepared by:** Rod Alexander Pobre  
**Date:** October 2026  
**Institution:** Villanova University, M.S. Business Analytics & AI  

---

## Business Question
> *"What factors drive MLS match attendance, and how can we accurately forecast attendance 
> for future matches to support stadium operations, marketing spend allocation, and 
> dynamic pricing decisions?"*

## Dataset Overview
- **5,403** matches from **2013** to **2026** (5,185 regular-season and 218 playoff; 2026 covers matches through October 1)
- Data sources: American Soccer Analysis (matches and venues, via itscalledsoccer), Open-Meteo historical weather archive, Wikipedia stadium data, hardcoded metro population and ticket-price estimates
- 2020 season removed (COVID-19 — zero attendance); 2021 retained with `is_capacity_restricted` flag
- Each match carries the venue it was actually played in (55 venues), its local calendar date, and observed weather at that venue
- **27** features used by the model: rolling form, opponent quality and recent road draw, rivalry and playoff flags, venue capacity, calendar, market size, weather
- Three post-match columns (`attendance_pct`, `estimated_revenue`, `goal_diff`) are deliberately excluded from the model because they are only known after the match

## Key Findings

### Attendance Patterns
1. **League average attendance is 21,345 per match** (2013–2026), with a median of 19,796. The peak season was 2024 at 23,353.  
2. **Playoff matches draw 17.8% more fans** than regular-season matches in the raw data (24,963 vs 21,192). Compared like-for-like — the same club in the same season — the premium is **8.1%**, because playoff hosts are already the better-supported clubs.  
3. **Atlanta United FC** consistently leads the league in home attendance (47,821 per match); **FC Dallas** trails among current clubs (15,207).  
4. **Summer months (June–August) see 4.2% higher attendance** than the spring window (March–May) in the regular season.  
5. **Weekend matches outperform weekday matches by approximately 8.1%** (21,686 vs 20,066).  
6. **The home club matters far more than the visitor.** Average attendance ranges by 40,107 across home clubs but by 12,086 across visiting clubs. The exception is Inter Miami CF: since August 2023 its away matches have drawn **58% more** than the host's season average (56 matches), partly because hosts move those games to larger stadiums.  
7. **Weather has a modest effect on announced attendance.** Relative to the same club's season average, matches played below 40°F draw 3.8% less (79 matches) and rainy days (more than 5 mm) draw 1.9% less (900 matches); heat of 90°F or more shows no measurable effect (494 matches).

### Modeling Results
Evaluated out-of-time: trained on 4,480 matches from seasons before 2025, tested on all 524 matches of the 2025 season.

| Model | MAE | R² |
|---|---|---|
| Naive benchmark (each team's historical average) | 3,393 | 0.410 |
| Linear Regression (baseline) | 4,083 | 0.334 |
| XGBoost (tuned) | **2,461** | **0.735** |

- XGBoost outperformed the linear baseline by **39.7% MAE reduction**, and the naive team-average benchmark by 27.5%
- The 80% quantile prediction interval captured **79.6%** of actual test observations, in line with its nominal level
- Top predictive features (SHAP): **stadium_capacity, home_team_enc, metro_population** — i.e. *where* and *who* is hosting. Among match-level factors, the visiting club's recent road draw (`away_team_draw`) matters most: removing the away-team features drops R² on the same holdout from about 0.74 to about 0.58
- On the 399 matches of the in-progress 2026 season, the same model scores MAE 3,117 / R² 0.658
- Matches moved away from a club's usual venue remain the hardest to forecast (12 such matches in 2025, with errors about 60% larger than at usual venues)

### Revenue Implications
- At the league average ticket price of ~$55, a **1,000-fan attendance improvement** yields an estimated **$55,000 in incremental gate revenue** per match
- The model's typical error of 2,461 fans corresponds to roughly **$135,000 of gate revenue per match** at that price — a useful measure of how much uncertainty remains in any single-match revenue forecast
- Ticket prices in this project are estimated averages, not club financials, so all revenue figures are directional

## Recommendations

1. **Invest in marquee opponent scheduling** — Inter Miami's visits draw 58% more fans than the host's season average, and designated rivalry fixtures draw about 15% more (323 matches). Clubs should advocate for weekend home fixtures against these opponents, price them accordingly, and plan early for a larger venue where one is available.

2. **Protect weekend slots before worrying about weather** — the weekend premium (8.1%) is larger than any weather effect in the data. Cold (below 40°F) and rain cost roughly 2–4% of announced attendance, and announced attendance counts tickets distributed, so the real effect on match-day turnout is probably larger. Weather-based offers are best aimed at reducing no-shows for early- and late-season fixtures rather than at headline pricing.

3. **Use the forecast dashboard for staffing decisions, with a margin** — the point forecast is typically within about 2,500 fans, and the 80% interval held up on the 2025 holdout (80% coverage). Operations teams can staff to the interval midpoint and keep flex contracts for the upper bound, with extra margin for matches moved to a larger venue.

4. **Treat 2021 as a separate regime, not a demand signal** — average attendance fell to 17,194 under capacity restrictions and recovered to 21,040 in 2022, about 97% of the 2019 level (21,703). The `is_capacity_restricted` flag lets the model learn from 2021 without mistaking it for weak demand.

5. **Expand to secondary market ticket pricing** — the current model uses estimated ticket prices. Integrating StubHub/SeatGeek secondary market data as a feature would unlock dynamic revenue forecasting rather than simple attendance × price multiplication.

## Data Limitations
- **Announced attendance.** MLS figures are tickets distributed, not turnstile counts, which mutes weather and weekday effects.
- **Listed capacity varies in meaning.** Capacity is the figure ASA lists for each venue — sometimes the full bowl, sometimes a reduced soccer configuration — so 338 matches exceed 105% of listed capacity. Attendance is kept as reported; utilization percentages are capped at 105% and only roughly comparable across clubs.
- **Estimated market inputs.** Ticket prices are estimates (FC Cincinnati and Chivas USA use a $55 placeholder), and metro populations are hardcoded 2023 figures because the Census API request did not return data.
- **Unreported attendance.** 150 matches with a reported attendance of zero (89 of them in 2021, 58 in 2022–2025) and 6 with no figure at all are removed, so recent-season averages rest on slightly incomplete data.

## Technical Notes
- All analysis is reproducible: `pip install -r requirements-dev.txt && python src/model.py`
- Source code: GitHub repository linked in README
- Model retrained automatically via GitHub Actions CI on every push to `main`
- The models served by the dashboard are refit on all 5,403 matches after evaluation; the metrics above come from the 2025 holdout only
