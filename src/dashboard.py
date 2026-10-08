"""
Phase 6 — Interactive Dashboard
Four-tab Plotly Dash app: league overview, team deep-dive, attendance/revenue
forecast tool, and model insights.

Run from the project root:  python src/dashboard.py   ->  http://127.0.0.1:8050
Every loader degrades gracefully: missing data or model files produce an alert
or an empty chart instead of a crash.
"""

import base64
import json
import os
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from src.feature_engineering import RIVALRY_PAIRS
except Exception:  # pragma: no cover - dashboard must still start without it
    RIVALRY_PAIRS = set()

DATA_DIR    = ROOT / "data" / "processed"
SQL_DIR     = DATA_DIR / "sql_results"
MODELS_DIR  = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

DEFAULT_TICKET_PRICE = 55.0
DEFAULT_CAPACITY = 25000.0

# Estimated average ticket prices (USD) — same values as TEAM_REFERENCE in collect_data.py,
# keyed by the team names used in the match data. Teams not listed fall back to $55.
TICKET_PRICES = {
    "Atlanta United FC": 52, "Austin FC": 61, "Chicago Fire FC": 48, "Columbus Crew": 42,
    "Charlotte FC": 55, "Colorado Rapids": 40, "FC Dallas": 43, "D.C. United": 50,
    "Houston Dynamo FC": 41, "LA Galaxy": 65, "Los Angeles FC": 78, "Inter Miami CF": 88,
    "Minnesota United FC": 49, "CF Montréal": 46, "New England Revolution": 53,
    "Nashville SC": 58, "New York Red Bulls": 67, "New York City FC": 82,
    "Orlando City SC": 44, "Philadelphia Union": 46, "Portland Timbers FC": 58,
    "Real Salt Lake": 44, "San Diego FC": 58, "Seattle Sounders FC": 71,
    "San Jose Earthquakes": 47, "Sporting Kansas City": 51, "St. Louis City SC": 54,
    "Toronto FC": 69, "Vancouver Whitecaps FC": 52,
}

MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ACCENT = "#00bc8c"   # DARKLY success green
WARN, DANGER = "#f39c12", "#e74c3c"


# ── Load data ───────────────────────────────────────────────────────────────
def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_sql(name: str) -> pd.DataFrame:
    return load_csv(SQL_DIR / f"{name}.csv")


def try_load(path: Path):
    try:
        return joblib.load(path) if path.exists() else None
    except Exception:
        return None


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


clean = load_csv(DATA_DIR / "mls_matches_clean.csv")
if not clean.empty:
    # Datasets built before the playoff fix have is_playoff = 0 everywhere
    playoff_src = "is_playoff"
    if clean["is_playoff"].sum() == 0 and "knockout_game" in clean.columns:
        playoff_src = "knockout_game"
    clean["playoff"] = clean[playoff_src].astype(int)

att_team   = load_sql("attendance_by_team")
att_season = load_sql("attendance_by_season")
att_month  = load_sql("attendance_by_month")
rivalries  = load_sql("top_rivalries")
cap_util   = load_sql("capacity_utilization")
playoff_df = load_sql("playoff_vs_regular")

xgb_model     = try_load(MODELS_DIR / "xgb_model.pkl")
xgb_lower_m   = try_load(MODELS_DIR / "xgb_lower.pkl")
xgb_upper_m   = try_load(MODELS_DIR / "xgb_upper.pkl")
feature_cols  = try_load(MODELS_DIR / "feature_cols.pkl") or []
metrics       = load_json(MODELS_DIR / "metrics.json")
team_defaults = load_json(MODELS_DIR / "team_feature_defaults.json")

TEAMS = sorted(clean["home_team"].dropna().unique()) if not clean.empty else []
LATEST_SEASON = int(clean["season"].max()) if not clean.empty else None
# Clubs that hosted a match in the latest season — the sensible choices for a forecast
ACTIVE_TEAMS = (
    sorted(clean.loc[clean["season"] == LATEST_SEASON, "home_team"].unique()) if not clean.empty else []
)
# Listed capacity of each club's usual venue in its most recent season (clubs change stadium)
CAPACITY = (
    clean[clean["season"] == clean.groupby("home_team")["season"].transform("max")]
    .groupby("home_team")["stadium_capacity"].agg(lambda s: s.mode().iloc[0]).to_dict()
    if not clean.empty else {}
)
AWAY_QUALITY = clean.groupby("away_team")["away_team_quality"].median().to_dict() if not clean.empty else {}
TEAM_ENC = clean.groupby("home_team")["home_team_enc"].first().to_dict() if not clean.empty else {}
# Each club's most recent road-draw value (see feature_engineering.build_away_team_draw)
AWAY_DRAW = (
    clean.sort_values("date").groupby("away_team")["away_team_draw"].last().to_dict()
    if "away_team_draw" in clean.columns else {}
)


# ── Shared UI helpers ───────────────────────────────────────────────────────
def empty_fig(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=15, color="#adb5bd"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_fig(fig)


def style_fig(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=50, b=40),
        title=title,
        showlegend=False,
    )
    return fig


def kpi_card(label: str, value: str, note: str | None = None) -> dbc.Col:
    body = [html.Div(label, className="text-muted small text-uppercase"),
            html.H3(value, className="mb-0")]
    if note:
        body.append(html.Div(note, className="text-muted small"))
    return dbc.Col(dbc.Card(dbc.CardBody(body), className="h-100"), md=True, sm=6, className="mb-3")


def graph_card(figure: go.Figure) -> dbc.Card:
    return dbc.Card(dbc.CardBody(dcc.Graph(figure=figure, config={"displayModeBar": False})),
                    className="mb-3")


def figure_image(filename: str, caption: str):
    """Embed a PNG from reports/figures as a base64 <img>, or None if it is missing."""
    path = FIGURES_DIR / filename
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode()
    return dbc.Card(dbc.CardBody([
        html.H5(caption),
        html.Img(src=f"data:image/png;base64,{encoded}",
                 style={"maxWidth": "100%", "backgroundColor": "white", "borderRadius": "4px"}),
    ]), className="mb-3")


# ── Tab 1: League Overview ──────────────────────────────────────────────────
def render_overview():
    if clean.empty:
        return dbc.Alert("Match data not found — run `python src/clean_data.py` first.", color="warning")

    kpis = dbc.Row([
        kpi_card("Avg attendance", f"{clean['attendance'].mean():,.0f}", "per match"),
        kpi_card("Seasons", f"{clean['season'].nunique()}",
                 f"{clean['season'].min()}–{clean['season'].max()}, 2020 excluded"),
        kpi_card("Matches", f"{len(clean):,}"),
        kpi_card("Avg capacity filled", f"{clean['attendance_pct'].mean() * 100:.1f}%",
                 "of listed venue capacity, capped at 105% per match"),
    ])

    # Season trend
    season = att_season if not att_season.empty else (
        clean.groupby("season", as_index=False)["attendance"].mean()
        .rename(columns={"attendance": "avg_attendance"}))
    fig_season = px.line(season, x="season", y="avg_attendance", markers=True,
                         labels={"season": "Season", "avg_attendance": "Average attendance"})
    fig_season.update_traces(line_color=ACCENT)
    fig_season.update_xaxes(dtick=1)
    if 2021 in season["season"].values:
        y2021 = float(season.loc[season["season"] == 2021, "avg_attendance"].iloc[0])
        fig_season.add_annotation(x=2021, y=y2021, text="2021: capacity restrictions",
                                  showarrow=True, arrowhead=2, ay=40)
    style_fig(fig_season, "Average attendance by season")

    # Top 10 teams
    teams = att_team if not att_team.empty else (
        clean.groupby("home_team", as_index=False)["attendance"].mean()
        .rename(columns={"attendance": "avg_attendance"}))
    top10 = teams.nlargest(10, "avg_attendance").sort_values("avg_attendance")
    fig_teams = px.bar(top10, x="avg_attendance", y="home_team", orientation="h",
                       labels={"avg_attendance": "Average home attendance", "home_team": ""})
    fig_teams.update_traces(marker_color=ACCENT)
    style_fig(fig_teams, "Top 10 teams by average home attendance")

    # Month pattern
    month = att_month if not att_month.empty else (
        clean.groupby("month", as_index=False)["attendance"].mean()
        .rename(columns={"attendance": "avg_attendance"}))
    month = month.assign(month_name=month["month"].map(MONTH_NAMES))
    fig_month = px.bar(month, x="month_name", y="avg_attendance",
                       labels={"month_name": "Month", "avg_attendance": "Average attendance"},
                       hover_data=[c for c in ["games_played"] if c in month.columns])
    fig_month.update_traces(marker_color=ACCENT)
    style_fig(fig_month, "Average attendance by month (Oct–Dec includes playoffs)")

    return html.Div([
        kpis,
        dbc.Row([dbc.Col(graph_card(fig_season), lg=12)]),
        dbc.Row([dbc.Col(graph_card(fig_teams), lg=6), dbc.Col(graph_card(fig_month), lg=6)]),
    ])


# ── Tab 2: Team Deep-Dive ───────────────────────────────────────────────────
def render_team_tab():
    if clean.empty:
        return dbc.Alert("Match data not found — run `python src/clean_data.py` first.", color="warning")
    default_team = ACTIVE_TEAMS[0] if ACTIVE_TEAMS else TEAMS[0]
    return html.Div([
        dbc.Row(dbc.Col([
            dbc.Label("Select a team"),
            dcc.Dropdown(id="team-dropdown", options=TEAMS, value=default_team,
                         clearable=False, className="text-dark"),
        ], md=4), className="mb-3"),
        html.Div(id="team-content"),
    ])


def team_content(team: str):
    if clean.empty or not team:
        return dbc.Alert("Select a team to see its attendance profile.", color="secondary")
    df = clean[clean["home_team"] == team]
    if df.empty:
        return dbc.Alert(f"No home matches found for {team}.", color="warning")

    best = df.loc[df["attendance"].idxmax()]
    kpis = dbc.Row([
        kpi_card("Avg home attendance", f"{df['attendance'].mean():,.0f}", f"{len(df)} home matches"),
        kpi_card("Highest single game", f"{best['attendance']:,.0f}",
                 f"vs {best['away_team']}, {str(best['date'])[:10]}"),
        kpi_card("Capacity utilization", f"{df['attendance_pct'].mean() * 100:.1f}%",
                 f"current venue lists {CAPACITY.get(team, DEFAULT_CAPACITY):,.0f}"),
    ])

    by_season = df.groupby("season", as_index=False)["attendance"].mean()
    fig_season = px.line(by_season, x="season", y="attendance", markers=True,
                         labels={"season": "Season", "attendance": "Average home attendance"})
    fig_season.update_traces(line_color=ACCENT)
    fig_season.update_xaxes(dtick=1)
    style_fig(fig_season, f"{team} — average home attendance by season")

    # Top 5 visiting opponents by the crowd they draw (prefer opponents with 3+ visits)
    opp = df.groupby("away_team")["attendance"].agg(avg_attendance="mean", visits="size").reset_index()
    frequent = opp[opp["visits"] >= 3]
    opp = (frequent if len(frequent) >= 5 else opp).nlargest(5, "avg_attendance").sort_values("avg_attendance")
    fig_opp = px.bar(opp, x="avg_attendance", y="away_team", orientation="h", hover_data=["visits"],
                     labels={"avg_attendance": "Average attendance when they visit", "away_team": ""})
    fig_opp.update_traces(marker_color=ACCENT)
    fig_opp.add_vline(x=df["attendance"].mean(), line_dash="dash", line_color="#adb5bd",
                      annotation_text="team average", annotation_position="top")
    style_fig(fig_opp, "Top 5 visiting opponents by attendance drawn")

    return html.Div([
        kpis,
        dbc.Row([dbc.Col(graph_card(fig_season), lg=7), dbc.Col(graph_card(fig_opp), lg=5)]),
    ])


# ── Tab 3: Attendance Forecast ──────────────────────────────────────────────
def models_ready() -> bool:
    return xgb_model is not None and bool(feature_cols)


def render_forecast_tab():
    if not models_ready():
        return dbc.Alert("Model not trained yet — run `python src/model.py` first.", color="warning")
    team_options = ACTIVE_TEAMS or TEAMS
    form = dbc.Card(dbc.CardBody([
        html.H5("Match parameters"),
        dbc.Label("Home team"),
        dcc.Dropdown(id="fc-home", options=team_options, value=team_options[0],
                     clearable=False, className="text-dark mb-3"),
        dbc.Label("Away team"),
        dcc.Dropdown(id="fc-away", options=team_options, value=team_options[1],
                     clearable=False, className="text-dark mb-3"),
        dbc.Label("Month"),
        dcc.Dropdown(id="fc-month", options=[{"label": n, "value": m} for m, n in MONTH_NAMES.items()],
                     value=6, clearable=False, className="text-dark mb-3"),
        dbc.Label("Day of week"),
        dcc.Dropdown(id="fc-dow", options=[{"label": d, "value": i} for i, d in enumerate(DAY_NAMES)],
                     value=5, clearable=False, className="text-dark mb-3"),
        dbc.Label("Playoff match?"),
        dcc.Dropdown(id="fc-playoff", options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                     value=0, clearable=False, className="text-dark mb-3"),
        dbc.Label("Expected temperature (°F)"),
        dcc.Slider(id="fc-temp", min=20, max=100, step=1, value=70,
                   marks={t: str(t) for t in range(20, 101, 20)},
                   tooltip={"placement": "bottom", "always_visible": False}),
        html.Div(className="mb-3"),
        dbc.Button("Forecast", id="fc-button", color="success", className="w-100"),
    ]))
    return dbc.Row([
        dbc.Col(form, lg=4, className="mb-3"),
        dbc.Col(html.Div(id="fc-output", children=dbc.Alert(
            "Set the match parameters and click Forecast.", color="secondary")), lg=8),
    ])


def _month_profile(month: int, is_playoff: int) -> dict:
    """Typical matchweek / days-into-season for a month, taken from historical matches."""
    cols = ["matchweek", "days_since_season_start"]
    subset = clean[(clean["month"] == month) & (clean["playoff"] == is_playoff)]
    if subset.empty:
        subset = clean[clean["month"] == month]
    if subset.empty:
        subset = clean
    return subset[cols].median().to_dict()


def build_feature_row(home: str, away: str, month: int, dow: int,
                      is_playoff: int, temp_f: float) -> pd.DataFrame:
    """Team defaults overridden by the user's inputs and everything derived from them."""
    row = dict(team_defaults.get(home, {}))
    temp_c = (temp_f - 32) * 5 / 9
    profile = _month_profile(month, is_playoff)
    matchweek = profile["matchweek"]
    row.update({
        "season": LATEST_SEASON,
        "month": month,
        "day_of_week": dow,
        "is_weekend": int(dow >= 5),
        "is_playoff": int(is_playoff),
        "is_capacity_restricted": 0,
        "temperature_max": temp_c,
        "is_cold": int(temp_c < 4),
        "is_extreme_heat": int(temp_c > 35),
        "is_rivalry": int(frozenset({home, away}) in RIVALRY_PAIRS),
        "away_team_quality": AWAY_QUALITY.get(away, row.get("away_team_quality", 0.0)),
        "away_team_enc": TEAM_ENC.get(away, -1),
        "away_team_draw": AWAY_DRAW.get(away, 1.0),
        "is_alternate_venue": 0,   # forecasts assume the club's usual venue
        "matchweek": matchweek,
        "days_since_season_start": profile["days_since_season_start"],
        # same rules as feature_engineering.build_season_stage
        "stage_Playoff": int(is_playoff),
        "stage_Early": int(not is_playoff and matchweek <= 5),
        "stage_Mid": int(not is_playoff and 5 < matchweek <= 25),
        "stage_Late": int(not is_playoff and matchweek > 25),
    })
    return pd.DataFrame([{col: float(row.get(col, 0.0) or 0.0) for col in feature_cols}])


def forecast(home, away, month, dow, is_playoff, temp_f):
    if not models_ready():
        return dbc.Alert("Model not trained yet — run `python src/model.py` first.", color="warning")
    if clean.empty:
        return dbc.Alert("Match data not found — run `python src/clean_data.py` first.", color="warning")
    if not home or not away:
        return dbc.Alert("Choose both a home and an away team.", color="warning")
    if home == away:
        return dbc.Alert("Home and away teams must be different.", color="warning")

    try:
        X = build_feature_row(home, away, int(month), int(dow), int(is_playoff), float(temp_f))
        capacity = float(CAPACITY.get(home, DEFAULT_CAPACITY))
        pred = max(float(xgb_model.predict(X)[0]), 0.0)
        lower = float(xgb_lower_m.predict(X)[0]) if xgb_lower_m is not None else pred
        upper = float(xgb_upper_m.predict(X)[0]) if xgb_upper_m is not None else pred
        # quantile models are fit independently and can cross the point forecast
        lower = max(min(lower, pred), 0.0)
        upper = max(upper, pred)
    except Exception as exc:
        return dbc.Alert(f"Forecast failed: {exc}", color="danger")

    price = float(TICKET_PRICES.get(home, DEFAULT_TICKET_PRICE))
    pct = pred / capacity * 100 if capacity else 0.0
    color = ACCENT if pct > 80 else WARN if pct >= 60 else DANGER

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=pct, number={"suffix": "%", "valueformat": ".0f"},
        gauge={"axis": {"range": [0, max(105, pct)]}, "bar": {"color": color},
               "steps": [{"range": [0, 60], "color": "rgba(231,76,60,0.15)"},
                         {"range": [60, 80], "color": "rgba(243,156,18,0.15)"},
                         {"range": [80, max(105, pct)], "color": "rgba(0,188,140,0.15)"}]},
    ))
    style_fig(gauge, f"Forecast % of listed capacity ({capacity:,.0f})")
    gauge.update_layout(height=300)

    coverage = metrics.get("quantile_interval_coverage_80pct")
    interval_note = "Nominal 80% interval."
    if coverage is not None:
        interval_note = (f"Nominal 80% interval; it captured {coverage:.0%} of matches in the "
                         f"{metrics.get('test_season')} holdout season"
                         + (", so treat it as optimistic." if coverage < 0.75 else "."))
    return html.Div([
        html.H5(f"{home} vs {away} — {MONTH_NAMES[int(month)]}, {DAY_NAMES[int(dow)]}"
                f"{' (playoff)' if int(is_playoff) else ''}"),
        dbc.Row([
            kpi_card("Predicted attendance", f"{pred:,.0f}"),
            kpi_card("80% prediction interval", f"{lower:,.0f} – {upper:,.0f}"),
            kpi_card("Estimated gate revenue", f"${pred * price:,.0f}",
                     f"at ${price:,.0f} est. avg ticket price"),
        ]),
        graph_card(gauge),
        html.Div(interval_note, className="text-muted small"),
        html.Div("Revenue = predicted attendance × an estimated average ticket price; "
                 "it is not based on club financials.", className="text-muted small"),
    ])


# ── Tab 4: Model Insights ───────────────────────────────────────────────────
def render_model_tab():
    if not metrics:
        return dbc.Alert("Train the model to see insights.", color="warning")

    xgb_m = metrics.get("xgboost", {})
    lr_m = metrics.get("linear_regression", {})
    naive_m = metrics.get("naive_team_average", {})
    coverage = metrics.get("quantile_interval_coverage_80pct")

    def fmt(value, spec):
        return format(value, spec) if value is not None else "—"

    cards = [
        kpi_card("XGBoost MAE", fmt(xgb_m.get("MAE"), ",.0f"), "fans per match"),
        kpi_card("XGBoost R²", fmt(xgb_m.get("R2"), ".3f")),
        kpi_card("Linear Regression MAE", fmt(lr_m.get("MAE"), ",.0f")),
        kpi_card("Linear Regression R²", fmt(lr_m.get("R2"), ".3f")),
        kpi_card("80% interval coverage", fmt(coverage, ".1%"), "nominal target: 80%"),
    ]
    children = [dbc.Row(cards)]

    context = (f"Evaluated on the {metrics.get('test_season')} season "
               f"({metrics.get('n_test', 0):,} matches) after training on "
               f"{metrics.get('n_train', 0):,} earlier matches.")
    if naive_m:
        context += (f" Benchmark — predicting each team's historical average: "
                    f"MAE {naive_m.get('MAE', 0):,.0f}, R² {naive_m.get('R2', 0):.3f}.")
    children.append(html.P(context, className="text-muted"))

    if xgb_model is not None and feature_cols:
        try:
            imp = (pd.DataFrame({"feature": feature_cols, "importance": xgb_model.feature_importances_})
                   .nlargest(15, "importance").sort_values("importance"))
            fig_imp = px.bar(imp, x="importance", y="feature", orientation="h",
                             labels={"importance": "XGBoost importance (gain)", "feature": ""})
            fig_imp.update_traces(marker_color=ACCENT)
            style_fig(fig_imp, "Top 15 features by XGBoost importance")
            fig_imp.update_layout(height=480)
            children.append(graph_card(fig_imp))
        except Exception:
            children.append(dbc.Alert("Feature importances unavailable for the loaded model.",
                                      color="secondary"))

    images = [
        figure_image("shap_summary.png", "SHAP summary — what drives each prediction"),
        figure_image("residual_analysis.png", "Residual analysis — holdout season"),
    ]
    images = [img for img in images if img is not None]
    if images:
        children.append(dbc.Row([dbc.Col(img, lg=6) for img in images]))
    return html.Div(children)


# ── App ─────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                suppress_callback_exceptions=True, assets_folder=str(ROOT / "assets"))
app.title = "MLS Attendance & Revenue Forecasting"
server = app.server

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("⚽ MLS Attendance & Revenue Forecasting",
                            className="text-center my-3 text-light"))),
    dbc.Tabs([
        dbc.Tab(label="📊 League Overview", tab_id="tab-overview"),
        dbc.Tab(label="🏟️ Team Deep-Dive", tab_id="tab-team"),
        dbc.Tab(label="🔮 Attendance Forecast", tab_id="tab-forecast"),
        dbc.Tab(label="🧠 Model Insights", tab_id="tab-model"),
    ], id="tabs", active_tab="tab-overview"),
    html.Div(id="tab-content", className="mt-3"),
], fluid=True)


@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab(active_tab):
    renderers = {
        "tab-overview": render_overview,
        "tab-team": render_team_tab,
        "tab-forecast": render_forecast_tab,
        "tab-model": render_model_tab,
    }
    try:
        return renderers.get(active_tab, render_overview)()
    except Exception as exc:
        return dbc.Alert(f"Could not render this tab: {exc}", color="danger")


@app.callback(Output("team-content", "children"), Input("team-dropdown", "value"))
def update_team(team):
    try:
        return team_content(team)
    except Exception as exc:
        return dbc.Alert(f"Could not load team data: {exc}", color="danger")


@app.callback(
    Output("fc-output", "children"),
    Input("fc-button", "n_clicks"),
    State("fc-home", "value"), State("fc-away", "value"), State("fc-month", "value"),
    State("fc-dow", "value"), State("fc-playoff", "value"), State("fc-temp", "value"),
    prevent_initial_call=True,
)
def update_forecast(n_clicks, home, away, month, dow, is_playoff, temp_f):
    return forecast(home, away, month, dow, is_playoff, temp_f)


if __name__ == "__main__":
    # Local-only by default. Set DASH_HOST=0.0.0.0 to expose on the network and
    # DASH_DEBUG=1 for hot reload (never combine the two on an untrusted network).
    app.run(
        debug=os.environ.get("DASH_DEBUG", "0") == "1",
        host=os.environ.get("DASH_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 8050)),
    )
