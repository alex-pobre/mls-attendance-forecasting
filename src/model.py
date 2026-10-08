"""
Phase 5 — Modeling
Trains a Linear Regression baseline and a tuned XGBoost attendance model with
80% quantile prediction intervals, explains it with SHAP, and saves all
artifacts to models/ and reports/figures/.

Evaluation is strictly out-of-time: the last *complete* season is held out as
the test set and only earlier seasons are used for training/tuning. The models
saved for the dashboard are then refit on every available match.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FEATURES_PATH = ROOT / "data" / "processed" / "model_features.csv"
CLEAN_PATH    = ROOT / "data" / "processed" / "mls_matches_clean.csv"
MODELS_DIR    = ROOT / "models"
FIGURES_DIR   = ROOT / "reports" / "figures"
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "attendance"
RANDOM_STATE = 42

# Identifier / text columns that are never features
NON_FEATURES = ["match_id", "game_id", "date", "home_team", "away_team",
                "stadium_name", "city", "state", TARGET]

# Columns that are only known *after* the match — using them would leak the answer:
#   attendance_pct    = attendance / stadium_capacity
#   estimated_revenue = attendance × avg_ticket_price_usd
#   goal_diff         = final score margin
LEAKAGE_COLS = ["attendance_pct", "estimated_revenue", "goal_diff"]

PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the modeling table (repairing the playoff flag if an older dataset lacks it)."""
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print("Columns:", df.columns.tolist())

    # Datasets built before the playoff fix have is_playoff = 0 everywhere;
    # fall back to ASA's `knockout_game` flag in that case.
    if df.get("is_playoff", pd.Series(dtype=int)).sum() == 0 and CLEAN_PATH.exists():
        clean = pd.read_csv(CLEAN_PATH, usecols=lambda c: c in ("match_id", "knockout_game"))
        if "knockout_game" in clean.columns:
            ko = clean.set_index("match_id")["knockout_game"].astype(int)
            df["is_playoff"] = df["match_id"].map(ko).fillna(0).astype(int)
            print(f"is_playoff rebuilt from knockout_game: {int(df['is_playoff'].sum())} playoff matches")

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def select_features(df: pd.DataFrame) -> list[str]:
    """All numeric columns except identifiers, the target and post-match leakage columns."""
    excluded = set(NON_FEATURES) | set(LEAKAGE_COLS)
    cols = [c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    leaked = [c for c in LEAKAGE_COLS if c in df.columns]
    print(f"\nExcluded as target leakage: {leaked}")
    return cols


def time_split(df: pd.DataFrame):
    """
    Hold out the last complete season. The latest season counts as complete only
    once it has reached the post-season (playoff matches played, or a match in
    November/December); otherwise it is kept aside as a secondary, partial holdout.
    """
    latest = int(df["season"].max())
    latest_rows = df[df["season"] == latest]
    latest_complete = bool(latest_rows["is_playoff"].sum() > 0 or latest_rows["date"].max().month >= 11)
    test_season = latest if latest_complete else int(df.loc[df["season"] < latest, "season"].max())
    train_mask = df["season"] < test_season
    test_mask = df["season"] == test_season
    later_mask = df["season"] > test_season
    return test_season, train_mask, test_mask, later_mask


def drop_redundant_dummies(X_df: pd.DataFrame) -> list[str]:
    """
    Resolve exact collinearity before the VIF check, so that VIF does not have to
    guess which of several perfectly collinear columns to remove:
      - a column that duplicates an earlier one (e.g. stage_Playoff == is_playoff)
      - one reference level of each complete one-hot group (the most common level)
    """
    redundant = []
    cols = X_df.columns.tolist()
    for i, col in enumerate(cols):
        if any(X_df[col].equals(X_df[prev]) for prev in cols[:i] if prev not in redundant):
            redundant.append(col)
    for prefix in ("conf_", "tier_", "stage_"):
        group = [c for c in cols if c.startswith(prefix) and c not in redundant]
        if len(group) > 1 and (X_df[group].sum(axis=1) + X_df[[c for c in cols if c.startswith(prefix) and c in redundant]].sum(axis=1) == 1).all():
            redundant.append(X_df[group].sum().idxmax())
    return redundant


def check_vif(X_df: pd.DataFrame, threshold: float = 10.0):
    """Iteratively drop the feature with the highest VIF until all are <= threshold."""
    dropped = []
    X_check = X_df.astype(float).copy()
    while X_check.shape[1] > 1:
        Xc = add_constant(X_check, has_constant="add")
        vif = pd.Series(
            [variance_inflation_factor(Xc.values, i + 1) for i in range(X_check.shape[1])],
            index=X_check.columns,
        ).fillna(np.inf)
        max_vif = vif.max()
        if max_vif <= threshold:
            break
        col = vif.idxmax()
        dropped.append((col, "inf" if np.isinf(max_vif) else round(float(max_vif), 1)))
        X_check = X_check.drop(columns=[col])
    if dropped:
        print(f"VIF: dropped {len(dropped)} features: {dropped}")
    return X_check.columns.tolist(), dropped


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def save_shap_plots(model, X_test: pd.DataFrame) -> list[str]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance — XGBoost Attendance Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Mean |SHAP| by Feature — XGBoost Attendance Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    top3 = (
        pd.DataFrame({"feature": X_test.columns,
                      "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .nlargest(3, "mean_abs_shap")["feature"].tolist()
    )
    for feat in top3:
        shap.dependence_plot(feat, shap_values, X_test, show=False)
        plt.tight_layout()
        safe_name = feat.replace("/", "_").replace(" ", "_")
        plt.savefig(FIGURES_DIR / f"shap_dep_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
    print(f"SHAP plots saved. Top 3 features: {top3}")
    return top3


def save_residual_plots(y_test: pd.Series, preds: np.ndarray, test_season: int) -> None:
    residuals = y_test.values - preds
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(preds, residuals, alpha=0.4, color="#2196F3")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Attendance")
    axes[0].set_ylabel("Residual (actual − predicted)")
    axes[0].set_title("Residuals vs Predicted")
    axes[1].hist(residuals, bins=40, color="#2196F3", edgecolor="white")
    axes[1].set_xlabel("Residual (actual − predicted)")
    axes[1].set_title("Residual Distribution")
    fig.suptitle(f"XGBoost residuals — {test_season} holdout season")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residual_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, preds, alpha=0.4, color="#2196F3")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, color="red", linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual Attendance")
    ax.set_ylabel("Predicted Attendance")
    ax.set_title(f"Predicted vs Actual — {test_season} holdout season")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Residual and predicted-vs-actual plots saved.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("MLS Attendance Forecasting — Model Training")
    print("=" * 60)

    df = load_data()
    feature_cols = select_features(df)
    print(f"Candidate features ({len(feature_cols)}): {feature_cols}")

    # ---- Time-based split -------------------------------------------------
    test_season, train_mask, test_mask, later_mask = time_split(df)
    medians = df.loc[train_mask, feature_cols].median()
    X_all = df[feature_cols].fillna(medians)
    y_all = df[TARGET]

    # ---- Drop constant columns, then VIF (fit on training rows only) ------
    constant = [c for c in feature_cols if X_all.loc[train_mask, c].nunique() <= 1]
    if constant:
        print(f"Dropped zero-variance features: {constant}")
    redundant = drop_redundant_dummies(X_all.loc[train_mask, [c for c in feature_cols if c not in constant]])
    if redundant:
        print(f"Dropped redundant dummy columns (duplicates / reference levels): {redundant}")
    candidates = [c for c in feature_cols if c not in constant and c not in redundant]
    feature_cols_final, vif_dropped = check_vif(X_all.loc[train_mask, candidates])
    print(f"\nFinal features ({len(feature_cols_final)}): {feature_cols_final}")

    X_train, y_train = X_all.loc[train_mask, feature_cols_final], y_all[train_mask]
    X_test, y_test = X_all.loc[test_mask, feature_cols_final], y_all[test_mask]
    print(f"Train: {len(X_train):,} rows (seasons < {test_season}) | "
          f"Test (season {test_season}): {len(X_test):,} rows")

    # ---- Naive benchmark: each team's training-period average -------------
    team_means = df.loc[train_mask].groupby("home_team")[TARGET].mean()
    naive_preds = df.loc[test_mask, "home_team"].map(team_means).fillna(y_train.mean()).values
    naive_mae = mean_absolute_error(y_test, naive_preds)
    naive_r2 = r2_score(y_test, naive_preds)
    print(f"\nNaive (team historical average) — MAE: {naive_mae:,.0f} | R²: {naive_r2:.3f}")

    # ---- Baseline: Linear Regression --------------------------------------
    scaler = StandardScaler()
    lr = LinearRegression().fit(scaler.fit_transform(X_train), y_train)
    lr_preds = lr.predict(scaler.transform(X_test))
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_r2 = r2_score(y_test, lr_preds)
    print(f"Linear Regression — MAE: {lr_mae:,.0f} | R²: {lr_r2:.3f}")

    # ---- XGBoost with time-series cross-validated grid search -------------
    grid = GridSearchCV(
        XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
        PARAM_GRID,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    print(f"\nBest XGBoost params: {best_params}")

    xgb = XGBRegressor(**best_params, random_state=RANDOM_STATE, verbosity=0).fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)
    print(f"XGBoost — MAE: {xgb_mae:,.0f} | R²: {xgb_r2:.3f}")
    print(f"Improvement over linear baseline: {(lr_mae - xgb_mae) / lr_mae * 100:.1f}% MAE reduction")

    # ---- Quantile regression: 80% prediction interval ---------------------
    def quantile_model(alpha: float) -> XGBRegressor:
        return XGBRegressor(**best_params, objective="reg:quantileerror",
                            quantile_alpha=alpha, random_state=RANDOM_STATE, verbosity=0)

    xgb_lower = quantile_model(0.10).fit(X_train, y_train)
    xgb_upper = quantile_model(0.90).fit(X_train, y_train)
    lower_preds, upper_preds = xgb_lower.predict(X_test), xgb_upper.predict(X_test)
    coverage = float(((y_test.values >= lower_preds) & (y_test.values <= upper_preds)).mean())
    interval_width = float(np.mean(upper_preds - lower_preds))
    print(f"\nQuantile regression — 80% interval coverage on test set: {coverage:.1%} "
          f"(mean width {interval_width:,.0f})")

    # ---- Explainability & diagnostics (evaluation model, holdout season) --
    top3 = save_shap_plots(xgb, X_test)
    save_residual_plots(y_test, xgb_preds, test_season)

    # ---- Secondary check on the in-progress season, if any ----------------
    partial_holdout = None
    if later_mask.any():
        X_late, y_late = X_all.loc[later_mask, feature_cols_final], y_all[later_mask]
        late_preds = xgb.predict(X_late)
        partial_holdout = {
            "seasons": sorted(int(s) for s in df.loc[later_mask, "season"].unique()),
            "n": int(later_mask.sum()),
            "xgboost_MAE": round(float(mean_absolute_error(y_late, late_preds)), 1),
            "xgboost_R2": round(float(r2_score(y_late, late_preds)), 4),
        }
        print(f"Partial-season holdout {partial_holdout['seasons']}: "
              f"MAE {partial_holdout['xgboost_MAE']:,.0f} | R² {partial_holdout['xgboost_R2']:.3f} "
              f"({partial_holdout['n']} matches)")

    # ---- Refit on all matches for the dashboard ---------------------------
    X_full = X_all[feature_cols_final]
    final_xgb = XGBRegressor(**best_params, random_state=RANDOM_STATE, verbosity=0).fit(X_full, y_all)
    final_lower = quantile_model(0.10).fit(X_full, y_all)
    final_upper = quantile_model(0.90).fit(X_full, y_all)
    final_scaler = StandardScaler()
    final_lr = LinearRegression().fit(final_scaler.fit_transform(X_full), y_all)

    joblib.dump(final_xgb, MODELS_DIR / "xgb_model.pkl")
    joblib.dump(final_lower, MODELS_DIR / "xgb_lower.pkl")
    joblib.dump(final_upper, MODELS_DIR / "xgb_upper.pkl")
    joblib.dump(final_lr, MODELS_DIR / "lr_baseline.pkl")
    joblib.dump(final_scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_cols_final, MODELS_DIR / "feature_cols.pkl")

    metrics = {
        "naive_team_average": {"MAE": round(float(naive_mae), 1), "R2": round(float(naive_r2), 4)},
        "linear_regression": {"MAE": round(float(lr_mae), 1), "R2": round(float(lr_r2), 4)},
        "xgboost": {"MAE": round(float(xgb_mae), 1), "R2": round(float(xgb_r2), 4),
                    "best_params": best_params},
        "quantile_interval_coverage_80pct": round(coverage, 4),
        "quantile_interval_mean_width": round(interval_width, 1),
        "test_season": test_season,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "partial_season_holdout": partial_holdout,
        "deployed_models_trained_on": f"all {len(X_full):,} matches (refit after evaluation)",
        "features_used": feature_cols_final,
        "leakage_excluded": [c for c in LEAKAGE_COLS if c in df.columns],
        "zero_variance_dropped": constant,
        "redundant_dummies_dropped": redundant,
        "vif_dropped": [d[0] for d in vif_dropped],
        "top3_shap_features": top3,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nModels and metrics saved to models/")
    print(json.dumps(metrics, indent=2))

    # ---- Per-team feature defaults for the dashboard forecast tool --------
    # Medians over each club's most recent season, so venue, capacity and form
    # reflect the club as it is now rather than its whole history.
    defaults = {}
    for team, grp in df.groupby("home_team"):
        recent = grp[grp["season"] == grp["season"].max()]
        defaults[str(team)] = {
            col: float(recent[col].median()) if recent[col].notna().any() else 0.0
            for col in feature_cols_final
        }
    with open(MODELS_DIR / "team_feature_defaults.json", "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=2, ensure_ascii=False)
    print(f"team_feature_defaults.json saved for {len(defaults)} teams.")

    print("\n[OK] Phase 5 complete -- models, metrics and figures saved.")


if __name__ == "__main__":
    main()
