"""
Phase 4 — SQL Loader
Loads cleaned CSVs into an in-memory SQLite database, executes all SQL files,
and saves query results to data/processed/sql_results/.
"""

import re
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLEAN_PATH   = ROOT / "data" / "processed" / "mls_matches_clean.csv"
STADIUM_PATH = ROOT / "data" / "raw" / "stadium_market.csv"
SQL_DIR      = ROOT / "sql"
RESULTS_DIR  = ROOT / "data" / "processed" / "sql_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_into_sqlite(engine) -> None:
    """Load cleaned CSV data into SQLite tables."""
    matches_df = pd.read_csv(CLEAN_PATH, low_memory=False)
    matches_df.to_sql("matches", con=engine, if_exists="replace", index=False)
    print(f"  Loaded {len(matches_df):,} rows into 'matches' table")

    if STADIUM_PATH.exists():
        teams_df = pd.read_csv(STADIUM_PATH)
        teams_df.to_sql("teams", con=engine, if_exists="replace", index=False)
        print(f"  Loaded {len(teams_df):,} rows into 'teams' table")
    else:
        print("  [WARN] stadium_market.csv not found; 'teams' table not loaded")


def _extract_named_queries(sql_text: str) -> list[tuple[str, str]]:
    """
    Parse a SQL file into (name, query) pairs using '-- QUERY: name' markers.
    Falls back to a single unnamed query if no markers are found.
    """
    parts = re.split(r"--\s*QUERY:\s*(\w+)", sql_text)
    if len(parts) == 1:
        return [("query", sql_text.strip())]

    queries = []
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            queries.append((name, body))
    return queries


def run_sql_file(path: Path, engine) -> None:
    """Execute all named queries in a SQL file and save results as CSVs."""
    sql_text = path.read_text(encoding="utf-8")

    # Skip pure DDL files (CREATE TABLE, DROP TABLE) — just execute them
    if re.search(r"\bCREATE\s+TABLE\b", sql_text, re.IGNORECASE):
        with engine.connect() as conn:
            # Execute statement by statement
            for stmt in sql_text.split(";"):
                stmt = stmt.strip()
                # Skip DROPs: tables are already loaded from the DataFrames above
                if stmt and not stmt.startswith("--") and not stmt.upper().startswith("DROP"):
                    try:
                        conn.execute(text(stmt))
                    except Exception:
                        pass  # Table may already exist from DataFrame load
            conn.commit()
        print(f"  Executed DDL: {path.name}")
        return

    # For query files, parse named blocks and save each as a CSV
    queries = _extract_named_queries(sql_text)
    with engine.connect() as conn:
        for name, query_sql in queries:
            # Strip leading comment lines to find the first keyword
            non_comment_lines = [
                ln for ln in query_sql.splitlines()
                if ln.strip() and not ln.strip().startswith("--")
            ]
            first_word = non_comment_lines[0].strip().split()[0].upper() if non_comment_lines else ""
            if first_word != "SELECT":
                continue
            try:
                df = pd.read_sql(text(query_sql), con=conn)
                out_path = RESULTS_DIR / f"{name}.csv"
                df.to_csv(out_path, index=False)
                print(f"  [{name}] -> {len(df)} rows -> {out_path.name}")
            except Exception as exc:
                print(f"  [WARN] Query '{name}' failed: {exc}")


def main():
    print("=" * 60)
    print("MLS Attendance Forecasting — SQL Analysis")
    print("=" * 60)

    engine = create_engine("sqlite:///:memory:", echo=False)

    print("\nLoading data into SQLite...")
    load_into_sqlite(engine)

    sql_files = sorted(SQL_DIR.glob("*.sql"))
    for sql_file in sql_files:
        print(f"\nRunning {sql_file.name}...")
        run_sql_file(sql_file, engine)

    print("\n[OK] Phase 4 complete -- SQL analysis results saved to", RESULTS_DIR)


if __name__ == "__main__":
    main()
