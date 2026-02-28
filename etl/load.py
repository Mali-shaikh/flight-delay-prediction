"""
ETL Step 3 — Load
Saves processed flight data into a SQLite database and CSV.
Uses Python's built-in sqlite3 (no SQLAlchemy required).
"""

import sqlite3
import pandas as pd
from pathlib import Path

try:
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

PROCESSED_FILE = Path(__file__).parent.parent / "data" / "processed" / "flights_processed.csv"
DATABASE_DIR = Path(__file__).parent.parent / "database"
DB_PATH = DATABASE_DIR / "flight_delay.db"
TABLE_NAME = "flights"


def load(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    """
    Load processed DataFrame into SQLite database.

    Args:
        df: Processed DataFrame from transform step
        db_path: Path to SQLite database file
    """
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Loading to DB: {db_path} ===")

    if SQLALCHEMY_AVAILABLE:
        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)
        engine.dispose()
    else:
        # Fallback: use Python's built-in sqlite3
        conn = sqlite3.connect(str(db_path))
        df.to_sql(TABLE_NAME, con=conn, if_exists="replace", index=False)
        conn.close()

    print(f"Loaded {len(df):,} rows into table '{TABLE_NAME}'.")

    # Verify with sqlite3
    conn = sqlite3.connect(str(db_path))
    count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    conn.close()
    print(f"Verified: {count:,} rows in database.")


def read_from_db(db_path: Path = DB_PATH, limit: int = None) -> pd.DataFrame:
    """
    Read flight data from SQLite database.

    Args:
        db_path: Path to SQLite database
        limit: Optional row limit

    Returns:
        DataFrame from database
    """
    query = f"SELECT * FROM {TABLE_NAME}"
    if limit:
        query += f" LIMIT {limit}"
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(query, con=conn)
    conn.close()
    return df


if __name__ == "__main__":
    from etl.extract import extract
    from etl.transform import transform

    print("=== ETL Step 3: Load ===")
    raw = extract(use_sample=True)
    processed = transform(raw)
    load(processed)

    # Test reading back
    df_from_db = read_from_db(limit=5)
    print(f"\nSample from DB:\n{df_from_db}")

