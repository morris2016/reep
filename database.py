import sqlite3
from typing import List, Tuple

DB_PATH = "trades.db"


def init_db() -> None:
    """Initialize the SQLite database and tables if they do not exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL
        )
        """
    )
    conn.commit()
    conn.close()


def insert_trade(timestamp: str, symbol: str, side: str, qty: float, price: float) -> None:
    """Insert a trade record into the database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO trades (timestamp, symbol, side, qty, price) VALUES (?, ?, ?, ?, ?)",
        (timestamp, symbol, side, qty, price),
    )
    conn.commit()
    conn.close()


def fetch_trades(limit: int = 100) -> List[Tuple]:
    """Retrieve the most recent trade records."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT timestamp, symbol, side, qty, price FROM trades ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows
