"""
Database interaction module for the trading application.

This module handles all SQLite database operations, including:
- Initialization of the database and tables.
- Insertion of trade records.
- Fetching of trade records.
It uses a generic query execution function with proper error handling and
connection management. Logging is integrated for monitoring database activities.
"""
import sqlite3
from typing import List, Tuple, Any, Optional
import logging
import os
import sys # For stderr print in logger setup fallback

# Logger for database operations
db_logger = logging.getLogger(__name__)

# Basic configuration if not already set by another module (e.g., main app entry point)
if not db_logger.handlers:
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s")

    # Attempt to create a file handler for logs
    try:
        # Ensure logs directory exists or create it
        # For simplicity, placing log in the same directory as this script.
        # In a real app, this might be a dedicated 'logs' folder or user-specific directory.
        log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_app.log")

        # Use append mode for the log file
        fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        fh.setFormatter(log_formatter)
        db_logger.addHandler(fh)
    except (IOError, OSError) as e:
        print(f"Warning: Could not set up file handler for db_logger: {e}", file=sys.stderr)

    # Always add a stream handler (console output)
    sh = logging.StreamHandler(sys.stdout) # Use sys.stdout for standard output
    sh.setFormatter(log_formatter)
    db_logger.addHandler(sh)

    db_logger.setLevel(logging.INFO) # Default log level for this module


DB_NAME: str = "trades.db"
# Determine the path for the database file, placing it in the script's directory
try:
    DB_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_NAME)
except NameError: # __file__ might not be defined in some contexts (e.g. interactive interpreter)
    DB_PATH: str = os.path.join(os.getcwd(), DB_NAME)
db_logger.info(f"Database path configured to: {DB_PATH}")


def execute_db_query(query: str, params: Tuple[Any, ...] = (), fetch_one: bool = False, fetch_all: bool = False, commit: bool = False) -> Any:
    """
    Generic function to execute SQLite queries with context-managed connections and cursors.

    Args:
        query (str): The SQL query to execute.
        params (Tuple[Any, ...]): A tuple of parameters to substitute into the query. Defaults to an empty tuple.
        fetch_one (bool): If True, fetches one row. Defaults to False.
        fetch_all (bool): If True, fetches all rows. Defaults to False.
        commit (bool): If True, commits the transaction. Defaults to False.

    Returns:
        Any: The result of the query (e.g., a single row, all rows, lastrowid) or None.

    Raises:
        sqlite3.Error: Propagates SQLite errors if they occur.
        Exception: Propagates other unexpected errors.
    """
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)  # Using context manager for connection
        cursor: sqlite3.Cursor = conn.cursor()
        db_logger.debug(f"Executing DB query: '{query[:100]}...' with params: {params}")
        cursor.execute(query, params)

        result: Any = None
        if commit:
            conn.commit()
            result = cursor.lastrowid
            db_logger.info(f"Query committed: '{query[:50]}...'. Last row ID: {result if result is not None else 'N/A'}")
        elif fetch_one:
            result = cursor.fetchone()
        elif fetch_all:
            result = cursor.fetchall()

        return result

    except sqlite3.OperationalError as e:
        db_logger.error(f"SQLite OperationalError executing query '{query[:100]}...': {e}", exc_info=True)
        if "database is locked" in str(e).lower():
            db_logger.warning("Database is locked. Consider adjusting timeout or transaction management.")
        raise
    except sqlite3.DatabaseError as e:
        db_logger.error(f"SQLite DatabaseError executing query '{query[:100]}...': {e}", exc_info=True)
        raise
    except Exception as e: # Catch any other non-SQLite exception during DB operation
        db_logger.error(f"Unexpected error during DB operation for query '{query[:100]}...': {e}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()
            db_logger.debug("Database connection closed.")


def init_db() -> None:
    """
    Initializes the SQLite database and the 'trades' table if they do not already exist.
    Ensures the table schema is correctly defined.
    """
    db_logger.info(f"Initializing database schema at '{DB_PATH}' if it does not exist.")
    create_table_query: str = """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL,
            UNIQUE(timestamp, symbol, side, qty, price) -- Basic duplicate prevention
        )
    """
    try:
        # For DDL like CREATE TABLE, commit isn't strictly necessary as they are often auto-committed.
        # However, explicit commit ensures the operation is finalized.
        execute_db_query(create_table_query, commit=True)
        db_logger.info("Table 'trades' initialized/verified successfully.")
    except Exception as e:
        # If table creation fails, it's a critical issue for the application.
        db_logger.critical(f"Failed to initialize database table 'trades': {e}", exc_info=True)
        # Depending on application requirements, might re-raise or exit.


def insert_trade(timestamp: str, symbol: str, side: str, qty: float, price: Optional[float]) -> Optional[int]:
    """
    Inserts a trade record into the database.

    Args:
        timestamp (str): The timestamp of the trade.
        symbol (str): The trading symbol (e.g., 'BTCUSDT').
        side (str): The side of the trade ('BUY' or 'SELL').
        qty (float): The quantity of the trade.
        price (Optional[float]): The price of the trade. Can be None for market orders if price is not available at insert time.

    Returns:
        Optional[int]: The ID of the inserted row if successful, otherwise None.
    """
    insert_query: str = "INSERT INTO trades (timestamp, symbol, side, qty, price) VALUES (?, ?, ?, ?, ?)"
    # Ensure price is a float or None, not an empty string or other invalid type from UI.
    actual_price: Optional[float] = float(price) if price is not None else None
    params: Tuple[str, str, str, float, Optional[float]] = (timestamp, symbol, side, qty, actual_price)
    try:
        last_row_id: Optional[int] = execute_db_query(insert_query, params, commit=True)
        if last_row_id is not None:
            db_logger.info(f"Trade inserted: ID {last_row_id} - {timestamp}, {symbol}, {side}, Qty: {qty}, Price: {actual_price if actual_price is not None else 'N/A'}")
        else: # Should not happen if commit=True and no exception, but good for robustness.
            db_logger.warning(f"Trade insertion for {params} reported no lastrowid, though no exception was raised.")
        return last_row_id
    except sqlite3.IntegrityError as e:
        db_logger.warning(f"Failed to insert trade due to integrity error (likely duplicate): {e}. Trade data: {params}", exc_info=True)
        return None
    except Exception as e: # Catch other potential errors from execute_db_query
        db_logger.error(f"Failed to insert trade {params}. Error: {e}", exc_info=True)
        return None


def fetch_trades(limit: int = 100) -> List[Tuple[Any, ...]]:
    """
    Retrieves the most recent trade records from the database.

    Args:
        limit (int): The maximum number of trade records to retrieve. Defaults to 100.

    Returns:
        List[Tuple[Any, ...]]: A list of tuples, where each tuple represents a trade record.
                               Returns an empty list if no trades are found or an error occurs.
    """
    if not isinstance(limit, int) or limit <= 0:
        db_logger.warning(f"Invalid limit '{limit}' provided for fetch_trades. Defaulting to 100.")
        limit = 100

    select_query: str = "SELECT timestamp, symbol, side, qty, price FROM trades ORDER BY id DESC LIMIT ?"
    try:
        rows: Optional[List[Tuple[Any, ...]]] = execute_db_query(select_query, (limit,), fetch_all=True)
        if rows is not None:
            db_logger.info(f"Fetched {len(rows)} trade(s) with limit {limit}.")
            return rows
        else: # Should not happen if fetch_all=True and no exception.
            db_logger.warning(f"fetch_trades with limit {limit} returned None unexpectedly.")
            return []
    except Exception as e: # Catch errors propagated from execute_db_query
        db_logger.error(f"Failed to fetch trades with limit {limit}. Error: {e}", exc_info=True)
        return [] # Return an empty list to ensure function signature is met and caller can handle it.
