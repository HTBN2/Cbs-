"""
Token database — SQLite, no external dependencies.
Tables:
  tokens : one row per client token
  usage  : daily usage counters per token
"""

import sqlite3
import os
from datetime import datetime, date

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokens.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS tokens (
            token        TEXT PRIMARY KEY,
            client_name  TEXT NOT NULL,
            plan_name    TEXT NOT NULL,
            daily_limit  INTEGER NOT NULL,
            total_limit  INTEGER NOT NULL,
            created_at   TEXT NOT NULL,
            expires_at   TEXT NOT NULL,
            is_active    INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS usage (
            token        TEXT NOT NULL,
            use_date     TEXT NOT NULL,
            count        INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (token, use_date)
        );
        """)


# ── Token checks ──────────────────────────────────────────────────────────────

def check_token(token: str) -> tuple:
    """
    Returns (allowed: bool, reason: str)
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM tokens WHERE token = ?", (token,)
        ).fetchone()

    if not row:
        return False, "Invalid token"
    if not row["is_active"]:
        return False, "Token disabled"

    today = date.today().isoformat()
    expires = row["expires_at"][:10]
    if today > expires:
        return False, "Token expired"

    with get_conn() as conn:
        usage_row = conn.execute(
            "SELECT count FROM usage WHERE token = ? AND use_date = ?",
            (token, today)
        ).fetchone()
        daily_used = usage_row["count"] if usage_row else 0

        total_used = conn.execute(
            "SELECT COALESCE(SUM(count), 0) as total FROM usage WHERE token = ?",
            (token,)
        ).fetchone()["total"]

    if daily_used >= row["daily_limit"]:
        return False, f"Daily limit reached ({row['daily_limit']})"
    if total_used >= row["total_limit"]:
        return False, f"Total limit reached ({row['total_limit']})"

    return True, "ok"


def increment_usage(token: str):
    today = date.today().isoformat()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO usage (token, use_date, count) VALUES (?, ?, 1)
            ON CONFLICT(token, use_date) DO UPDATE SET count = count + 1
        """, (token, today))


def get_usage(token: str) -> dict:
    today = date.today().isoformat()
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM tokens WHERE token = ?", (token,)).fetchone()
        if not row:
            return {}
        daily_row = conn.execute(
            "SELECT count FROM usage WHERE token = ? AND use_date = ?",
            (token, today)
        ).fetchone()
        total_row = conn.execute(
            "SELECT COALESCE(SUM(count), 0) as total FROM usage WHERE token = ?",
            (token,)
        ).fetchone()

    return {
        "token":        token,
        "client":       row["client_name"],
        "plan":         row["plan_name"],
        "daily_used":   daily_row["count"] if daily_row else 0,
        "daily_limit":  row["daily_limit"],
        "total_used":   total_row["total"],
        "total_limit":  row["total_limit"],
        "expires_at":   row["expires_at"][:10],
        "is_active":    bool(row["is_active"]),
    }


# ── Admin helpers ─────────────────────────────────────────────────────────────

def create_token(client_name: str, plan_name: str, daily_limit: int,
                 total_limit: int, validity_days: int) -> str:
    import secrets
    token = secrets.token_hex(16)
    now = datetime.utcnow()
    from datetime import timedelta
    expires = (now + timedelta(days=validity_days)).isoformat()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO tokens (token, client_name, plan_name, daily_limit, total_limit, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (token, client_name, plan_name, daily_limit, total_limit, now.isoformat(), expires))
    return token


def list_tokens() -> list:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM tokens ORDER BY created_at DESC").fetchall()]


def disable_token(token: str):
    with get_conn() as conn:
        conn.execute("UPDATE tokens SET is_active = 0 WHERE token = ?", (token,))


def enable_token(token: str):
    with get_conn() as conn:
        conn.execute("UPDATE tokens SET is_active = 1 WHERE token = ?", (token,))


def delete_token(token: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
        conn.execute("DELETE FROM usage WHERE token = ?", (token,))


init_db()
