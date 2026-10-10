"""ReadmitIQ SQLite Persistence Layer.

Provides WAL-mode SQLite storage without heavy ORMs.
Tables:
- discharges: live & simulated discharge encounters, scores, tiers, status, outcomes
- actions: clinical workflow touchpoints (calls, medication reconciliations, appointments)
- audit_log: tamper-evident operational mutation logs
- feed_truth: sealed true outcomes for simulated replay feed (never exposed before time-machine)
- monitoring_snapshots: historical drift & calibration checkpoints

SECURITY / AUTH / PRIVACY NOTE:
- In production, actor authentication requires enterprise SSO (SAML/OIDC) and Role-Based
  Access Control (RBAC with roles like CareCoordinator, NurseNavigator, QualityDirector, SystemAdmin).
  In this prototype, actor is passed as a client-selected header/field ('Care Coordinator', 'Physician', etc.).
- PRIVACY: No Patient Health Information (PHI) such as patient names, SSNs, or addresses
  are ever accepted or persisted. Only numeric and categorical clinical encounter features are stored.
  Free-text clinical action notes MUST NOT contain PHI.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, Optional

from src.config import BASE_DIR

DEFAULT_DB_PATH = BASE_DIR / "data" / "readmitiq.db"


def get_db_path() -> Path:
    """Return configured SQLite database path from environment or default."""
    env_path = os.getenv("READMITIQ_DB_PATH")
    if env_path:
        p = Path(env_path)
    else:
        p = DEFAULT_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create a SQLite connection configured with WAL mode and foreign keys enabled."""
    target_path = db_path or get_db_path()
    conn = sqlite3.connect(str(target_path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for high concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def get_db(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding a SQLite connection and automatically committing transactions."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize schema tables and indexes."""
    with get_db(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS discharges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at TEXT NOT NULL,
                source TEXT NOT NULL CHECK(source IN ('manual', 'csv', 'api', 'simulated_feed')),
                external_ref TEXT,
                features_json TEXT NOT NULL,
                raw_score REAL NOT NULL,
                calibrated_prob REAL NOT NULL,
                tier TEXT NOT NULL CHECK(tier IN ('Critical', 'High', 'Medium', 'Low')),
                model_version TEXT NOT NULL,
                neighbor_summary_json TEXT,
                status TEXT NOT NULL DEFAULT 'new' CHECK(status IN ('new', 'assigned', 'contacted', 'followup_scheduled', 'closed')),
                assignee TEXT,
                updated_at TEXT NOT NULL,
                outcome_readmitted_30d INTEGER,
                outcome_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_discharges_raw_score ON discharges(raw_score DESC);
            CREATE INDEX IF NOT EXISTS idx_discharges_status ON discharges(status);
            CREATE INDEX IF NOT EXISTS idx_discharges_tier ON discharges(tier);
            CREATE INDEX IF NOT EXISTS idx_discharges_received_at ON discharges(received_at);
            CREATE INDEX IF NOT EXISTS idx_discharges_source ON discharges(source);

            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                discharge_id INTEGER NOT NULL REFERENCES discharges(id) ON DELETE CASCADE,
                type TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                actor TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_actions_discharge_id ON actions(discharge_id);

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                actor TEXT NOT NULL,
                event TEXT NOT NULL,
                payload_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_audit_log_ts ON audit_log(ts);

            -- Sealed ground-truth for simulated feed. NEVER exposed to clinicians or worklist.
            CREATE TABLE IF NOT EXISTS feed_truth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                discharge_id INTEGER NOT NULL UNIQUE REFERENCES discharges(id) ON DELETE CASCADE,
                external_ref TEXT,
                true_readmitted_30d INTEGER NOT NULL,
                simulated_discharge_date TEXT,
                eligible_after_date TEXT,
                released INTEGER NOT NULL DEFAULT 0,
                released_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_feed_truth_discharge_id ON feed_truth(discharge_id);
            CREATE INDEX IF NOT EXISTS idx_feed_truth_released ON feed_truth(released);

            CREATE TABLE IF NOT EXISTS monitoring_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                raw_score_psi REAL,
                tier_distribution_json TEXT,
                metrics_json TEXT,
                alerts_json TEXT
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                discharge_id INTEGER NOT NULL REFERENCES discharges(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL CHECK(priority IN ('Critical', 'High', 'Medium', 'Low')),
                trigger_driver TEXT,
                rationale TEXT NOT NULL,
                due_date TEXT,
                owner TEXT,
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'in_progress', 'completed', 'cancelled')),
                completed_at TEXT,
                completed_by TEXT,
                completion_note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_discharge_id ON tasks(discharge_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date);

            CREATE TABLE IF NOT EXISTS data_quality_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                source TEXT NOT NULL,
                event_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                payload_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_data_quality_events_ts ON data_quality_events(ts);
            CREATE INDEX IF NOT EXISTS idx_data_quality_events_type ON data_quality_events(event_type);
            """
        )


def log_audit(
    conn: sqlite3.Connection,
    actor: str,
    event: str,
    payload: Optional[Dict[str, Any]] = None,
    ts: Optional[str] = None,
) -> int:
    """Record an immutable event entry in audit_log."""
    now_ts = ts or datetime.now(timezone.utc).isoformat()
    payload_str = json.dumps(payload, default=str) if payload is not None else None
    cursor = conn.execute(
        "INSERT INTO audit_log (ts, actor, event, payload_json) VALUES (?, ?, ?, ?)",
        (now_ts, actor or "system", event, payload_str),
    )
    return cursor.lastrowid


def log_data_quality_event(
    conn: sqlite3.Connection,
    source: str,
    event_type: str,
    error_message: str,
    payload: Optional[Dict[str, Any]] = None,
    ts: Optional[str] = None,
) -> int:
    """Record a data-quality failure, validation error, or RAG availability event."""
    now_ts = ts or datetime.now(timezone.utc).isoformat()
    payload_str = json.dumps(payload, default=str) if payload is not None else None
    cursor = conn.execute(
        "INSERT INTO data_quality_events (ts, source, event_type, error_message, payload_json) VALUES (?, ?, ?, ?, ?)",
        (now_ts, source, event_type, error_message, payload_str),
    )
    return cursor.lastrowid
