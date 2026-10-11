"""Simulated Feed Engine for ReadmitIQ.

Replays held-out TEST encounters in a seeded order with label hidden.
Inference goes through the exact same live scoring pipeline as all other endpoints.
The sealed ground-truth outcome is stored ONLY in feed_truth and is NEVER exposed
until released by the simulated Time Machine (>30 days).
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from api.database import get_db, log_audit
from api.service import get_service
from src.config import BASE_DIR, PROCESSED_DATA_DIR


class SimulatedFeedEngine:
    def __init__(self):
        self.is_running: bool = False
        self.rate_per_minute: float = 60.0
        self.seed: int = 42
        self.total_emitted: int = 0
        self.current_index: int = 0
        # Simulated clock starts at 2026-01-01 08:00:00 UTC
        self.simulated_time: datetime = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        self._task: Optional[asyncio.Task] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._shuffled_records: Optional[list] = None

    def _load_and_shuffle(self, seed: int):
        self.seed = seed
        test_path = PROCESSED_DATA_DIR / "test_rag.parquet"
        if not test_path.exists():
            test_path = BASE_DIR / "data" / "processed" / "test_rag.parquet"

        df = pd.read_parquet(test_path)
        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(len(df))
        self._shuffled_records = df.iloc[shuffled_indices].to_dict(orient="records")
        self.current_index = 0

    def start(self, rate_per_minute: float = 60.0, seed: int = 42) -> Dict[str, Any]:
        """Start the simulated background encounter feed."""
        if self.is_running:
            return self.get_status()

        self.rate_per_minute = max(1.0, min(rate_per_minute, 600.0))
        self._load_and_shuffle(seed)
        self.is_running = True
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._feed_loop())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            self._task = loop.create_task(self._feed_loop())
        return self.get_status()

    def stop(self) -> Dict[str, Any]:
        """Stop the simulated background encounter feed."""
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Return feed runtime telemetry and simulated clock."""
        return {
            "is_running": self.is_running,
            "rate_per_minute": self.rate_per_minute,
            "seed": self.seed,
            "total_emitted": self.total_emitted,
            "simulated_time": self.simulated_time.isoformat(),
            "feed_source": "simulated_feed (held-out test encounters)",
        }

    async def _feed_loop(self):
        """Asynchronous emitter loop."""
        svc = get_service()
        while self.is_running:
            try:
                interval = 60.0 / self.rate_per_minute
                await asyncio.sleep(interval)

                if not self.is_running:
                    break

                self.emit_single_encounter(svc)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[SimulatedFeed Error]: {e}")
                await asyncio.sleep(1.0)

    def emit_single_encounter(self, svc=None) -> Optional[Dict[str, Any]]:
        """Emit and score a single encounter, advancing simulated time."""
        if svc is None:
            svc = get_service()

        if self._shuffled_records is None:
            self._load_and_shuffle(self.seed)

        if self.current_index >= len(self._shuffled_records):
            self.current_index = 0  # Loop around

        raw_row = self._shuffled_records[self.current_index]
        self.current_index += 1

        # Advance simulated time (e.g. 15 to 45 minutes between hospital discharges)
        self.simulated_time += timedelta(minutes=30)
        now_iso = self.simulated_time.isoformat()
        eligible_date = (self.simulated_time + timedelta(days=30)).isoformat()

        # Extract sealed ground truth label
        true_outcome = int(raw_row.get("readmitted_30d", 0))
        enc_id = raw_row.get("encounter_id", self.current_index)
        ext_ref = f"SIM-{enc_id}"

        # Strip ground truth before live scoring
        scoring_payload = dict(raw_row)
        scoring_payload.pop("readmitted_30d", None)
        scoring_payload.pop("rag_struct_rate_k20", None)  # Ensure live retrieval is run

        # Run through the EXACT SAME scoring pipeline as API
        scored_list = svc.score_batch([scoring_payload], include_neighbor_details=True)
        if not scored_list:
            return None
        scored = scored_list[0]

        with get_db() as conn:
            # 1. Insert into discharges (label is hidden)
            cur = conn.execute(
                """INSERT INTO discharges (
                    received_at, source, external_ref, features_json,
                    raw_score, calibrated_prob, tier, model_version,
                    neighbor_summary_json, status, updated_at
                ) VALUES (?, 'simulated_feed', ?, ?, ?, ?, ?, ?, ?, 'new', ?)""",
                (
                    now_iso,
                    ext_ref,
                    json.dumps(scored["features"]),
                    scored["raw_score"],
                    scored["calibrated_prob"],
                    scored["tier"],
                    svc.frozen_config.get("model_name", "LightGBM + Structured RAG"),
                    json.dumps(scored["neighbor_summary"]),
                    now_iso,
                ),
            )
            discharge_id = cur.lastrowid

            # 2. Insert into sealed feed_truth (inaccessible to worklist or clinical queries)
            conn.execute(
                """INSERT INTO feed_truth (
                    discharge_id, external_ref, true_readmitted_30d,
                    simulated_discharge_date, eligible_after_date, released
                ) VALUES (?, ?, ?, ?, ?, 0)""",
                (
                    discharge_id,
                    ext_ref,
                    true_outcome,
                    now_iso,
                    eligible_date,
                ),
            )

            # Audit log
            log_audit(
                conn,
                actor="simulated_feed",
                event="encounter_emitted",
                payload={
                    "discharge_id": discharge_id,
                    "external_ref": ext_ref,
                    "simulated_time": now_iso,
                },
                ts=now_iso,
            )

        self.total_emitted += 1
        return {
            "discharge_id": discharge_id,
            "external_ref": ext_ref,
            "raw_score": scored["raw_score"],
            "calibrated_prob": scored["calibrated_prob"],
            "tier": scored["tier"],
            "simulated_time": now_iso,
        }

    def advance_time_machine(self, days: int = 30) -> Dict[str, Any]:
        """Advance simulated time and release outcomes for discharges older than 30 days."""
        if days < 1:
            days = 1

        self.simulated_time += timedelta(days=days)
        new_time_iso = self.simulated_time.isoformat()

        released_count = 0
        with get_db() as conn:
            # Find all unreleased truths whose 30-day window has expired
            cursor = conn.execute(
                """SELECT ft.id, ft.discharge_id, ft.true_readmitted_30d, ft.external_ref
                   FROM feed_truth ft
                   WHERE ft.released = 0 AND ft.eligible_after_date <= ?""",
                (new_time_iso,),
            )
            rows = cursor.fetchall()

            for row in rows:
                ft_id = row["id"]
                d_id = row["discharge_id"]
                outcome = row["true_readmitted_30d"]

                # Release outcome to discharges table
                conn.execute(
                    """UPDATE discharges
                       SET outcome_readmitted_30d = ?, outcome_at = ?, updated_at = ?
                       WHERE id = ?""",
                    (outcome, new_time_iso, new_time_iso, d_id),
                )

                # Mark sealed truth as released
                conn.execute(
                    """UPDATE feed_truth
                       SET released = 1, released_at = ?
                       WHERE id = ?""",
                    (new_time_iso, ft_id),
                )
                released_count += 1

            if released_count > 0:
                log_audit(
                    conn,
                    actor="time_machine",
                    event="outcomes_released",
                    payload={
                        "days_advanced": days,
                        "released_count": released_count,
                        "new_simulated_time": new_time_iso,
                    },
                    ts=new_time_iso,
                )

        return {
            "days_advanced": days,
            "new_simulated_time": new_time_iso,
            "outcomes_released": released_count,
            "message": f"Advanced simulated clock by {days} days. Released {released_count} verified 30-day outcomes.",
        }


_FEED_INSTANCE: Optional[SimulatedFeedEngine] = None


def get_feed_engine() -> SimulatedFeedEngine:
    global _FEED_INSTANCE
    if _FEED_INSTANCE is None:
        _FEED_INSTANCE = SimulatedFeedEngine()
    return _FEED_INSTANCE
