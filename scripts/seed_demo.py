"""Demo Seeding Script for ReadmitIQ.

Populates an empty database through the exact same live scoring code paths:
- Ingests 300 simulated encounters via SimulatedFeedEngine.
- Advances Time Machine by 20 days so that ~150 encounters have passed their
  30-day window and have verified outcomes released.
- Creates initial care actions and assignments for clinical realism.
- All seeded records are labeled 'simulated_feed'.
"""

import os
from pathlib import Path
import sys

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.database import get_db, init_db, log_audit
from api.feed import get_feed_engine
from api.service import get_service


def seed_demo_data(target_total: int = 300, release_target: int = 150):
    """Seed demo database through the official live scoring and time-machine pipeline."""
    init_db()
    svc = get_service()
    engine = get_feed_engine()

    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM discharges").fetchone()[0]
        if count >= target_total:
            print(f"[SEED] Database already contains {count} discharges. Skipping seeding.")
            return

    print(f"[SEED] Ingesting {target_total} simulated discharges through live scoring...")
    engine._load_and_shuffle(seed=42)

    # 1. Ingest encounters
    emitted_ids = []
    for i in range(target_total):
        res = engine.emit_single_encounter(svc)
        if res:
            emitted_ids.append(res["discharge_id"])
        if (i + 1) % 50 == 0:
            print(f"  .. emitted and scored {i + 1}/{target_total} encounters")

    print(f"[SEED] Successfully ingested {len(emitted_ids)} discharges.")

    # 2. Add realistic clinical touchpoints and care actions to Critical and High tier patients
    with get_db() as conn:
        high_risk_rows = conn.execute(
            """SELECT id, tier FROM discharges
               WHERE tier IN ('Critical', 'High')
               ORDER BY id ASC LIMIT 40"""
        ).fetchall()

        coordinators = ["Nurse Martinez, RN", "Alex Chen, LCSW", "Dr. Vance, MD", "Jordan Lee, PharmD"]
        for idx, row in enumerate(high_risk_rows):
            d_id = row["id"]
            assignee = coordinators[idx % len(coordinators)]
            new_status = "contacted" if idx % 2 == 0 else "followup_scheduled"

            conn.execute(
                "UPDATE discharges SET status = ?, assignee = ? WHERE id = ?",
                (new_status, assignee, d_id),
            )

            conn.execute(
                """INSERT INTO actions (discharge_id, type, note, created_at, actor)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    d_id,
                    "phone_call" if idx % 2 == 0 else "med_reconciliation",
                    "Post-discharge transition check completed. Patient educated on insulin dosage signs.",
                    engine.simulated_time.isoformat(),
                    assignee,
                ),
            )

    # 3. Advance Time Machine by 32 days so ~150 encounters have passed their 30-day window
    print(f"[SEED] Advancing Time Machine to release verified outcomes for ~{release_target} encounters...")
    tm_result = engine.advance_time_machine(days=32)
    print(f"[SEED] {tm_result['message']}")

    with get_db() as conn:
        counts = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome_readmitted_30d IS NOT NULL THEN 1 ELSE 0 END) as with_outcome,
                SUM(CASE WHEN outcome_readmitted_30d = 1 THEN 1 ELSE 0 END) as positives
               FROM discharges"""
        ).fetchone()

    print(
        f"[SEED COMPLETE] Total: {counts['total']} discharges | "
        f"Outcomes released: {counts['with_outcome']} | "
        f"Positive readmissions: {counts['positives']}"
    )


def seed_if_empty():
    """Auto-seed only if database currently contains zero discharges (for ephemeral cloud hosting)."""
    try:
        init_db()
        with get_db() as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM discharges").fetchone()[0]
            if cnt == 0:
                print("[OPS] Ephemeral disk detected empty database. Running startup demo seed...")
                seed_demo_data(target_total=300, release_target=150)
    except Exception as e:
        print(f"[OPS WARNING] Auto-seed error: {e}")


if __name__ == "__main__":
    seed_demo_data()
