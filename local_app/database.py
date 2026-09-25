"""SQLite storage for Nocturne Video jobs and completed generations."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    runpod_job_id TEXT,
    status TEXT NOT NULL DEFAULT 'QUEUED',
    task TEXT NOT NULL,
    quality TEXT,
    spec_json TEXT NOT NULL,
    prompt_text TEXT DEFAULT '',
    error TEXT,
    gen_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

CREATE TABLE IF NOT EXISTS generations (
    gen_id TEXT PRIMARY KEY,
    job_id TEXT,
    task TEXT NOT NULL,
    quality TEXT,
    checkpoint TEXT,
    prompt_text TEXT DEFAULT '',
    meta_json TEXT DEFAULT '{}',
    video_path TEXT,
    s3_uri TEXT,
    sha256 TEXT,
    bytes INTEGER,
    width INTEGER,
    height INTEGER,
    frames INTEGER,
    duration REAL,
    seed TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gen_created ON generations(created_at);
CREATE INDEX IF NOT EXISTS idx_gen_task ON generations(task);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- jobs ---------------------------------------------------------------

    def insert_job(self, job_id: str, runpod_job_id: str, task: str, quality: str,
                   spec: dict, prompt_text: str) -> None:
        now = _now()
        self.conn.execute(
            "INSERT INTO jobs (id, runpod_job_id, status, task, quality, spec_json, prompt_text, created_at, updated_at) "
            "VALUES (?, ?, 'QUEUED', ?, ?, ?, ?, ?, ?)",
            (job_id, runpod_job_id, task, quality, json.dumps(spec), prompt_text, now, now))
        self.conn.commit()

    def update_job(self, job_id: str, **fields) -> None:
        if not fields:
            return
        fields.setdefault("updated_at", _now())
        cols = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [job_id]
        self.conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", values)
        self.conn.commit()

    def get_job(self, job_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def active_jobs(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE status IN ('QUEUED', 'RUNNING') ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def list_jobs(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # -- generations ----------------------------------------------------------

    def insert_generation(self, **fields) -> str:
        fields.setdefault("created_at", _now())
        cols = ", ".join(fields.keys())
        marks = ", ".join("?" for _ in fields)
        self.conn.execute(f"INSERT INTO generations ({cols}) VALUES ({marks})",
                          list(fields.values()))
        self.conn.commit()
        return fields["gen_id"]

    def update_generation(self, gen_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(f"UPDATE generations SET {cols} WHERE gen_id = ?",
                          list(fields.values()) + [gen_id])
        self.conn.commit()

    def get_generation(self, gen_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM generations WHERE gen_id = ?", (gen_id,)).fetchone()
        return dict(row) if row else None

    def search_generations(self, q: str = "", task: str = "", limit: int = 60,
                           offset: int = 0) -> list[dict]:
        where, args = [], []
        if q:
            where.append("(prompt_text LIKE ? OR gen_id LIKE ?)")
            args += [f"%{q}%", f"%{q}%"]
        if task:
            where.append("task = ?")
            args.append(task)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = self.conn.execute(
            f"SELECT * FROM generations {clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            args + [limit, offset]).fetchall()
        return [dict(r) for r in rows]

    def generations_missing_files(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM generations WHERE s3_uri IS NOT NULL AND video_path IS NULL").fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) c FROM generations").fetchone()["c"]
        total_bytes = self.conn.execute(
            "SELECT COALESCE(SUM(bytes), 0) s FROM generations").fetchone()["s"]
        jobs_total = self.conn.execute("SELECT COUNT(*) c FROM jobs").fetchone()["c"]
        return {"generations": total, "total_bytes": total_bytes, "jobs": jobs_total}
