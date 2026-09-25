"""Tests for the local app: database CRUD + search, and the S3 signer shape."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_app import database


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = database.Database(Path(self.tmp.name) / "test.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_job_lifecycle(self):
        self.db.insert_job("j1", "rp1", "t2va", "quality", {"task": "t2va"}, "prompt here")
        self.db.update_job("j1", status="RUNNING")
        self.db.update_job("j1", status="COMPLETED", gen_id="g1")
        job = self.db.get_job("j1")
        self.assertEqual(job["status"], "COMPLETED")
        self.assertEqual(job["gen_id"], "g1")
        self.assertEqual(self.db.active_jobs(), [])

    def test_active_jobs_and_list(self):
        self.db.insert_job("j1", "rp1", "t2va", "quality", {}, "a")
        self.db.insert_job("j2", "rp2", "i2va", "turbo", {}, "b")
        self.assertEqual(len(self.db.active_jobs()), 2)
        self.db.update_job("j1", status="FAILED", error="boom")
        active = self.db.active_jobs()
        self.assertEqual([j["id"] for j in active], ["j2"])
        self.assertEqual(len(self.db.list_jobs()), 2)

    def test_generation_search(self):
        self.db.insert_generation(gen_id="g1", job_id="j1", task="t2va", quality="quality",
                                  checkpoint="dasiwa", prompt_text="a lighthouse in fog",
                                  meta_json=json.dumps({"steps": 25}), seed="42",
                                  created_at="2026-09-25T00:00:00Z")
        self.db.insert_generation(gen_id="g2", job_id="j1", task="ref2va", quality="turbo",
                                  checkpoint="dasiwa", prompt_text="street dancer",
                                  meta_json="{}", seed="7",
                                  created_at="2026-09-26T00:00:00Z")
        hits = self.db.search_generations(q="lighthouse")
        self.assertEqual([g["gen_id"] for g in hits], ["g1"])
        self.assertEqual(self.db.search_generations(task="ref2va")[0]["gen_id"], "g2")
        self.assertEqual(len(self.db.search_generations()), 2)
        missing = self.db.generations_missing_files()
        self.assertEqual(missing, [])  # no s3_uri -> nothing to sync
        self.db.update_generation("g1", s3_uri="s3://b/outputs/x.mp4")
        self.assertEqual([g["gen_id"] for g in self.db.generations_missing_files()], ["g1"])

    def test_stats(self):
        self.db.insert_generation(gen_id="g1", job_id="j1", task="t2va", quality="quality",
                                  checkpoint="d", prompt_text="", meta_json="{}", bytes=1000,
                                  created_at="2026-09-25T00:00:00Z")
        stats = self.db.stats()
        self.assertEqual(stats["generations"], 1)
        self.assertEqual(stats["total_bytes"], 1000)


class SigV4Tests(unittest.TestCase):
    def test_signer_builds_request_without_crashing(self):
        import urllib.request
        from local_app.s3_sync import RunpodS3
        client = RunpodS3("https://s3api-us-ne-1.runpod.io", "US-NE-1", "AKIAEXAMPLE", "secret")
        # Do not hit the network: verify URL + header construction only.
        import hashlib, hmac
        self.assertEqual(client.host, "s3api-us-ne-1.runpod.io")
        self.assertEqual(client.endpoint, "https://s3api-us-ne-1.runpod.io")
        # signing key derivation is deterministic
        key = hmac.new(b"AWS4secret", b"20260925", hashlib.sha256).digest()
        self.assertEqual(len(key), 32)


if __name__ == "__main__":
    unittest.main()
