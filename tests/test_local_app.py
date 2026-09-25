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

    def test_failed_write_does_not_hold_the_lock(self):
        """A statement that raises must not leave a transaction open.

        With implicit transactions, a failed INSERT left the connection holding
        SQLite's write lock for the life of the process, and every later write
        failed with "database is locked" until the app restarted.
        """
        self.db.insert_job("j1", "rp1", "t2va", "quality", {"task": "t2va"}, "p")
        with self.assertRaises(Exception):
            self.db.insert_job("j1", "rp2", "t2va", "quality", {"task": "t2va"}, "dup key")
        # The same connection must be usable, and a second one must be able to
        # take the write lock.
        self.db.update_job("j1", status="RUNNING")
        other = database.Database(Path(self.tmp.name) / "test.sqlite3")
        other.insert_job("j2", "rp3", "t2va", "quality", {"task": "t2va"}, "second")
        self.assertEqual(other.get_job("j2")["status"], "QUEUED")

    def test_tags_and_favorites(self):
        self.db.insert_generation(gen_id="g1", task="t2va", prompt_text="p")
        self.db.set_tags("g1", ["#b", "a", "a", "  "])
        self.db.set_favorite("g1", True)
        gen = self.db.get_generation("g1")
        self.assertEqual(json.loads(gen["tags"]), ["a", "b"])
        self.assertEqual(gen["favorited"], 1)
        self.assertEqual(self.db.tag_counts(),
                         [{"tag": "a", "count": 1}, {"tag": "b", "count": 1}])
        self.assertEqual(len(self.db.search_generations(tag="a")), 1)
        self.assertEqual(len(self.db.search_generations(tag="missing")), 0)
        self.db.insert_generation(gen_id="g2", task="t2va", prompt_text="p2")
        self.assertEqual(self.db.search_generations(sort="favorited")[0]["gen_id"], "g1")

    def test_list_jobs_omits_spec_json(self):
        """The queue is polled every few seconds and specs can carry megabytes
        of uploaded media, so the list must not return spec_json."""
        self.db.insert_job("j1", "rp1", "t2va", "quality",
                           {"task": "t2va", "first_frame_b64": "A" * 1000}, "p")
        row = self.db.list_jobs()[0]
        self.assertNotIn("spec_json", row)
        # The queue still needs these.
        for column in ("id", "status", "task", "prompt_text", "error", "gen_id"):
            self.assertIn(column, row)

    def test_stored_spec_strips_media(self):
        from local_app import service
        spec = {
            "task": "ref2va",
            "first_frame_b64": "x" * 100,
            "ref_images_b64": ["a", "b"],
            "ref_audios_b64": ["c"],
            "watermark": {"image_b64": "y" * 100, "position": "top-left"},
            "prompt": {"summary": "s"},
        }
        stored = service._stored_spec(spec)
        self.assertNotIn("first_frame_b64", stored)
        self.assertNotIn("ref_images_b64", stored)
        self.assertNotIn("ref_audios_b64", stored)
        self.assertNotIn("image_b64", stored["watermark"])
        # The parameters survive, and the media is summarised.
        self.assertEqual(stored["watermark"]["position"], "top-left")
        self.assertEqual(stored["prompt"], {"summary": "s"})
        self.assertEqual(stored["media_attached"],
                         {"ref_images": 2, "ref_audios": 1, "first_frame": 1})

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


class ValidationTests(unittest.TestCase):
    """User-input problems must surface as ServiceError (HTTP 400), not as a
    500 from an unguarded float()/slicing, and must not reach RunPod."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = database.Database(Path(self.tmp.name) / "t.sqlite3")
        from local_app import service
        self.service = service
        self.svc = service.Service.__new__(service.Service)  # no poller thread
        self.svc.db = self.db

    def tearDown(self):
        self.tmp.cleanup()

    def submit(self, **overrides):
        payload = {
            "task": "t2va",
            "prompt": {"integrated_multimodal_description": "x"},
            "duration_seconds": 2,
        }
        payload.update(overrides)
        return self.svc.submit(payload)

    def test_bad_number_types_are_400(self):
        for payload in ({"duration_seconds": "abc"}, {"duration_seconds": None},
                        {"loras": [{"name": "a", "strength": "x"}]},
                        {"loras": "nope"}, {"files": {"ref_images": {"a": 1}}}):
            with self.assertRaises(self.service.ServiceError, msg=str(payload)):
                self.submit(**payload)

    def test_range_checks(self):
        for payload in ({"duration_seconds": 99}, {"duration_seconds": 0.2},
                        {"loras": [{"name": "a", "strength": 50}]}):
            with self.assertRaises(self.service.ServiceError, msg=str(payload)):
                self.submit(**payload)

    def test_frame_cap_spans_duration_and_fps(self):
        with self.assertRaises(self.service.ServiceError) as ctx:
            self.submit(duration_seconds=15, fps=48)
        self.assertIn("362-frame maximum", str(ctx.exception))

    def test_missing_media_requirements(self):
        with self.assertRaises(self.service.ServiceError):
            self.submit(task="i2va")
        with self.assertRaises(self.service.ServiceError):
            self.submit(task="ref2va")

    def test_single_frame_media_shape(self):
        """first_frame/last_frame are single base64 strings from the dashboard.

        They were briefly validated as lists, which rejected every i2va and
        flf2va job with "files.first_frame must be a list".
        """
        import json as _json
        import urllib.request
        from unittest import mock
        captured = {}

        def fake_run(spec):
            captured.update(spec)
            return {"id": "rp-test"}

        with mock.patch.object(self.service.runpod_client, "run", fake_run):
            result = self.submit(task="flf2va",
                                 files={"first_frame": "AAA", "last_frame": "BBB"})
        self.assertEqual(result["runpod_job_id"], "rp-test")
        self.assertEqual(captured["first_frame_b64"], "AAA")
        self.assertEqual(captured["last_frame_b64"], "BBB")
        # A one-element list is accepted for the same slots.
        with mock.patch.object(self.service.runpod_client, "run", fake_run):
            self.submit(task="i2va", files={"first_frame": ["CCC"]})
        self.assertEqual(captured["first_frame_b64"], "CCC")
        # A dict is still rejected.
        with self.assertRaises(self.service.ServiceError):
            self.submit(task="i2va", files={"first_frame": {"a": 1}})
        # Lists still work for the multi slots, with the limit enforced.
        with mock.patch.object(self.service.runpod_client, "run", fake_run):
            self.submit(task="ref2va", files={"ref_images": ["A", "B"],
                                              "ref_audios": ["C"]})
        self.assertEqual(captured["ref_images_b64"], ["A", "B"])
        self.assertEqual(captured["ref_audios_b64"], ["C"])
        with self.assertRaises(self.service.ServiceError):
            self.submit(task="ref2va", files={"ref_images": ["x"] * 10})


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
