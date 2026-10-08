"""Transactional sharded physical-teacher corpus compatible with the reader."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import zipfile
from pathlib import Path

import numpy as np


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(scene):
    pairs = np.round(np.concatenate((scene["starts"], scene["goals"]), axis=1).astype(np.float64), 8)
    pairs = pairs[np.lexsort(tuple(pairs[:, k] for k in range(3, -1, -1)))]
    return sha256(pairs.tobytes())


def geometry_hash(scene, center_bound=0.95):
    return sha256(json.dumps(dict(start_goal=canonical_hash(scene),
                                  radii=sorted(scene["radii"]),
                                  center_bound=center_bound), sort_keys=True).encode())


def noise_seeds(scene, variants=8):
    key = canonical_hash(scene)
    population = len(scene["starts"])
    domain = "n28-teacher-25k-v1" if population == 28 else f"n{population}-teacher-v1"
    return [int.from_bytes(hashlib.sha256(f"{domain}:{key}:{i}".encode()).digest()[:8], "little")
            % (2**63 - 1) for i in range(variants)]


class TeacherCorpusWriter:
    """Atomic ZIP shards and SQLite rows with restart-safe scene identities."""

    def __init__(self, directory, shard_size=128):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "shards").mkdir(exist_ok=True)
        self.shard_size = int(shard_size)
        self.db = sqlite3.connect(str(self.directory / "corpus_manifest.sqlite3"))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS scenes (
              attempt INTEGER PRIMARY KEY, ordinal INTEGER UNIQUE, scene_id TEXT UNIQUE,
              geometry_hash TEXT UNIQUE, start_goal_hash TEXT UNIQUE, status TEXT,
              row_json TEXT, shard TEXT, member TEXT, qualified_rank INTEGER UNIQUE);
            CREATE TABLE IF NOT EXISTS shards (
              name TEXT PRIMARY KEY, sha256 TEXT, count INTEGER, bytes INTEGER);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS seeds (seed INTEGER PRIMARY KEY, scene_id TEXT);
        """)
        self._recover()

    def _recover(self):
        known = dict(self.db.execute("SELECT name,sha256 FROM shards"))
        for path in sorted((self.directory / "shards").glob("teachers_*.zip")):
            if path.name not in known:
                self._commit_shard(path)
            elif sha256(path.read_bytes()) != known[path.name]:
                raise RuntimeError("Corrupt corpus shard: " + path.name)
        for name in known:
            if not (self.directory / "shards" / name).exists():
                raise RuntimeError("Missing corpus shard: " + name)
        for path in (self.directory / "shards").glob("*.partial"):
            path.unlink()

    def has_scene(self, scene_id):
        return self.db.execute("SELECT 1 FROM scenes WHERE scene_id=?", (str(scene_id),)).fetchone() is not None

    def append(self, scene, trajectory, coefficients, metadata, guide=None):
        scene_id = str(scene["configuration_id"])
        if self.has_scene(scene_id):
            raise ValueError("Scene already present: " + scene_id)
        count, qualified_count = self.db.execute(
            "SELECT count(*),count(qualified_rank) FROM scenes").fetchone()
        qualified = metadata["status"].startswith("qualified")
        rank = qualified_count if qualified else None
        payload = None
        if qualified:
            stream = io.BytesIO()
            np.savez_compressed(stream, physical=np.asarray(trajectory, dtype=np.float32),
                                coefficients_q=np.asarray(coefficients, dtype=np.float32),
                                requested_clearance_m=0.120)
            payload = stream.getvalue()
        row = dict(attempt=count + 1, ordinal=count, scene_id=scene_id, scene=scene,
                   geometry_hash=geometry_hash(scene), start_goal_hash=canonical_hash(scene),
                   status=metadata["status"], teacher_clearance=metadata.get("actual_clearance_m"),
                   teacher_sampled_clearance=metadata.get("sampled_clearance_m"),
                   teacher_sha256=sha256(payload) if payload else None,
                   teacher_metadata=metadata,
                   noise_seeds=noise_seeds(scene) if qualified else [],
                   qualified_rank=rank)
        if payload:
            staging = self.directory / "staging"
            staging.mkdir(exist_ok=True)
            (staging / f"{scene_id}.npz").write_bytes(payload)
            if guide is not None:
                (staging / f"{scene_id}.guide.npz").write_bytes(Path(guide).read_bytes())
        with self.db:
            self.db.execute("INSERT INTO scenes VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (row["attempt"], row["ordinal"], scene_id, row["geometry_hash"],
                             row["start_goal_hash"], row["status"], json.dumps(row, sort_keys=True),
                             None, None, rank))
            if qualified:
                self.db.executemany("INSERT INTO seeds VALUES (?,?)",
                                    [(seed, scene_id) for seed in row["noise_seeds"]])
        self.flush()
        return row

    def flush(self, force=False):
        rows = [json.loads(item[0]) for item in self.db.execute(
            "SELECT row_json FROM scenes WHERE qualified_rank IS NOT NULL AND shard IS NULL "
            "ORDER BY qualified_rank LIMIT ?", (self.shard_size,))]
        if not rows or (len(rows) < self.shard_size and not force):
            return False
        number = self.db.execute("SELECT count(*) FROM shards").fetchone()[0]
        name = f"teachers_{number:05d}.zip"
        path = self.directory / "shards" / name
        temporary = path.with_suffix(".zip.partial")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for row in rows:
                scene_id = row["scene_id"]
                row["teacher_shard"] = name
                row["teacher_member"] = f"teachers/{scene_id}.npz"
                teacher = self.directory / "staging" / f"{scene_id}.npz"
                if sha256(teacher.read_bytes()) != row["teacher_sha256"]:
                    raise RuntimeError("Staging teacher checksum mismatch")
                archive.write(teacher, row["teacher_member"])
                guide = self.directory / "staging" / f"{scene_id}.guide.npz"
                if guide.exists():
                    row["guide_member"] = f"guides/{scene_id}.npz"
                    archive.write(guide, row["guide_member"])
            archive.writestr("manifest.json", json.dumps(rows, sort_keys=True))
        os.replace(temporary, path)
        self._commit_shard(path)
        for row in rows:
            (self.directory / "staging" / f"{row['scene_id']}.npz").unlink(missing_ok=True)
            (self.directory / "staging" / f"{row['scene_id']}.guide.npz").unlink(missing_ok=True)
        self.export_manifests()
        return True

    def _commit_shard(self, path):
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Corrupt ZIP shard")
            rows = json.loads(archive.read("manifest.json"))
            for row in rows:
                if sha256(archive.read(row["teacher_member"])) != row["teacher_sha256"]:
                    raise RuntimeError("Teacher checksum mismatch")
        with self.db:
            self.db.execute("INSERT INTO shards VALUES (?,?,?,?)",
                            (path.name, sha256(path.read_bytes()), len(rows), path.stat().st_size))
            for row in rows:
                result = self.db.execute("UPDATE scenes SET shard=?,member=?,row_json=? "
                                         "WHERE attempt=? AND shard IS NULL",
                                         (path.name, row["teacher_member"],
                                          json.dumps(row, sort_keys=True), row["attempt"]))
                if result.rowcount != 1:
                    raise RuntimeError("Unmatched corpus scene row")

    def export_manifests(self):
        for name, predicate in (("corpus_manifest.jsonl", "1"),
                                ("failures.jsonl", "qualified_rank IS NULL")):
            path = self.directory / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w") as stream:
                for row in self.db.execute("SELECT row_json FROM scenes WHERE " + predicate + " ORDER BY attempt"):
                    stream.write(row[0] + "\n")
            os.replace(temporary, path)
        shards = [dict(name=r[0], sha256=r[1], count=r[2], bytes=r[3])
                  for r in self.db.execute("SELECT * FROM shards ORDER BY name")]
        (self.directory / "shard_index.json").write_text(
            json.dumps(dict(shard_size=self.shard_size, shards=shards), indent=2) + "\n")

    def close(self):
        self.flush(force=True)
        self.export_manifests()
        self.db.close()
