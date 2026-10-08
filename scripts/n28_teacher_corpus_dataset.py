"""Lazy scene/teacher/noise access for the sharded physical-teacher corpus.

One dataset item is one scene and one of its eight frozen rollout seeds. The
physical teacher is shared by all eight items.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TeacherCorpusDataset(Dataset):
    """Snapshot of completed teachers, with bounded per-worker shard handles.

    Index ``8 * qualified_rank + variant`` selects a rollout condition. SQLite
    supplies the indexed manifest row; teacher arrays are read on demand. Open
    a new dataset after generation appends more completed shards to refresh
    the snapshot. CPU tensors support DataLoader pinning without device I/O
    inside workers. ``batch_size=None`` preserves the current trainer shape.
    """

    randomizations = 8

    def __init__(self, directory, max_open_shards=4):
        self.directory = Path(directory).resolve()
        self.database = self.directory / "corpus_manifest.sqlite3"
        self.max_open_shards = int(max_open_shards)
        if self.max_open_shards < 1:
            raise ValueError("max_open_shards must be positive")
        self._pid = None
        self._connection = None
        self._shards = OrderedDict()
        self._verified = {}
        row = (
            self._connect()
            .execute(
                "SELECT COUNT(*), MIN(qualified_rank), MAX(qualified_rank) FROM scenes "
                "WHERE status LIKE 'qualified%' AND shard IS NOT NULL"
            )
            .fetchone()
        )
        self.qualified_count = int(row[0])
        if self.qualified_count and (row[1], row[2]) != (0, self.qualified_count - 1):
            raise RuntimeError("Completed qualified ranks must be contiguous from zero")

    def _connect(self):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
            self._connection = sqlite3.connect(
                self.database.as_uri() + "?mode=ro", uri=True
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def close(self):
        for archive in self._shards.values():
            archive.close()
        self._shards.clear()
        self._verified.clear()
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_connection=None, _pid=None, _shards=OrderedDict(), _verified={})
        return state

    def __len__(self):
        return self.qualified_count * self.randomizations

    def _archive(self, name):
        shard_directory = self.directory / "shards"
        path = (shard_directory / name).resolve()
        if shard_directory not in path.parents:
            raise RuntimeError("Shard path leaves corpus shard directory")
        metadata = (
            self._connect()
            .execute("SELECT sha256, bytes FROM shards WHERE name = ?", (name,))
            .fetchone()
        )
        if metadata is None:
            raise RuntimeError("Shard missing from completed index: " + name)
        stat = path.stat()
        fingerprint = (metadata["sha256"], stat.st_size, stat.st_mtime_ns)
        if self._verified.get(name) != fingerprint:
            if stat.st_size != metadata["bytes"] or _sha256(path) != metadata["sha256"]:
                raise RuntimeError("Shard checksum/size mismatch: " + name)
            self._verified[name] = fingerprint
            if name in self._shards:
                self._shards.pop(name).close()
        if name not in self._shards:
            self._shards[name] = zipfile.ZipFile(path, "r")
        self._shards.move_to_end(name)
        while len(self._shards) > self.max_open_shards:
            _, archive = self._shards.popitem(last=False)
            archive.close()
        return self._shards[name]

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        rank, variant = divmod(index, self.randomizations)
        row = (
            self._connect()
            .execute(
                "SELECT * FROM scenes WHERE qualified_rank = ? "
                "AND status LIKE 'qualified%' AND shard IS NOT NULL",
                (rank,),
            )
            .fetchone()
        )
        if row is None:
            raise RuntimeError("Qualified manifest row disappeared: " + str(rank))
        metadata = json.loads(row["row_json"])
        seeds = metadata["noise_seeds"]
        if len(seeds) != self.randomizations or len(set(seeds)) != self.randomizations:
            raise RuntimeError("Expected eight distinct frozen noise seeds")
        payload = self._archive(row["shard"]).read(row["member"])
        if hashlib.sha256(payload).hexdigest() != metadata["teacher_sha256"]:
            raise RuntimeError("Teacher checksum mismatch: " + row["scene_id"])
        with np.load(io.BytesIO(payload), allow_pickle=False) as artifact:
            physical = artifact["physical"].copy()
        scene = dict(metadata["scene"])
        population = len(scene["starts"])
        if (
            physical.ndim != 4
            or physical.shape[0] != 1
            or physical.shape[2:] != (population, 4)
            or len(scene["goals"]) != population
            or physical.dtype != np.float32
        ):
            raise RuntimeError(
                "Teacher shape/dtype disagrees with scene: " + row["scene_id"]
            )
        seed = int(seeds[variant])
        # Preserve the stored scene/noise identity bound by input_sha256.
        # The canonical runtime reads the separately selected rollout seed.
        scene.update(rollout_seed=seed, _rollout_index=0)
        return dict(
            scene=scene,
            teacher=torch.from_numpy(physical),
            noise_seed=seed,
            noise_variant=variant,
            qualified_rank=rank,
            scene_id=row["scene_id"],
            geometry_hash=row["geometry_hash"],
            start_goal_hash=row["start_goal_hash"],
            teacher_sha256=metadata["teacher_sha256"],
        )
