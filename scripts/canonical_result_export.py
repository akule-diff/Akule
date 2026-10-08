"""Export an existing planner outcome through the frozen task/result contract."""
from __future__ import annotations

import hashlib
import json
import subprocess

import canonical_n28_runtime as c
import numpy as np


def export(scene, row, engine, directory):
    method = row["method"]
    normalizer = dict(
        minimum=engine.codec.lo.cpu().tolist(),
        maximum=engine.codec.hi.cpu().tolist(),
        conversion="unclipped affine",
    )
    normalizer["sha256"] = hashlib.sha256(
        json.dumps(normalizer, sort_keys=True).encode()
    ).hexdigest()
    mpd = getattr(engine, "base_checkpoint_path", c.CHECKPOINTS / "MPD.pth")
    checkpoints = {"MPD": {"path": str(mpd), "sha256": c.sha(mpd)}}
    if method in ("dense", "sparse", "random"):
        checkpoints.update(
            G={
                "path": engine.mixer.checkpoint_path,
                "sha256": engine.mixer.checkpoint_sha256,
            },
            R_phi={
                "path": str(
                    getattr(
                        engine,
                        "residual_path",
                        c.CHECKPOINTS / "R_ft.pt",
                    )
                ),
                "sha256": c.sha(
                    getattr(
                        engine,
                        "residual_path",
                        c.CHECKPOINTS / "R_ft.pt",
                    )
                ),
            },
        )
    if method in ("sparse", "random"):
        checkpoints["U"] = {
            "path": engine.u.checkpoint_path,
            "sha256": engine.u.checkpoint_sha256,
        }
    final = directory / (method + "_final.npz")
    timestamps = np.load(final)["timestamps"].tolist() if final.exists() else None
    result = dict(
        physical_output_npz=str(final) if final.exists() else None,
        auxiliary_meaning="forward_displacement_m_per_sample",
        native_status=row["native_status"],
        native_success=row["native_status"] == "SUCCESS",
        common_valid=bool(
            row["final_quality"] and row["final_quality"]["complete_valid"]
        ),
        failure_reasons=row["failure_reasons"],
        root_seconds=row["root_seconds"],
        repair_seconds=row["repair_seconds"],
        post_check_seconds=row["post_check_seconds"],
        complete_total_seconds=row["complete_total_seconds"],
        execution_makespan_seconds=row["completion"][
            "scheduled_execution_makespan_seconds"
        ],
        request_to_completion_seconds=row["completion"][
            "request_to_completion_seconds"
        ],
        output_timestamps_s=timestamps,
        output_horizon=len(timestamps) if timestamps else None,
        primary_success=row["complete_success"],
        all_attempt_record=str(directory / (method + ".json")),
    )
    value = dict(
        contract_version=getattr(engine, "contract_version", "n100-empty-v1"),
        task=dict(
            family=scene["family"],
            map=scene.get("model_id", "EnvEmpty2D"),
            scene_id=scene["configuration_id"],
            N=len(scene["starts"]),
            radii_m=scene["radii"],
            starts_m=scene["starts"],
            goals_m=scene["goals"],
            coordinate_frame="Empty2D Cartesian metres",
            nominal_duration_s=5,
            timestamps_s=[i * c.DT for i in range(64)],
            start_offsets_s=[0] * len(scene["starts"]),
            horizon=64,
            source_split=scene["split"],
            parent_geometry_id=scene["parent_geometry_id"],
            input_sha256=scene["input_sha256"],
        ),
        artifacts=dict(
            source_sha="v1.0.0",
            model=method,
            checkpoints=checkpoints,
            normalizer=normalizer,
        ),
        result=result,
    )
    c.write(directory / (method + ".contract.json"), value)
    return value
