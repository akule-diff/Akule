"""Aggregate every available fixed-test attempt and paired scene comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import canonical_n28_runtime as runtime

OUT = runtime.OUT
MANIFEST = runtime.ROOT / "artifacts/manifests/FINAL_N28_TEST_MANIFEST_V2.json"


METHODS = ("sparse", "dense", "unary", "official")
LABELS = dict(sparse="Sparse Ours", dense="Dense Ours", unary="Unary MPD", official="MMD")


def stats(values):
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(array):
        return dict(n=0, mean=None, median=None, p95=None, total=None)
    return dict(n=len(array), mean=float(array.mean()), median=float(np.median(array)),
                p95=float(np.percentile(array, 95)), total=float(array.sum()))


def metric(row, key, *, final=True):
    quality = row["final_quality" if final else "pre_repair_quality"]
    return quality.get(key) if quality is not None else None


def summarize(rows, target):
    n = len(rows)
    return dict(
        completed=n, target=target, remaining=target - n,
        final_success_fraction=sum(row["final_success"] for row in rows) / target,
        final_success_observed_fraction=sum(row["final_success"] for row in rows) / n if n else None,
        timeout_count=sum(row["timeout"] for row in rows),
        native_failure_count=sum(row["native_status"] != "SUCCESS" for row in rows),
        pre_repair_goal_completion_fraction=stats([
            row.get("pre_repair_goal_completion_fraction") for row in rows]),
        final_goal_completion_fraction=stats([
            row.get("final_goal_completion_fraction") for row in rows]),
        all_goals_completed_scene_fraction=sum(row["all_goals_completed"] for row in rows) / n if n else None,
        collision_free_scene_fraction=sum(
            row["post_repair_conflicts"] == 0 for row in rows) / n if n else None,
        pre_collision_free_scene_fraction=sum(
            row["pre_repair_conflicts"] == 0 for row in rows) / n if n else None,
        pre_conflicts=stats([row["pre_repair_conflicts"] for row in rows]),
        post_conflicts=stats([row["post_repair_conflicts"] for row in rows]),
        pre_conflicts_per_agent=stats([row["pre_repair_conflicts"] / 28
                                       if row["pre_repair_conflicts"] is not None else None for row in rows]),
        post_conflicts_per_agent=stats([row["post_repair_conflicts"] / 28
                                        if row["post_repair_conflicts"] is not None else None for row in rows]),
        proposal_seconds=stats([row["proposal_seconds"] for row in rows]),
        repair_seconds=stats([row["repair_seconds"] for row in rows]),
        total_planning_seconds=stats([row["total_planning_seconds"] for row in rows]),
        ct_expansions=stats([row["ct_expansions"] for row in rows]),
        low_level_calls=stats([row["low_level_calls"] for row in rows]),
        native_conflicts_entering_repair=stats([
            row.get("native_conflicts_entering_repair") for row in rows]),
        native_conflicts_remaining=stats([
            row.get("native_conflicts_remaining") for row in rows]),
        pre_path_m=stats([metric(row, "mean_path_length", final=False) for row in rows]),
        pre_gp=stats([metric(row, "gp_physical", final=False) for row in rows]),
        pre_acceleration_rms=stats([metric(row, "acceleration_rms", final=False) for row in rows]),
        pre_jerk_rms=stats([metric(row, "jerk_rms", final=False) for row in rows]),
        final_path_m=stats([metric(row, "mean_path_length") for row in rows]),
        final_gp=stats([metric(row, "gp_physical") for row in rows]),
        final_acceleration_rms=stats([metric(row, "acceleration_rms") for row in rows]),
        final_jerk_rms=stats([metric(row, "jerk_rms") for row in rows]),
        degree=stats([row["degree"] for row in rows]),
        true_r_ft_savings=stats([row["true_r_ft_savings"] for row in rows]),
        agent_arrival_seconds=stats([
            arrival for row in rows if row["final_success"]
            for arrival in row["completion"]["per_agent_goal_arrival_seconds"]
        ]),
        task_makespan_seconds=stats([
            row["completion"]["scheduled_execution_makespan_seconds"]
            for row in rows]),
        response_to_completion_seconds=stats([
            row["completion"]["request_to_completion_seconds"]
            for row in rows]),
    )


def paired(sparse, comparator, field):
    a = {row["index"]: row for row in sparse}
    b = {row["index"]: row for row in comparator}
    if field == "pre_conflicts":
        get = lambda row: row["pre_repair_conflicts"]
    elif field == "post_conflicts":
        get = lambda row: row["post_repair_conflicts"]
    elif field == "planning_seconds":
        get = lambda row: row["total_planning_seconds"]
    elif field == "makespan_seconds":
        get = lambda row: row["completion"]["scheduled_execution_makespan_seconds"]
    elif field == "path_m":
        get = lambda row: metric(row, "mean_path_length")
    else:
        raise ValueError(field)
    differences = [get(a[index]) - get(b[index]) for index in sorted(a.keys() & b.keys())
                   if get(a[index]) is not None and get(b[index]) is not None]
    if not differences:
        return dict(n=0, mean=None, standard_error=None, ci95=None)
    x = np.asarray(differences, dtype=float)
    se = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None
    mean = float(x.mean())
    return dict(n=len(x), mean=mean, standard_error=se,
                ci95=[mean - 1.96 * se, mean + 1.96 * se] if se is not None else None)


def report(count, smoke):
    root = OUT / "smoke" if smoke else OUT
    entries = json.loads(MANIFEST.read_text())["conditions"][:count]
    rows = {}
    for method in METHODS:
        rows[method] = []
        for condition in entries:
            path = root / method / f"{condition['index']:04d}" / "benchmark.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if (record["scene_id"] != condition["scene_id"]
                    or record["input_sha256"] != condition["scene"]["input_sha256"]):
                raise RuntimeError(f"Mismatched method scene: {path}")
            rows[method].append(record)
    summaries = {method: summarize(rows[method], count) for method in METHODS}
    comparisons = {
        method: {field: paired(rows["sparse"], rows[method], field)
                 for field in ("pre_conflicts", "post_conflicts", "planning_seconds",
                               "makespan_seconds", "path_m")}
        for method in ("dense", "unary", "official")
    }
    value = dict(manifest_sha256=runtime.sha(MANIFEST), target=count, smoke=smoke,
                 summaries=summaries, paired_sparse_minus_comparator=comparisons,
                 notes=dict(
                     failed_makespans="censored to null; makespan summaries and pairs include only valid completed plans",
                     missing_trajectories="not dropped from success denominator; metric-specific n disclosed",
                     collision_contract="sampled .100 m pair-times, all scene attempts retained",
                 ))
    path = OUT / ("SMOKE_SUMMARY.json" if smoke else "FINAL_BENCHMARK_SUMMARY.json")
    runtime.write(path, value)
    print(json.dumps({"progress": {method: (summary["completed"], count)
                                    for method, summary in summaries.items()},
                      "summary_path": str(path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    report(arguments.count, arguments.smoke)
