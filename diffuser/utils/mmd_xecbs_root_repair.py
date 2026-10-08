"""Inject a joint proposal into MMD XECBS without changing its repair logic."""
from __future__ import annotations

import time

import torch


MANIFEST_ENDPOINT_ATOL = 1e-7


def prepare_manifest_joint_paths(
    physical, starts, goals, *, atol=MANIFEST_ENDPOINT_ATOL
):
    """Audit and canonically convert grouped ``[B,H,N,4]`` paths for XECBS.

    Hard conditioning is exact in normalized MPD coordinates.  The official
    dataset normalizer's inverse can introduce single-precision roundoff when
    mapping a normalized zero back to a physical zero.  Accept only that
    bounded inverse-normalization error, then restore the exact physical
    manifest endpoint state ``[x, y, 0, 0]`` before constructing MMD paths.
    No agent/time permutation occurs here: agent ``i`` is always
    ``physical[0, :, i]``.
    """
    if physical.ndim != 4 or physical.shape[0] != 1 or physical.shape[-1] != 4:
        raise ValueError("physical root must be a single grouped [B,H,N,4] tensor")
    starts = starts.to(device=physical.device, dtype=physical.dtype)
    goals = goals.to(device=physical.device, dtype=physical.dtype)
    if starts.shape != (physical.shape[2], 2) or goals.shape != starts.shape:
        raise ValueError("manifest endpoints must be [N,2] in grouped agent order")

    def per_agent_error(actual, expected):
        return (actual - expected[None]).abs().amax(dim=(0, 2))

    before_start = per_agent_error(physical[:, 0, :, :2], starts)
    before_goal = per_agent_error(physical[:, -1, :, :2], goals)
    if bool((before_start > atol).any()) or bool((before_goal > atol).any()):
        raise AssertionError(
            "normalized-to-physical endpoint conversion exceeds the allowed "
            f"roundoff tolerance {atol}: start={before_start.tolist()}, "
            f"goal={before_goal.tolist()}"
        )

    exact_start = torch.cat((starts, torch.zeros_like(starts)), dim=-1)
    exact_goal = torch.cat((goals, torch.zeros_like(goals)), dim=-1)
    canonical = physical.clone()
    canonical[:, 0] = exact_start
    canonical[:, -1] = exact_goal
    joint_paths = [
        canonical[0, :, agent].contiguous() for agent in range(canonical.shape[2])
    ]
    after_start = torch.stack([path[0, :2] for path in joint_paths])
    after_goal = torch.stack([path[-1, :2] for path in joint_paths])
    if not torch.equal(after_start, starts) or not torch.equal(after_goal, goals):
        raise AssertionError("canonical XECBS paths changed manifest endpoint ordering")
    return (
        canonical,
        joint_paths,
        {
            "grouped_trajectory_shape": list(physical.shape),
            "manifest_start_shape": list(starts.shape),
            "manifest_goal_shape": list(goals.shape),
            "per_agent_start_max_abs_before_conversion": before_start.cpu().tolist(),
            "per_agent_goal_max_abs_before_conversion": before_goal.cpu().tolist(),
            "per_agent_start_max_abs_after_conversion": (after_start - starts)
            .abs()
            .amax(dim=-1)
            .cpu()
            .tolist(),
            "per_agent_goal_max_abs_after_conversion": (after_goal - goals)
            .abs()
            .amax(dim=-1)
            .cpu()
            .tolist(),
            "endpoint_roundoff_atol": atol,
            "trajectory_coordinates": "physical after normalizer.unnormalize",
            "path_coordinates": "physical MMD [H,4], path i = grouped[0,:,i]",
        },
    )


def injected_root(search_state_class, xecbs, joint_paths, starts, goals):
    """Build the exact ``SearchState`` representation consumed by MMD XECBS.

    ``CBS.expand`` warms MPD with ``state.path_bl[agent_id]``.  MMD's local
    inference requires that seed batch to match ``planner.num_samples``; each
    injected path is therefore repeated only along the candidate-batch axis.
    The selected root path remains candidate zero and is not modified.
    """
    if len(joint_paths) != xecbs.num_agents:
        raise ValueError("joint proposal must contain one path per XECBS agent")
    if len(starts) != xecbs.num_agents or len(goals) != xecbs.num_agents:
        raise ValueError("start/goal count must match XECBS population")
    path_batches = []
    for agent, path in enumerate(joint_paths):
        if path.ndim != 2 or path.shape[-1] != 4:
            raise ValueError("every injected Boundary path must be [H,4]")
        low_level = xecbs.low_level_planner_l[agent]
        if path.shape[0] != low_level.n_support_points:
            raise ValueError("injected path horizon must equal MPD n_support_points")
        if not torch.equal(path[0, :2], starts[agent]):
            raise ValueError("injected path start does not match XECBS start")
        if not torch.equal(path[-1, :2], goals[agent]):
            raise ValueError("injected path goal does not match XECBS goal")
        path = path.to(**xecbs.tensor_args)
        path_batches.append(path[None].expand(low_level.num_samples, -1, -1).clone())
    root = search_state_class([0] * xecbs.num_agents, path_batches)
    root.update_g_l2()
    root.conflict_l = xecbs.get_conflicts(root)
    return root


def repair_from_injected_root(xecbs, root, runtime_limit, success_status_class,
                              accept_state=None):
    """Run MMD's existing XECBS open-list/conflict/expand loop from ``root``."""
    start = time.perf_counter()
    status = success_status_class.UNKNOWN
    state = root
    xecbs.open_l.clear()
    xecbs.open_l.append(root)
    expansions = 0
    while status == success_status_class.UNKNOWN:
        if not xecbs.open_l:
            status = success_status_class.FAIL_NO_SOLUTION
            break
        xecbs.open_l.sort(key=lambda candidate: len(candidate.conflict_l))
        state = xecbs.open_l.pop(0)
        if not state.conflict_l:
            candidate_paths = [
                state.path_bl[agent][path_index].squeeze(0)
                for agent, path_index in enumerate(state.ix_best_path_in_batch_l)
            ]
            if accept_state is None or accept_state(candidate_paths):
                status = success_status_class.SUCCESS
                break
            # A post-smoothing static or swept failure is never called solved.
            # All children were screened before insertion; this is a final
            # independent guard against an implementation mismatch.
            continue
        # This is the unmodified MMD CBS.expand implementation: it converts
        # conflicts to constraints and invokes official MPDEnsemble replans.
        xecbs.expand(state)
        expansions += 1
        if time.perf_counter() - start > runtime_limit:
            status = success_status_class.FAIL_RUNTIME_LIMIT
            break
    paths = [
        state.path_bl[agent][path_index].squeeze(0)
        for agent, path_index in enumerate(state.ix_best_path_in_batch_l)
    ]
    from mmd.common import global_pad_paths

    return (
        global_pad_paths(paths, xecbs.start_time_l),
        expansions,
        status,
        len(state.conflict_l),
        time.perf_counter() - start,
    )
