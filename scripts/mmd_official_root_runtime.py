"""Uncached production root generation with the released MPDEnsemble.

No result writing or external evaluation is performed here. Construct the
official configured planner before entering the complete-planner timer, as
the local MMD runner does. Calling generate() always pays for native planning.
"""
from dataclasses import dataclass
import time
import torch

@dataclass
class OfficialRoot:
    selected_pre_smoothing: torch.Tensor
    selected_returned: torch.Tensor
    candidate_banks_returned: list
    native_outputs: list
    selected_indices: list
    free_candidate_indices: list
    wall_seconds: float

def synchronize(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)

def generate(planner):
    """Independent official low-level calls, preserving all candidate banks.

    constraints=None is intentional: MMD's prioritized root constraints are
    part of MMD coordination, not the independent shared single-agent prior.
    Candidate filtering, selection, guidance and smoothing occur inside each
    unchanged official call. Empty native candidate sets fail explicitly.
    """
    device=planner.start_state_pos_l[0].device
    synchronize(device);t0=time.perf_counter()
    pre=[];returned=[];banks=[];outputs=[];indices=[];free_indices=[]
    for agent,low in enumerate(planner.low_level_planner_l):
        output=low(planner.start_state_pos_l[agent],planner.goal_state_pos_l[agent],
            constraints_l=None,experience=None)
        free=output.trajs_final_free_idxs
        if not free.numel():
            raise RuntimeError(f'Official MPDEnsemble returned no native-free candidate for agent {agent}')
        index=int(output.idx_best_traj)
        # Deterministic state-unit conversion only; positions are unchanged.
        a=output.trajs_iters[-1,index].detach().clone()
        b=output.trajs_final[index].detach().clone()
        bank=output.trajs_final.detach().clone()
        for value in (a,b,bank):value[...,2:]*=5/64
        pre.append(a);returned.append(b);banks.append(bank)
        outputs.append(output);indices.append(index);free_indices.append(free.detach().clone())
    pre=torch.stack(pre,1);returned=torch.stack(returned,1)
    synchronize(device);seconds=time.perf_counter()-t0
    return OfficialRoot(pre,returned,banks,outputs,indices,free_indices,seconds)

def injected_root(search_state_class, planner, coordinated_paths, official_root):
    """Initialize unchanged hard search with real unary candidate banks.

    Only the selected candidate is replaced by the coordinated path. The
    other 63 candidates keep their original values/order, matching released
    experience reuse rather than duplicating the selected path. The caller
    supplies already validated/coordinated physical-displacement states.
    """
    if len(coordinated_paths)!=planner.num_agents or len(official_root.candidate_banks_returned)!=planner.num_agents:
        raise ValueError('Candidate bank and coordinated root population mismatch')
    banks=[]
    for agent,path in enumerate(coordinated_paths):
        bank=official_root.candidate_banks_returned[agent].clone()
        low=planner.low_level_planner_l[agent]
        if tuple(bank.shape)!=(low.num_samples,low.n_support_points,4) or tuple(path.shape)!=(low.n_support_points,4):
            raise ValueError('Official experience bank shape mismatch')
        if not torch.isfinite(path).all():raise ValueError('Nonfinite coordinated root')
        bank[official_root.selected_indices[agent]]=path.to(bank)
        banks.append(bank)
    state=search_state_class(list(official_root.selected_indices),banks)
    state.update_g_l2();state.conflict_l=planner.get_conflicts(state)
    return state
