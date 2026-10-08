"""Official local MMD-xECBS setup and exact frozen-task plumbing."""
from __future__ import annotations
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
import torch
from mmd_zero_shot_adapter import ROOT, MMD, official_environment, official_geometry, official_metrics
sys.path.insert(0,str(MMD/'deps/experiment_launcher'))

ZERO = ROOT/'results/mmd_maps_zero_shot_smd_akule_20261005'
OUT = ROOT/'results/mmd_maps_paired_akule_vs_mmd_xecbs_20261005'
MANIFEST = ZERO/'TASK_MANIFEST.json'
MODEL_IDS = {'highways':'EnvHighways2D-RobotPlanarDisk',
             'conveyor':'EnvConveyor2D-RobotPlanarDisk',
             'drop_region':'EnvDropRegion2D-RobotPlanarDisk'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def scene_hash(scene):
    payload={k:v for k,v in scene.items() if k!='input_sha256'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()

def prototype(family, device, output):
    from mmd.config.mmd_params import MMDParams as p
    p.device=device; p.tensor_args={'device':torch.device(device),'dtype':torch.float32}
    names=('use_guide_on_extra_objects_only','n_samples','n_local_inference_noising_steps',
        'n_local_inference_denoising_steps','start_guide_steps_fraction','n_guide_steps',
        'n_diffusion_steps_without_noise','weight_grad_cost_collision','weight_grad_cost_smoothness',
        'weight_grad_cost_constraints','weight_grad_cost_soft_constraints',
        'factor_num_interpolated_points_for_collision','trajectory_duration','debug','seed')
    cwd=Path.cwd(); compile_fn=torch.compile
    # CPU audit only. GPU evaluation retains official compilation/warmup.
    if device=='cpu':torch.compile=lambda model,*a,**kw:model
    try:
        os.chdir(MMD/'scripts/inference')
        import mmd.planners.single_agent.mpd_ensemble as module
        # Earlier canonical imports may initialize Git-based paths from the outer repo.
        # Always use the unchanged official released trajectory assets.
        import mmd.datasets.trajectories as datasets_module
        datasets_module.dataset_base_dir=str(MMD/'data_trajectories')
        module.TRAINED_MODELS_DIR=str(MMD/'data_trained_models')
        return module.MPDEnsemble(model_ids=(MODEL_IDS[family],),
            transforms={0:torch.zeros(2,device=device)},planner_alg='mmd',
            start_state_pos=torch.tensor([-.8,-.8],device=device),
            goal_state_pos=torch.tensor([.8,.8],device=device),device=device,
            results_dir=str(output),trained_models_dir=str(MMD/'data_trained_models'),
            **{name:getattr(p,name) for name in names})
    finally:
        os.chdir(cwd);torch.compile=compile_fn

def configure(proto,scene):
    """Independent planner state; only immutable models/datasets/tasks shared."""
    from mmd.planners.multi_agent import CBS
    from mmd.common.conflicts import PointConflict
    from mmd.common.constraints import MultiPointConstraint
    assert official_geometry(proto.datasets[0].env)==scene['obstacles']
    starts=torch.tensor(scene['starts'],**proto.tensor_args)
    goals=torch.tensor(scene['goals'],**proto.tensor_args)
    shared=[*proto.models.values(),*proto.datasets,proto.task,proto.robot]
    bank=[copy.deepcopy(proto,{id(x):x for x in shared}) for _ in scene['starts']]
    for agent,low in enumerate(bank):
        hard=low.datasets[0].get_single_pt_hard_conditions(starts[agent],0,True)
        hard.update(low.datasets[0].get_single_pt_hard_conditions(goals[agent],-1,True))
        low.hard_conds={0:hard}
        low.start_state_pos=starts[agent].clone();low.goal_state_pos=goals[agent].clone()
    return CBS(bank,list(starts.unbind()),list(goals.unbind()),start_time_l=[0]*len(bank),
        is_xcbs=True,is_ecbs=True,conflict_type_to_constraint_types={PointConflict:{MultiPointConstraint}},
        reference_robot=proto.robot,reference_task=proto.task)

def mmd_metrics(native,scene,env=None):
    # The saved native MMD channels are m/s. Adapt only evaluator input to the
    # already-frozen Akule evaluator's displacement contract, never positions.
    physical=native.detach().cpu().numpy().copy()
    physical[...,2:] *= 5./64.
    result=official_metrics(physical,scene,env)
    result['velocity_conversion']='native MMD m/s -> evaluator displacement input (*5/64) -> native m/s'
    return result
