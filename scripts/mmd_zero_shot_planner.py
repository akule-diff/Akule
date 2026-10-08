"""CPU/CUDA evaluation adapter for the immutable final SMD Akule stack.

Reuses the frozen rollout, root repair, RRT parameters and XECBS search code.
Only geometry/device plumbing and evaluation accounting are supplied here.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mmd_zero_shot_adapter import (ROOT, MMD, MMDSpatialMapField, official_environment,
                                   official_geometry, official_metrics)

PINNED = ROOT
# Use the same frozen runtime and repair utilities as final SMD evaluation.
sys.path[:0] = [str(ROOT / 'integrations/smd_runtime/scripts'), str(ROOT / 'scripts'), str(ROOT)]
import canonical_n28_runtime as runtime
from canonical_n28_rollout import rollout
import obstacle_aware_repair as static_repair
from mmd_structured_bridge import load_base, limits_from_official_trajectories, NativeVelocityCodec
from diffuser.models.mpd_v2 import MPDUnaryAdapter, grouped_to_mpd_per_agent, mpd_per_agent_to_grouped
from diffuser.models.smd_goal_topological_mpd import GoalTopologicalMPD
from diffuser.models.canonical_set_g import SetContextG
from diffuser.models.quality_dynamic_u_v2 import DynamicSupportU
from diffuser.models.quality_sparse_v1 import SmoothPairResidual, SignedQualityMixer
from diffuser.utils.quality_sparse_v1 import spline_matrix, QualitySpec


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pinned(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repair = load_pinned('mmd_zero_shot_frozen_root_repair',
                     PINNED / 'diffuser/utils/mmd_xecbs_root_repair.py')
rrt = load_pinned('mmd_zero_shot_frozen_rrt', ROOT / 'scripts/smd_radius_native_rrt_static_fallback.py')


def load_final_engine(device='cpu'):
    """Load final weights directly, without loading a training corpus or fitting."""
    final = json.loads((ROOT / 'FINAL_SMD_AKULE.json').read_text())
    for key in ('unary', 'r', 'g', 'u', 'backend_source'):
        field = key + ('_path' if key == 'backend_source' else '_checkpoint_path')
        final[field] = str(ROOT / final[field])
    expected = {'unary': 'cd99bcff189953cda606761a16b8299c2dee76d89494b771370e55b7058c344a',
                'r': '90a6d3f2de1f9265066379338ff17c9e3636328e2ef0d564698c23f7d4299eb3',
                'g': 'e9d0cb8bd515236bbfc77c4dcecbc4f9fdbad753fbf502c26075a2a14f2bb084',
                'u': 'ea930df5c02d27585e1d56e5a0bf38ac4415fad7376210797d61c18dfe5ec3e2'}
    for key, digest in expected.items():
        if final[key+'_sha256'] != digest or sha(final[key+'_checkpoint_path']) != digest:
            raise RuntimeError('Frozen weight mismatch: '+key)
    if sha(final['backend_source_path']) != final['backend_source_sha256']:
        raise RuntimeError('Frozen backend source changed')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    engine = runtime.Engine.__new__(runtime.Engine)
    engine.device = torch.device(device)
    old_cwd = Path.cwd()
    try:
        diffusion, _, _ = load_base('nowait', ROOT/'external/mmd/data_trained_models', device=device)
    finally:
        os.chdir(old_cwd)
    engine.unary = MPDUnaryAdapter(diffusion).eval().requires_grad_(False)
    engine.normalizer = limits_from_official_trajectories('nowait')
    engine.codec = NativeVelocityCodec(engine.normalizer, device, native_velocity_is_mps=False)
    engine.spec = QualitySpec()
    engine.residual = SmoothPairResidual(spline_matrix(16, device=device)[:, 2:-2]).to(device)
    engine.residual.load_state_dict(torch.load(final['r_checkpoint_path'], map_location=device, weights_only=False)['model'])
    scales = runtime.read(PINNED/'artifacts/manifests/SIGNED_SCALES.json')['scales']
    engine.signed = SignedQualityMixer(spline_matrix(8, device=device), scales).to(device)
    engine.signed.g = SetContextG(scales).to(device)
    engine.signed.g.pair_chunk_size = 256
    engine.signed.g.load_state_dict(torch.load(final['g_checkpoint_path'], map_location=device, weights_only=False)['g'])
    engine.u = DynamicSupportU(width=48, pair_residual=True, pair_residual_width=192).to(device)
    engine.u.pair_chunk_size = 256
    engine.u.load_state_dict(torch.load(final['u_checkpoint_path'], map_location=device, weights_only=False)['u'])
    unary = GoalTopologicalMPD(copy.deepcopy(engine.unary.model), engine.codec).to(device)
    unary.load_state_dict(torch.load(final['unary_checkpoint_path'], map_location=device, weights_only=False)['model'])
    for module in (engine.residual, engine.signed, engine.u, unary):
        module.eval().requires_grad_(False)
    engine.setup_seconds = 0.
    return engine, unary, final


def bind_map(engine, unary, scene):
    field = MMDSpatialMapField(scene, device=engine.device)
    starts = torch.as_tensor(scene['starts'], device=engine.device, dtype=torch.float32)
    goals = torch.as_tensor(scene['goals'], device=engine.device, dtype=torch.float32)
    encoded = unary.encode_map(field, starts, goals)

    def base(x, timestep, hard):
        batch, _, agents, _ = x.shape
        flat = grouped_to_mpd_per_agent(x)
        eps = mpd_per_agent_to_grouped(unary(flat, timestep.repeat_interleave(agents),
            unary.context_from_state(flat, field, encoded)), batch, agents)
        mean = engine.unary.reverse_mean(x, eps, timestep, hard)
        c1 = engine.unary.diffusion.posterior_mean_coef1[timestep][:, None, None, None]
        c2 = engine.unary.diffusion.posterior_mean_coef2[timestep][:, None, None, None]
        return engine.codec.decode(engine.unary.apply_hard_conditions((mean-c2*x)/c1, hard)), c1, c2
    engine.base_with_grad = base
    return field


def segment_clearance(a, b, obstacles, robot_radius=.05):
    """Exact segment-to-rounded-box clearance outside the obstacle interior.

    A rounded rectangle is the Minkowski sum of its inset AABB and corner disk.
    Segment/AABB distance therefore gives exact separation; negative results
    are sufficient for collision rejection and need not be an interior SDF.
    """
    from common_strict_safety import _segment_box_distance
    return min((_segment_box_distance(np.asarray(a), np.asarray(b), np.asarray(o['center']),
                    np.asarray(o['size'])-2*o['corner_radius'])-o['corner_radius']-robot_radius
                for o in obstacles['items']), default=float('inf'))


def install_geometry(ensemble, world, *, tensor_args):
    from torch_robotics.environments.primitives import MultiRoundedBoxField, ObjectField
    field = MultiRoundedBoxField(np.asarray([o['center'] for o in world.obstacles['items']]),
                                np.asarray([o['size'] for o in world.obstacles['items']]), tensor_args=tensor_args)
    field.radius = torch.as_tensor([o['corner_radius'] for o in world.obstacles['items']], **tensor_args)
    obj = ObjectField([field], 'frozen-evaluation-geometry')
    task = ensemble.tasks[0]
    env = task.env
    # Replace objects on the existing task so all guidance field callbacks see
    # exactly this geometry; no old EmptyNoWait obstacles or duplicate map.
    env.obj_fixed_list = [obj]
    env.obj_extra_list = []
    env.obj_all_list = {obj}
    # Match the official local-map validator/guidance grid, including its
    # released .005 m sampling. Analytic unary conditioning remains exact.
    from torch_robotics.environments.grid_map_sdf import GridMapSDF
    env.grid_map_sdf_obj_fixed = GridMapSDF(env.limits,.005,env.obj_fixed_list,tensor_args=tensor_args)
    if ensemble.env is not env:
        ensemble.env.obj_fixed_list = [obj]
        ensemble.env.obj_extra_list = []
        ensemble.env.obj_all_list = {obj}


@dataclass(frozen=True)
class MMDWorld:
    scene: dict
    task: object
    obstacles: dict
    starts: np.ndarray
    goals: np.ndarray
    workspace: np.ndarray
    robot_radius: float = .05
    horizon: int = 64
    duration: float = 5.
    max_step: float | None = None
    # Keep final SMD repair's all-candidate post-smoothing screening and RRT
    # arclength dispatch. Collision semantics are supplied by path_valid below,
    # which calls official MMD; this label never selects SMD circle tolerances.
    contract: str = 'smd_native'

    @property
    def dt(self):
        return self.duration/self.horizon

    def path_valid(self, path, agent):
        values = np.asarray(path)
        if values.shape != (64,4) or not np.isfinite(values).all():
            return False
        if (np.max(np.abs(values[0,:2]-self.starts[agent]))>1e-5 or
                np.max(np.abs(values[-1,:2]-self.goals[agent]))>1e-5):
            return False
        if (values[:,:2] < self.workspace[0]+.05-1e-7).any() or (values[:,:2] > self.workspace[1]-.05+1e-7).any():
            return False
        if self.task.compute_collision(torch.as_tensor(values[:,:2],dtype=torch.float32)).any():
            return False
        # RRT's common_strict diagnostic additionally checks swept edges.
        if self.contract == 'common_strict':
            return all(segment_clearance(a,b,self.obstacles)>=-1e-9
                       for a,b in zip(values[:-1,:2],values[1:,:2]))
        return True

    def joint_audit(self, paths):
        paths = np.asarray(paths)
        n = paths.shape[1]
        i,j = np.triu_indices(n,1)
        pairs = int((np.linalg.norm(paths[:,i,:2]-paths[:,j,:2],axis=-1)<.1).sum())
        static = sum(not self.path_valid(paths[:,a],a) for a in range(n))
        return {'success': pairs==0 and static==0, 'pair_conflicts':pairs,'static_invalid_agents':static}


def make_world(scene):
    from torch_robotics.robots import RobotPlanarDisk
    from torch_robotics.tasks.tasks import PlanningTask
    env = official_environment(scene['family'])
    if official_geometry(env) != scene['obstacles'] or env.limits.tolist() != scene['workspace']:
        raise RuntimeError('Manifest geometry differs from official environment')
    args = {'device':'cpu','dtype':torch.float32}
    task = PlanningTask(env=env, robot=RobotPlanarDisk(radius=.05,tensor_args=args), tensor_args=args)
    return MMDWorld(scene,task,scene['obstacles'],np.asarray(scene['starts']),np.asarray(scene['goals']),np.asarray(scene['workspace']))


class LowLevel:
    def __init__(self, native, agent, backend):
        self.native, self.agent, self.backend = native, agent, backend

    def __getattr__(self,name):
        return getattr(self.native,name)

    def __call__(self,*args,**kwargs):
        record = {'agent_id':self.agent,'constraints':len(kwargs.get('constraints_l') or []),
                  'warm_start':kwargs.get('experience') is not None}
        self.backend.calls.append(record)
        self.backend.checkpoint()
        begun=time.monotonic()
        result=self.native(*args,**kwargs)
        record['mpd_seconds']=time.monotonic()-begun
        static_repair.filter_native_output(result,self.backend.world,self.agent,
            constraints=kwargs.get('constraints_l'))
        if record['constraints'] and not len(result.trajs_final_free_idxs):
            begun=time.monotonic()
            candidate,attempts=rrt.rrt_static_path(self.backend.world,self.agent,
                constraints=kwargs.get('constraints_l'), max_time_s=1.,seed_offsets=(37,0,73,149))
            record.update(rrt_fallback_seconds=time.monotonic()-begun,rrt_fallback_attempts=attempts,
                          rrt_fallback_accepted=candidate is not None)
            if candidate is not None:
                candidate=candidate.to(result.trajs_final)
                result.trajs_final=candidate[None].expand_as(result.trajs_final).clone()
                result.trajs_final_free_idxs=torch.zeros(1,device=candidate.device,dtype=torch.long)
                result.trajs_final_free=result.trajs_final[:1]
                result.traj_final_free_best=result.trajs_final[0]
                result.idx_best_traj=0
                result.success_free_trajs=True
                result.fraction_free_trajs=1./len(result.trajs_final)
        self.backend.checkpoint()
        return result


class RepairBackend:
    def __init__(self, device, output):
        from mmd.config.mmd_params import MMDParams as p
        import mmd.planners.single_agent.mpd_ensemble as native_module
        self.device=torch.device(device)
        p.tensor_args={'device':self.device,'dtype':torch.float32}
        p.device=str(device)
        native_module.TRAINED_MODELS_DIR=str(MMD/'data_trained_models')
        names=('use_guide_on_extra_objects_only','n_samples','n_local_inference_noising_steps',
               'n_local_inference_denoising_steps','start_guide_steps_fraction','n_guide_steps',
               'n_diffusion_steps_without_noise','weight_grad_cost_collision','weight_grad_cost_smoothness',
               'weight_grad_cost_constraints','weight_grad_cost_soft_constraints',
               'factor_num_interpolated_points_for_collision','trajectory_duration','debug','seed')
        kwargs={name:getattr(p,name) for name in names}
        old_cwd=Path.cwd()
        # CPU tests run the same model eagerly; CUDA keeps released compilation.
        compile_fn=torch.compile
        if self.device.type=='cpu': torch.compile=lambda model,*args,**kwargs:model
        try:
            os.chdir(MMD/'scripts/inference')
            self.prototype=native_module.MPDEnsemble(model_ids=('EnvEmptyNoWait2D-RobotPlanarDisk',),
                transforms={0:torch.zeros(2,device=device)},planner_alg='mmd',
                start_state_pos=torch.tensor([-.8,-.8],device=device),
                goal_state_pos=torch.tensor([.8,.8],device=device),device=str(device),
                results_dir=str(output),trained_models_dir=str(MMD/'data_trained_models'),**kwargs)
        finally:
            torch.compile=compile_fn
            os.chdir(old_cwd)
        # Existing frozen fallbacks, with geometry plumbing replaced only in
        # these isolated module objects. Never mutate files used by live jobs.
        rrt.install_native_obstacles=install_geometry
        rrt.obstacle_segment_clearance=segment_clearance
        static_repair.obstacle_segment_clearance=segment_clearance

    def checkpoint(self):
        if getattr(self,'progress',None) is not None:self.progress()

    def configure(self,scene,world):
        from mmd.planners.multi_agent.cbs import CBS
        from mmd.common.conflicts import PointConflict
        from mmd.common.constraints import MultiPointConstraint
        self.world,self.calls,self.root_rrt=world,[],[]
        self.ct_expansions=0
        self.post_static=None
        prototype=self.prototype
        install_geometry(prototype.task,world,tensor_args=prototype.tensor_args)
        shared=[*prototype.models.values(),*prototype.datasets,prototype.task,prototype.robot]
        bank=[copy.deepcopy(prototype,{id(x):x for x in shared}) for _ in scene['starts']]
        starts=torch.tensor(scene['starts'],device=self.device,dtype=torch.float32)
        goals=torch.tensor(scene['goals'],device=self.device,dtype=torch.float32)
        for agent,low in enumerate(bank):
            hard=low.datasets[0].get_single_pt_hard_conditions(starts[agent],0,True)
            hard.update(low.datasets[0].get_single_pt_hard_conditions(goals[agent],-1,True))
            low.hard_conds={0:hard}
            low.start_state_pos=starts[agent].clone();low.goal_state_pos=goals[agent].clone()
        wrappers=[LowLevel(low,agent,self) for agent,low in enumerate(bank)]
        planner=CBS(wrappers,list(starts.unbind()),list(goals.unbind()),start_time_l=[0]*len(bank),
            is_xcbs=True,is_ecbs=True,conflict_type_to_constraint_types={PointConflict:{MultiPointConstraint}},
            reference_robot=prototype.robot,reference_task=prototype.task)
        def expand(planner,state):
            repair.expand_all_branches(planner,state)
            self.ct_expansions+=1
            self.checkpoint()
        planner.expand=types.MethodType(expand,planner)
        return planner,starts,goals

    def solve(self,scene,proposal,world,deadline):
        from mmd.planners.multi_agent.cbs import SearchState
        from mmd.common.experiences import PathBatchExperience
        from mmd.common.experiments import TrialSuccessStatus
        planner,starts,goals=self.configure(scene,world)
        physical,paths,audit=repair.prepare_manifest_joint_paths(proposal,starts,goals)
        pre=world.joint_audit(physical[0].detach().cpu().numpy())
        def fallback(agent):
            self.checkpoint()
            candidate,attempts=rrt.rrt_static_path(world,agent)
            self.root_rrt.append({'agent':agent,'attempts':attempts,'accepted':candidate is not None})
            self.checkpoint()
            return None if candidate is None else candidate.to(self.device)
        begun=time.monotonic()
        paths,replaced=static_repair.repair_root_with_native_low_level(planner,paths,world,
            PathBatchExperience,static_fallback=fallback)
        static_seconds=time.monotonic()-begun
        root=torch.stack(paths,1)
        post=world.joint_audit(root.detach().cpu().numpy())
        self.post_static=post
        self.checkpoint()
        state=repair.injected_root(SearchState,planner,paths,starts,goals)
        final,ct,status,conflicts,_=repair.repair_from_injected_root(planner,state,
            max(.001,deadline-time.monotonic()),TrialSuccessStatus,
            accept_state=lambda paths:world.joint_audit(torch.stack(paths,1).detach().cpu().numpy())['success'])
        return torch.stack(final,1),root,{'native_status':status.name,'physical_complete_success':bool(status),
            'root_pair_conflicts':pre['pair_conflicts'],'root_static_invalid_agents':pre['static_invalid_agents'],
            'post_static_pair_conflicts':post['pair_conflicts'],'post_static_invalid_agents':post['static_invalid_agents'],
            'ct_expansions':ct,'low_level_calls':len(self.calls),'low_level_call_records':self.calls,
            'root_rrt_fallbacks':len(self.root_rrt),'root_rrt_records':self.root_rrt,
            'child_rrt_fallbacks':sum('rrt_fallback_seconds' in c for c in self.calls),
            'static_replaced_agents':replaced,'root_static_repair_seconds':static_seconds,
            'final_native_conflicts':conflicts,'endpoint_conversion_audit':audit}
