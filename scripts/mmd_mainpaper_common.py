"""Official-function contracts for the fixed two-map paper panel."""
from __future__ import annotations
import contextlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from mmd_paired_common import ROOT, MMD, ZERO, MANIFEST, MODEL_IDS, sha
from mmd_zero_shot_adapter import official_environment, official_geometry, official_metrics

OUT = ROOT/'results/mmd_highways_conveyor_mainpaper_20261006'
PRIOR = ROOT/'results/mmd_akule_environment_prior_20261005'
PAIRED = ROOT/'results/mmd_maps_paired_akule_vs_mmd_xecbs_20261005'

def panel():
    return [s for s in json.loads(MANIFEST.read_text())['scenes'] if s['family'] in ('highways','conveyor')]

class NativeEvaluator:
    def __init__(self,family,device='cpu'):
        from torch_robotics.robots import RobotPlanarDisk
        from torch_robotics.tasks.tasks import PlanningTask
        self.args={'device':device,'dtype':torch.float32}
        self.env=official_environment(family,device)
        self.robot=RobotPlanarDisk(radius=.05,tensor_args=self.args)
        self.task=PlanningTask(env=self.env,robot=self.robot,tensor_args=self.args)

    def requested_valid(self,scene):
        from mmd.common.multi_agent_utils import is_multi_agent_start_goal_states_valid
        assert official_geometry(self.env)==scene['obstacles']
        starts=torch.tensor(scene['starts'],**self.args).unbind()
        goals=torch.tensor(scene['goals'],**self.args).unbind()
        with contextlib.redirect_stdout(io.StringIO()):
            return bool(is_multi_agent_start_goal_states_valid(self.robot,self.task,list(starts),list(goals)))

    def check(self,physical,scene,certificate=None):
        """No endpoint-equality condition; original official candidate/conflict APIs.

        certificate is the pre-smoothing low-level success flag for MPD/MMD.
        With no certificate, Akule's unsmoothed final candidate is filtered here.
        """
        from mmd.planners.multi_agent.cbs import CBS,SearchState
        from mmd.common.conflicts import PointConflict
        q=torch.as_tensor(physical,**self.args)
        assert q.ndim==3 and q.shape[1]==scene['population'] and q.shape[-1]==4
        assert torch.isfinite(q).all()
        batch=q.permute(1,0,2).contiguous()
        if certificate is None:
            _,_,_,free,_=self.task.get_trajs_collision_and_free(batch,return_indices=True)
            free_ids=set(free.reshape(-1).tolist())
            invalid=[a for a in range(scene['population']) if a not in free_ids]
            certified=not invalid
        else:
            certified=bool(certificate);invalid=None
        planner=CBS.__new__(CBS)
        planner.reference_robot=self.robot
        planner.start_time_l=[0]*scene['population']
        planner.conflict_type_to_constraint_types={PointConflict:set()}
        state=SearchState([],[])
        state.path_bl=[p[None] for p in batch]
        state.ix_best_path_in_batch_l=[0]*scene['population']
        conflicts=planner.get_conflicts(state)
        # Exact final runner scan (not a replacement for CBS's .105 m check).
        final_pairs=0
        for a in range(scene['population']):
            for b in range(a+1,scene['population']):
                final_pairs+=int((torch.norm(q[:,a,:2]-q[:,b,:2],dim=-1)<.1).sum())
        requested=self.requested_valid(scene)
        return {'native_feasible':bool(requested and certified and not conflicts and final_pairs==0),
                'requested_start_goal_valid':requested,'low_level_free_certificate':certified,
                'invalid_candidate_agents':invalid,'point_conflicts_105mm':len(conflicts),
                'final_pair_collisions_100mm':final_pairs,'returned_endpoint_equality_checked':False}

def metrics(physical,scene,native_velocity=False,env=None):
    q=np.asarray(physical).copy()
    if native_velocity:q[...,2:]*=5/64
    with contextlib.redirect_stdout(io.StringIO()):
        return official_metrics(q,scene,env)

def save(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
