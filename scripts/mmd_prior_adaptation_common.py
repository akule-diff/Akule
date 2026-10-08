"""Geometry and frozen-prior plumbing for disjoint MMD interaction adaptation."""
import contextlib
import json
from pathlib import Path
import numpy as np
import torch
from functools import lru_cache
from mmd_mainpaper_common import ROOT, NativeEvaluator, metrics, sha
from mmd_zero_shot_planner import load_final_engine, rollout
from mmd_paired_common import prototype
from mmd_structured_bridge import NativeVelocityCodec
from diffuser.models.mpd_v2 import MPDUnaryAdapter
from diffuser.utils import quality_sparse_v1 as q
from diffuser.models.quality_dynamic_u_v2 import full_support
from mmd_final_clean_runtime import NullLog

OUT=ROOT/'results/mmd_prior_interaction_adaptation_20261006'
PREVIOUS=ROOT/'results/mmd_highways_conveyor_final_20261006'

def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def setup(device='cuda'):
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'setup.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        engine,_,frozen=load_final_engine(device)
        protos={f:prototype(f,device,OUT/'setup_environment') for f in ['highways','conveyor']}
    priors={}
    for f,p in protos.items():
        diffusion=getattr(p.models[0],'_orig_mod',p.models[0])
        normal=p.datasets[0].normalizer.normalizers['traj']
        priors[f]=(MPDUnaryAdapter(diffusion).eval().requires_grad_(False),NativeVelocityCodec({'minimum':normal.mins.tolist(),'maximum':normal.maxs.tolist()},device,native_velocity_is_mps=True))
    return engine,priors,protos,frozen

def bind(engine,priors,scene):
    engine.unary,engine.codec=priors[scene['family']]

def clearance_tensor(positions,scene):
    centers=positions.new_tensor([o['center'] for o in scene['obstacles']['items']])
    size=positions.new_tensor([o['size'] for o in scene['obstacles']['items']])
    radius=positions.new_tensor([o['corner_radius'] for o in scene['obstacles']['items']])
    d=(positions[...,None,:]-centers).abs()-(size/2-radius[:,None])
    return d.clamp_min(0).norm(dim=-1)+d.amax(-1).clamp_max(0)-radius-.05

@lru_cache(maxsize=2)
def evaluator_for(family):
    return NativeEvaluator(family)

def stats(physical,scene,unary=None,evaluator=None):
    p=torch.as_tensor(np.asarray(physical),dtype=torch.float32)
    clear=clearance_tensor(p[...,:2],scene).amin(-1).amin(0)
    invalid=clear<0
    bounds=p.new_tensor(scene['workspace'])
    workspace_invalid=((p[...,:2]<bounds[0])|(p[...,:2]>bounds[1])).any(-1).any(0)
    i,j=torch.triu_indices(p.shape[1],p.shape[1],1)
    dist=(p[:,i,:2]-p[:,j,:2]).norm(dim=-1);conflict=dist<.105
    agents=torch.zeros(p.shape[1],dtype=torch.bool)
    ids=conflict.any(0);agents[i[ids]]=True;agents[j[ids]]=True
    base_valid=~invalid if unary is None else ~torch.as_tensor(unary['static_invalid_mask'])
    initially_valid=int(base_valid.sum());new_invalid=int((base_valid&invalid).sum())
    ev=evaluator or evaluator_for(scene['family'])
    return {'sampled_static_invalid_agents':int(invalid.sum()),'static_invalid_mask':invalid.tolist(),
        'static_agent_clearance_m':clear.tolist(),'minimum_obstacle_clearance_m':float(clear.min()),
        'workspace_invalid_agents':int(workspace_invalid.sum()),'pair_conflicts':int(conflict.sum()),
        'conflicted_agents':int(agents.sum()),'minimum_pair_clearance_m':float(dist.min()-.1),
        'initially_valid_agents':initially_valid,'initially_valid_becoming_invalid':new_invalid,
        'initially_valid_preserved_fraction':1-new_invalid/initially_valid if initially_valid else None,
        'D':metrics(np.asarray(physical),scene,env=ev.env)['D'],
        'goal_max_error_m':float((p[-1,:,:2]-p.new_tensor(scene['goals'])).norm(dim=-1).max())}

def aggregate(rows):
    groups=[]
    for family in ['highways','conveyor','overall']:
        for n in [3,6,9,12,15,20,None]:
            cell=[r for r in rows if (family=='overall' or r['family']==family) and (n is None or r['N']==n)]
            if not cell:continue
            stages={}
            for name in cell[0]['stages']:
                values=[r['stages'][name] for r in cell]
                stages[name]={k:float(np.mean([v[k] for v in values if v.get(k) is not None])) for k in
                    ['sampled_static_invalid_agents','pair_conflicts','conflicted_agents','minimum_obstacle_clearance_m','minimum_pair_clearance_m','D']}
                count=sum(v['initially_valid_agents'] for v in values)
                lost=sum(v['initially_valid_becoming_invalid'] for v in values)
                stages[name]['fraction_initially_valid_becoming_invalid']=lost/count if count else None
            groups.append({'map':family,'N':n,'scenes':len(cell),'stages':stages})
    return groups

def frozen_fields(engine,scene,base):
    _,ends,endpoints=engine.conditions(scene)
    index=full_support(base).nonzero()
    with torch.no_grad():
        values=engine.all_fields_with_grad(base,endpoints,torch.tensor([0],device=engine.device),index)
    n=base.shape[2];fields=base.new_zeros((1,n,n,64,2));fields[tuple(index.T)]=values.detach()
    return fields,index,ends

def coefficient_capacity(engine,scene,path,steps=240,native_gate=False):
    """Canonical bounded signed-field Adam capacity search; no network updates.

    Uses the SMD capacity recipe (240 steps, two starts, lr .025, [-1,1]
    bounds). Rounded-box geometry replaces the SMD circle geometry only.
    Static validity ranks first, conflicts second; official D is a guard only.
    """
    from diffuser.utils.physical_teacher import swept_distances
    base=torch.as_tensor(path,dtype=torch.float32,device=engine.device)[None].clone()
    fields,index,ends=frozen_fields(engine,scene,base)
    n=base.shape[2];_,i,j=index.T;basis=q.spline_matrix(8,device=engine.device)
    scale=float(engine.signed.g.timestep_scales[0])
    ev=evaluator_for(scene['family'])
    def score(candidate,initial=None):
        result=stats(candidate,scene,initial,ev)
        if native_gate:
            value=torch.as_tensor(candidate,dtype=torch.float32).permute(1,0,2).contiguous()
            _,_,_,free,_=ev.task.get_trajs_collision_and_free(value,return_indices=True)
            valid_ids=set(free.flatten().tolist())
            mask=[agent not in valid_ids for agent in range(n)]
            prior_mask=mask if initial is None else initial['static_invalid_mask']
            protected=sum(not b for b in prior_mask)
            lost=sum(not before and after for before,after in zip(prior_mask,mask))
            result.update(analytic_sampled_static_invalid_agents=result['sampled_static_invalid_agents'],
                sampled_static_invalid_agents=sum(mask),native_static_invalid_agents=sum(mask),static_invalid_mask=mask,
                initially_valid_agents=protected,initially_valid_becoming_invalid=lost,
                initially_valid_preserved_fraction=1-lost/protected if protected else None,
                capacity_static_predicate='Exact official PlanningTask.get_trajs_collision_and_free pre-smoothing candidate predicate')
        return result
    initial=score(path)
    best=initial;best_path=path.copy();best_weights=base.new_zeros((len(index),8));history=[]
    def construct(weights):
        coeff=base.new_zeros((1,n,n,8));coeff[0,i,j]=scale*weights
        return q.endpoint_condition(q.compose_smooth(base,fields,coeff,basis),ends)
    for restart in range(2):
        weights=torch.nn.Parameter(base.new_zeros((len(index),8)) if restart==0 else .05*torch.sign(fields[0,i,j].mean((1,2)))[:,None].expand(-1,8).clone())
        optimizer=torch.optim.Adam([weights],lr=.025)
        for step in range(steps+1):
            output=construct(weights);pos=output[...,:2]
            # Six interpolation samples per interval: official candidate grid.
            a=pos[:,:-1];b=pos[:,1:]
            samples=torch.cat([a*(1-t)+b*t for t in [0,1/6,2/6,3/6,4/6,5/6]],dim=1)
            clear=clearance_tensor(samples,scene)
            obstacle=((.002-clear).clamp_min(0)/.05)
            bounds=pos.new_tensor(scene['workspace'])
            workspace=((bounds[0]-pos).clamp_min(0)+(pos-bounds[1]).clamp_min(0))/.05
            pair=(.105-swept_distances(output)).clamp_min(0)/.1
            loss=10*(obstacle.square().mean()+obstacle.square().amax()+workspace.square().mean()+workspace.square().amax())+pair.square().mean()+pair.square().amax()+.001*weights.square().mean()+.001*(pos-base[...,:2]).square().mean()
            if step%20==0 or step==steps:
                candidate=output[0].detach().cpu().numpy();record=score(candidate,initial)
                preserved=record['initially_valid_becoming_invalid']==0
                guard=record['D']>=initial['D']-.10
                rank=(record['sampled_static_invalid_agents'],record['pair_conflicts'])
                if preserved and guard and record['workspace_invalid_agents']==0 and rank<(best['sampled_static_invalid_agents'],best['pair_conflicts']):
                    best=record;best_path=candidate.copy();best_weights=weights.detach().clone()
                history.append({'restart':restart,'step':step,'loss':float(loss.detach()),'static_invalid':record['sampled_static_invalid_agents'],'pairs':record['pair_conflicts'],'D':record['D'],'preserved':preserved})
            if step==steps:break
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            with torch.no_grad():weights.clamp_(-1,1)
    return {'unary':initial,'optimized':best,'history':history,'coefficient_scale':scale,'coefficient_bound':[-1,1]},best_path,best_weights.cpu().numpy()
