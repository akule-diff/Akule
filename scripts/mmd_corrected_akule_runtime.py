"""Official selected unary -> canonical terminal composition -> existing search.

Search expansion/open-list and RRT fallback are reused. Search trajectories,
candidate banks and constrained native calls retain official m/s units.
Only the learned interaction interface uses displacement velocities.
"""
import contextlib
import copy
import json
import signal
import time
import types
import os
import subprocess
from dataclasses import replace,fields
import numpy as np
import torch
from diffuser.models.quality_dynamic_u_v2 import full_support
from mmd_official_root_runtime import generate,injected_root,synchronize
from mmd_paired_common import configure
from mmd_mainpaper_common import NativeEvaluator,metrics
from mmd_final_clean_runtime import NullLog,Deadline,GPUComponents
from mmd_zero_shot_planner import repair,rrt,make_world,MMDWorld,install_geometry,segment_clearance

# Same exact rounded-box bindings used by the established MMD adapter.
# The fallback algorithm and its budgets are unchanged.
rrt.install_native_obstacles=install_geometry
rrt.obstacle_segment_clearance=segment_clearance
from mmd.common.experiences import PathBatchExperience
from mmd.common.experiments import TrialSuccessStatus
from mmd.planners.multi_agent.cbs import SearchState
from mmd.common import smooth_trajs

def native_free(task,physical):
    values=physical.permute(1,0,2).contiguous()
    _,_,_,free,_=task.get_trajs_collision_and_free(values,return_indices=True)
    return set(free.flatten().tolist())

def pair_counts(physical):
    i,j=torch.triu_indices(physical.shape[1],physical.shape[1],1,device=physical.device)
    distances=(physical[:,i,:2]-physical[:,j,:2]).norm(dim=-1)
    return int((distances<.105).sum()),int((distances<.105).any(0).sum())

def compose(engine,scene,physical,mode='dense',timers=None):
    """Exactly the canonical t=0 mixer tested by the corrected-root gate."""
    _,_,endpoints=engine.conditions(scene);base=physical[None]
    timestep=torch.zeros(1,dtype=torch.long,device=base.device)
    section=timers.section if timers else lambda _:contextlib.nullcontext()
    with section('U'):
        valid=full_support(base)
        support=valid if mode=='dense' else valid*(engine.u.logits(base,endpoints,timestep)>0)
        index=support.nonzero()
    with section('R'):fields=engine.all_fields_with_grad(base,endpoints,timestep,index)
    with section('G'):output,coeff=engine.signed.compose(base,endpoints,timestep,index,fields,support.to(base))
    return output[0],support,coeff

class NativeLowLevel:
    def __init__(self,low,agent,calls,world):self.low=low;self.agent=agent;self.calls=calls;self.world=world;self.phase='hard'
    def __getattr__(self,name):return getattr(self.low,name)
    def __call__(self,*args,**kwargs):
        device=self.low.tensor_args['device'];synchronize(device);begun=time.perf_counter()
        record={'agent':self.agent,'phase':self.phase,'constraints':len(kwargs.get('constraints_l') or []),
            'experience':kwargs.get('experience') is not None,'native_velocity_units':'m/s'}
        try:
            out=self.low(*args,**kwargs)
            if record['constraints'] and not out.trajs_final_free_idxs.numel():
                # Existing Akule fallback and budget; no behavior objective.
                started=time.perf_counter()
                candidate,attempts=rrt.rrt_static_path(self.world,self.agent,constraints=kwargs.get('constraints_l'),max_time_s=1.,seed_offsets=(37,0,73,149))
                record.update(rrt_seconds=time.perf_counter()-started,rrt_attempts=attempts,rrt_accepted=candidate is not None)
                if candidate is not None:
                    candidate=candidate.to(out.trajs_final);candidate[...,2:]*=64/5
                    out.trajs_final=candidate[None].expand_as(out.trajs_final).clone()
                    out.trajs_final_free_idxs=torch.zeros(1,device=candidate.device,dtype=torch.long)
                    out.trajs_final_free=out.trajs_final[:1];out.traj_final_free_best=out.trajs_final[0];out.idx_best_traj=0
                    out.success_free_trajs=True;out.fraction_free_trajs=1/len(out.trajs_final)
            return out
        finally:
            synchronize(device);record['seconds']=time.perf_counter()-begun;self.calls.append(record)

class NativeWorld(MMDWorld):
    def path_valid(self,path,agent):
        values=torch.as_tensor(np.asarray(path),dtype=torch.float32)
        if values.shape!=(64,4) or not torch.isfinite(values).all():return False
        _,_,_,free,_=self.task.get_trajs_collision_and_free(values[None],return_indices=True)
        return bool(free.numel())
def native_world(scene):
    world=make_world(scene)
    return NativeWorld(**{field.name:getattr(world,field.name) for field in fields(world)})

def solve(planner,root,proposal,scene,world,deadline,calls,progress):
    device=proposal.device;synchronize(device);start=time.perf_counter();progress['static_start']=start
    free=native_free(planner.reference_task.tasks[0],proposal)
    invalid=[a for a in range(scene['population']) if a not in free]
    progress.update(static_invalid_agents=invalid,static_repaired_agents=[])
    # Released smoothing runs in native units and is applied after learned
    # pre-smoothing composition. Its endpoint movement is not snapped away.
    batch=proposal.permute(1,0,2).clone();batch[...,2:]*=64/5
    smoothed=smooth_trajs(batch)
    search_root=replace(root,candidate_banks_returned=[o.trajs_final.detach().clone() for o in root.native_outputs],
        selected_indices=list(root.selected_indices))
    wrappers=[NativeLowLevel(low,a,calls,world) for a,low in enumerate(planner.low_level_planner_l)]
    planner.low_level_planner_l=wrappers
    for agent in invalid:
        low=wrappers[agent];low.phase='static';out=None
        experience=PathBatchExperience(search_root.candidate_banks_returned[agent])
        for seed in [experience,None]:
            out=low(planner.start_state_pos_l[agent],planner.goal_state_pos_l[agent],constraints_l=[],experience=seed)
            if out.trajs_final_free_idxs.numel():break
        if not out.trajs_final_free_idxs.numel():
            candidate,attempts=rrt.rrt_static_path(world,agent)
            if candidate is None:raise RuntimeError('Static fail-safe returned no solution')
            candidate=candidate.to(smoothed);candidate[...,2:]*=64/5
            smoothed[agent]=candidate;search_root.candidate_banks_returned[agent]=candidate[None].expand_as(search_root.candidate_banks_returned[agent]).clone()
            search_root.selected_indices[agent]=0
        else:
            search_root.candidate_banks_returned[agent]=out.trajs_final.clone();search_root.selected_indices[agent]=int(out.idx_best_traj)
            smoothed[agent]=out.trajs_final[int(out.idx_best_traj)]
        low.phase='hard'
        progress['static_repaired_agents'].append(agent)
    synchronize(device);static_seconds=time.perf_counter()-start
    # Unchanged existing Akule all-branch expansion and open-list repair loop.
    def expand(self,state):
        progress['ct_expansions_started']=progress.get('ct_expansions_started',0)+1
        repair.expand_all_branches(self,state)
        progress['ct_expansions_completed']=progress.get('ct_expansions_completed',0)+1
    planner.expand=types.MethodType(expand,planner)
    start=time.perf_counter();progress['hard_start']=start;progress['post_static_native']=smoothed.permute(1,0,2)
    state=injected_root(SearchState,planner,list(smoothed.unbind()),search_root)
    paths,ct,status,conflicts,_=repair.repair_from_injected_root(planner,state,max(.001,deadline-time.perf_counter()),TrialSuccessStatus)
    final=torch.stack(paths,1);synchronize(device);hard_seconds=time.perf_counter()-start
    return final,smoothed.permute(1,0,2),{'native_status':status.name,'ct_expansions':ct,
        'static_repaired_agents':invalid,'static_repair_seconds':static_seconds,'hard_repair_seconds':hard_seconds,
        'final_conflicts':conflicts,'root_pre_smoothing_native_static_invalid_agents':len(invalid),
        'native_static_certificate':True,'candidate_bank_reuse':'official diverse native m/s bank, selected slot replaced by coordinated path only'}

def request(engine,proto,scene,directory,mode='dense',cap=60):
    from torch_robotics.torch_utils.seed import fix_random_seed
    from mmd.config.mmd_params import MMDParams
    directory.mkdir(parents=True,exist_ok=True)
    with contextlib.redirect_stdout(NullLog()):planner=configure(proto,scene);world=native_world(scene)
    fix_random_seed(MMDParams.seed)
    def foreign_gpu_pids():
        output=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
        return [int(line.strip()) for line in output.splitlines() if line.strip().isdigit() and int(line.strip())!=os.getpid()]
    foreign=foreign_gpu_pids()
    if foreign:raise RuntimeError(f'GPU unavailable for uncontended production timing: {foreign}')
    calls=[];progress={};timers=GPUComponents();root=proposal=final=post_static=None
    row={'scene_id':scene['configuration_id'],'scene_input_sha256':scene['input_sha256'],
        'family':scene['family'],'N':scene['population'],'mode':mode,'cap_seconds':cap,
        'native_status':'UNKNOWN','implementation_valid':True}
    previous=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(Deadline()))
    synchronize(engine.device);t0=time.perf_counter();signal.setitimer(signal.ITIMER_REAL,cap)
    try:
        with contextlib.redirect_stdout(NullLog()),contextlib.redirect_stderr(NullLog()):
            root=generate(planner)
            with torch.no_grad():proposal,support,coeff=compose(engine,scene,root.selected_pre_smoothing,mode,timers)
            final,post_static,result=solve(planner,root,proposal,scene,world,t0+cap,calls,progress);row.update(result)
    except Deadline:row['native_status']='HARD_TIMEOUT'
    except RuntimeError as exc:
        if str(exc).startswith(('Official MPDEnsemble returned no native-free','Static fail-safe returned no solution')):
            row.update(native_status='FAIL_NO_SOLUTION',failure=str(exc))
        else:
            import traceback
            row.update(native_status='IMPLEMENTATION_ERROR',implementation_valid=False,error=traceback.format_exc())
    except Exception:
        import traceback
        row.update(native_status='IMPLEMENTATION_ERROR',implementation_valid=False,error=traceback.format_exc())
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    synchronize(engine.device);t1=time.perf_counter();gpu=timers.values()
    row['foreign_gpu_pids_at_return']=foreign_gpu_pids()
    row['timing_contaminated']=bool(row['foreign_gpu_pids_at_return'])
    if progress:
        row['static_repair_seconds']=progress.get('hard_start',t1)-progress['static_start']
        row['hard_repair_seconds']=t1-progress['hard_start'] if 'hard_start' in progress else 0.
        row['static_repaired_agents']=progress.get('static_repaired_agents',[])
        row.setdefault('ct_expansions',progress.get('ct_expansions_completed',0))
        row['ct_expansions_started']=progress.get('ct_expansions_started',0)
        row['root_pre_smoothing_native_static_invalid_agents']=len(progress.get('static_invalid_agents',[]))
        if post_static is None:post_static=progress.get('post_static_native')
    hard_low=sum(c['seconds'] for c in calls if c['phase']=='hard')
    row.update(total_compute_seconds=t1-t0,unary_root_seconds=root.wall_seconds if root else None,
        U_seconds=gpu['U'],R_seconds=gpu['R'],G_seconds=gpu['G'],low_level_calls=calls,
        hard_low_level_seconds=hard_low,hard_search_host_seconds=row.get('hard_repair_seconds',0)-hard_low)
    ev=NativeEvaluator(scene['family']);row['native_success_60']=False
    if proposal is not None:
        row['root_pair_time_conflicts'],row['root_unique_conflicted_pairs']=pair_counts(proposal)
        actual=int(support.sum());dense=scene['population']*(scene['population']-1)
        row.update(actual_r_evaluations=actual,dense_r_evaluations=dense,true_r_savings=1-actual/dense,
            u_degree_mean=actual/scene['population'],u_normalized_degree=actual/dense)
    if final is not None:
        physical=final.cpu().numpy().copy();physical[...,2:]*=5/64
        check=ev.check(physical,scene,certificate=row.get('native_static_certificate',False))
        row.update(native_check=check,metrics=metrics(physical,scene))
        row['native_success_60']=row['native_status']=='SUCCESS' and check['native_feasible'] and t1-t0<=cap
    row['stage_D']={}
    arrays={}
    for name,value,is_native in [('unary',root.selected_pre_smoothing if root else None,False),('proposal',proposal,False),('post_static',post_static,True),('final',final,True)]:
        if value is not None:
            physical=value.detach().cpu().numpy().copy()
            if is_native:physical[...,2:]*=5/64
            arrays[name]=physical;row['stage_D'][name]=metrics(physical,scene,env=ev.env)['D']
    begun=time.perf_counter()
    for name,array in arrays.items():np.savez_compressed(directory/(name+'.npz'),physical=array)
    row['serialization_seconds']=time.perf_counter()-begun
    (directory/'RESULT.json').write_text(json.dumps(row,indent=2)+'\n')
    return row
