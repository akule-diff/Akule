"""Clean computation boundary and map-prior-aligned existing Akule repair."""
from __future__ import annotations
import copy
import contextlib
import json
import signal
import time
import types
import numpy as np
import torch
from mmd_mainpaper_common import NativeEvaluator,metrics
from mmd_zero_shot_planner import (RepairBackend,LowLevel,make_world,repair,rrt,static_repair,rollout)
from diffuser.models.quality_dynamic_u_v2 import full_support
from diffuser.utils.quality_sparse_v1 import endpoint_condition

class NullLog:
    def write(self,s):return len(s)
    def flush(self):pass
class Deadline(BaseException):pass
def sync():torch.cuda.synchronize()

class GPUComponents:
    """Exclusive stream event windows; no per-step device synchronization."""
    def __init__(self):self.events={k:[] for k in ['unary','U','R','G']}
    @contextlib.contextmanager
    def section(self,name):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record()
        try:yield
        finally:b.record();self.events[name].append((a,b))
    def values(self):return {k:sum(a.elapsed_time(b)/1000 for a,b in pairs) for k,pairs in self.events.items()}

@torch.inference_mode()
def timed_rollout(engine,scene,timers):
    with timers.section('unary'):
        hard,ends,endpoints=engine.conditions(scene);saved=engine.noise(scene)
        x=engine.unary.apply_hard_conditions(saved['initial'].clone(),hard)
    gates=[]
    for step in reversed(range(25)):
        t=torch.tensor([step],device=x.device)
        with timers.section('unary'):base,c1,c2=engine.base_with_grad(x,t,hard)
        with timers.section('U'):
            valid=full_support(base);support=(engine.u.logits(base,endpoints,t)>0)*valid
            index=support.nonzero();gates.append(support)
        with timers.section('R'):fields=engine.all_fields_with_grad(base,endpoints,t,index)
        with timers.section('G'):composed,_=engine.signed.compose(base,endpoints,t,index,fields,support.to(base))
        with timers.section('unary'):
            mean=engine.unary.apply_hard_conditions(c1*engine.codec.encode(composed)+c2*x,hard)
            noise=saved['posterior'][24-step].clone()
            if step==0:noise.zero_()
            x=engine.posterior_fixed(x,mean,t,hard,noise)
    with timers.section('unary'):output=endpoint_condition(engine.codec.decode(x),ends)
    return {'output':output,'gates':torch.stack(gates)}

class VelocityAdapter:
    """Official low-level m/s <-> Akule displacement; no position change."""
    def __init__(self,native):self.native=native
    def __getattr__(self,name):return getattr(self.native,name)
    def __call__(self,*args,**kwargs):
        experience=kwargs.get('experience')
        if experience is not None:
            experience=copy.copy(experience)
            key='path_b' if hasattr(experience,'path_b') else 'path'
            q=getattr(experience,key).clone();q[...,2:]/=5/64
            setattr(experience,key,q);kwargs=dict(kwargs,experience=experience)
        output=self.native(*args,**kwargs)
        for key in ['trajs_iters','trajs_final','trajs_final_coll','trajs_final_free','traj_final_free_best']:
            q=getattr(output,key,None)
            if isinstance(q,torch.Tensor) and q.numel() and q.shape[-1]==4:
                q=q.clone();q[...,2:]*=5/64;setattr(output,key,q)
        return output

class CleanLowLevel(LowLevel):
    def __call__(self,*args,**kwargs):
        r={'agent_id':self.agent,'constraints':len(kwargs.get('constraints_l') or []),'warm_start':kwargs.get('experience') is not None}
        self.backend.calls.append(r);begun=time.perf_counter()
        out=self.native(*args,**kwargs);r['mpd_seconds']=time.perf_counter()-begun
        static_repair.filter_native_output(out,self.backend.world,self.agent,constraints=kwargs.get('constraints_l'))
        if r['constraints'] and not len(out.trajs_final_free_idxs):
            begun=time.perf_counter()
            candidate,attempts=rrt.rrt_static_path(self.backend.world,self.agent,constraints=kwargs.get('constraints_l'),max_time_s=1.,seed_offsets=(37,0,73,149))
            r.update(rrt_fallback_seconds=time.perf_counter()-begun,rrt_fallback_attempts=attempts,rrt_fallback_accepted=candidate is not None)
            if candidate is not None:
                candidate=candidate.to(out.trajs_final);out.trajs_final=candidate[None].expand_as(out.trajs_final).clone()
                out.trajs_final_free_idxs=torch.zeros(1,device=candidate.device,dtype=torch.long)
                out.trajs_final_free=out.trajs_final[:1];out.traj_final_free_best=out.trajs_final[0];out.idx_best_traj=0
                out.success_free_trajs=True;out.fraction_free_trajs=1/len(out.trajs_final)
        return out

class CleanRepair(RepairBackend):
    def __init__(self,prototype,aligned):
        self.prototype=prototype;self.device=prototype.tensor_args['device'];self.aligned=aligned;self.progress=None
    def configure(self,scene,world):
        from mmd.planners.multi_agent import CBS
        from mmd.common.conflicts import PointConflict
        from mmd.common.constraints import MultiPointConstraint
        self.world=world;self.calls=[];self.root_rrt=[];self.ct_expansions=0;self.post_static=None;self.last_root=None
        self.pre=None;self.static_replaced_agents=None
        self.static_seconds=0.;self.hard_seconds=0.
        proto=self.prototype
        if not self.aligned:
            from mmd_zero_shot_planner import install_geometry
            install_geometry(proto.task,world,tensor_args=proto.tensor_args)
        shared=[*proto.models.values(),*proto.datasets,proto.task,proto.robot]
        bank=[copy.deepcopy(proto,{id(x):x for x in shared}) for _ in scene['starts']]
        starts=torch.tensor(scene['starts'],device=self.device);goals=torch.tensor(scene['goals'],device=self.device)
        for a,low in enumerate(bank):
            hard=low.datasets[0].get_single_pt_hard_conditions(starts[a],0,True)
            hard.update(low.datasets[0].get_single_pt_hard_conditions(goals[a],-1,True))
            low.hard_conds={0:hard};low.start_state_pos=starts[a].clone();low.goal_state_pos=goals[a].clone()
        wrappers=[CleanLowLevel(VelocityAdapter(low) if self.aligned else low,a,self) for a,low in enumerate(bank)]
        planner=CBS(wrappers,list(starts.unbind()),list(goals.unbind()),start_time_l=[0]*len(bank),
            is_xcbs=True,is_ecbs=True,conflict_type_to_constraint_types={PointConflict:{MultiPointConstraint}},reference_robot=proto.robot,reference_task=proto.task)
        def expand(planner,state):repair.expand_all_branches(planner,state);self.ct_expansions+=1
        planner.expand=types.MethodType(expand,planner);self.prepared=(planner,starts,goals)
        return self.prepared
    def solve(self,scene,proposal,world,deadline):
        from mmd.planners.multi_agent.cbs import SearchState
        from mmd.common.experiences import PathBatchExperience
        from mmd.common.experiments import TrialSuccessStatus
        planner,starts,goals=self.prepared
        sync();begun=time.perf_counter()
        try:
            physical,paths,audit=repair.prepare_manifest_joint_paths(proposal,starts,goals)
            pre=world.joint_audit(physical[0].cpu().numpy())
            self.pre=pre
            def fallback(a):
                candidate,attempts=rrt.rrt_static_path(world,a)
                self.root_rrt.append({'agent':a,'attempts':attempts,'accepted':candidate is not None})
                return None if candidate is None else candidate.to(self.device)
            paths,replaced=static_repair.repair_root_with_native_low_level(planner,paths,world,PathBatchExperience,static_fallback=fallback)
            self.static_replaced_agents=replaced
            root=torch.stack(paths,1);self.last_root=root;post=world.joint_audit(root.cpu().numpy());self.post_static=post
        finally:sync();self.static_seconds=time.perf_counter()-begun
        begun=time.perf_counter()
        try:
            state=repair.injected_root(SearchState,planner,paths,starts,goals)
            final,ct,status,conflicts,_=repair.repair_from_injected_root(planner,state,max(.001,deadline-time.perf_counter()),TrialSuccessStatus,
                accept_state=lambda p:world.joint_audit(torch.stack(p,1).cpu().numpy())['success'])
            final=torch.stack(final,1)
        finally:sync();self.hard_seconds=time.perf_counter()-begun
        return final,root,{'native_status':status.name,'ct_expansions':ct,'final_native_conflicts':conflicts,'static_replaced_agents':replaced,
            'root_pair_conflicts':pre['pair_conflicts'],'root_static_invalid_agents':pre['static_invalid_agents'],
            'post_static_pair_conflicts':post['pair_conflicts'],'post_static_invalid_agents':post['static_invalid_agents']}

def request(engine,backend,scene,directory,manifest_sha,cap=60):
    directory.mkdir(parents=True,exist_ok=True);world=make_world(scene)
    with contextlib.redirect_stdout(NullLog()),contextlib.redirect_stderr(NullLog()):backend.configure(scene,world)
    torch.manual_seed(scene['noise_seed']%(2**32));np.random.seed(scene['noise_seed']%(2**32))
    timers=GPUComponents();proposal=None;final=None;root=None;gates=None
    r={'scene_id':scene['configuration_id'],'family':scene['family'],'N':scene['population'],'trial':scene['trial'],
       'scene_input_sha256':scene['input_sha256'],'manifest_sha256':manifest_sha,'repair_aligned':backend.aligned,
       'native_status':'UNKNOWN','implementation_valid':True,'hard_limit_seconds':cap}
    def stop(*_):raise Deadline()
    previous=signal.signal(signal.SIGALRM,stop)
    sync();t0=time.perf_counter();signal.setitimer(signal.ITIMER_REAL,cap)
    try:
        with contextlib.redirect_stdout(NullLog()),contextlib.redirect_stderr(NullLog()):
            output=timed_rollout(engine,scene,timers);proposal=output['output'];gates=output['gates']
            final,root,result=backend.solve(scene,proposal,world,t0+cap);r.update(result)
    except Deadline:r.update(native_status='HARD_TIMEOUT',timeout=True)
    except static_repair.StaticRepairNoSolution as exc:r.update(native_status='FAIL_STATIC_REPAIR',failure=str(exc))
    except Exception:
        import traceback
        r.update(native_status='IMPLEMENTATION_ERROR',implementation_valid=False,failure=traceback.format_exc())
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    sync();t1=time.perf_counter();values=timers.values()
    r['component_times']={'unary_seconds':values['unary'],'U_seconds':values['U'],'R_seconds':values['R'],'G_seconds':values['G'],
        'interaction_seconds':sum(values[k] for k in ['U','R','G']),'static_repair_seconds':backend.static_seconds,
        'hard_repair_seconds':backend.hard_seconds,'total_compute_seconds':t1-t0,
        'gpu_component_semantics':'exclusive CUDA event seconds; repair and total are synchronized wall seconds; host overhead is in total'}
    r.update(total_compute_seconds=t1-t0,total_planning_seconds=t1-t0,low_level_calls=len(backend.calls),low_level_call_records=backend.calls,
        ct_expansions=backend.ct_expansions,root_rrt_fallbacks=len(backend.root_rrt),root_rrt_records=backend.root_rrt,
        child_rrt_fallbacks=sum('rrt_fallback_seconds' in x for x in backend.calls))
    if backend.pre is not None:
        r.update(root_pair_conflicts=backend.pre['pair_conflicts'],root_static_invalid_agents=backend.pre['static_invalid_agents'])
    if backend.post_static is not None:
        r.update(post_static_pair_conflicts=backend.post_static['pair_conflicts'],post_static_invalid_agents=backend.post_static['static_invalid_agents'])
    r['static_replaced_agents']=backend.static_replaced_agents
    if gates is not None:
        actual=int(gates.sum());dense=25*scene['population']*(scene['population']-1)
        r.update(actual_r_evaluations=actual,dense_r_evaluations=dense,true_r_savings=1-actual/dense,u_degree_mean=float(gates.sum(-1).float().mean()))
        r['u_normalized_degree']=r['u_degree_mean']/(scene['population']-1)
    begun=time.perf_counter();ev=NativeEvaluator(scene['family']);r['metrics']=None;r['native_check']=None
    if final is not None:
        q=final.cpu().numpy();r['native_check']=ev.check(q,scene);r['metrics']=metrics(q,scene,env=ev.env)
    r['native_success_60']=bool(r['native_status']=='SUCCESS' and r['native_check'] and r['native_check']['native_feasible'] and t1-t0<=cap)
    r['stage_D']={}
    root=root if root is not None else backend.last_root
    for name,value in [('pre_repair',proposal[0] if proposal is not None else None),('post_static',root),('final',final)]:
        if value is not None:r['stage_D'][name]=metrics(value.cpu().numpy(),scene,env=ev.env)['D']
    r['external_evaluation_seconds']=time.perf_counter()-begun
    begun=time.perf_counter()
    for name,value in [('proposal',proposal[0] if proposal is not None else None),('root',root),('final',final)]:
        if value is not None:np.savez_compressed(directory/(name+'.npz'),physical=value.cpu().numpy())
    (directory/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
    r['serialization_seconds']=time.perf_counter()-begun
    (directory/'SERIALIZATION_SECONDS.json').write_text(json.dumps({'serialization_seconds':r['serialization_seconds'],'scope':'trajectory artifacts and result JSON; excludes this accounting sidecar'})+'\n')
    return r
