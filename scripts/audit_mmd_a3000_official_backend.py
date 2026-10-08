"""Three-arm validation-only official xECBS causal audit. No training/test.

Each arm pays for the same released prioritized root construction. Repair-only
time is nested after root generation/coordination/cleanup, so it excludes them
without granting free unary generation in complete Plan.
"""
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import types
import statistics

import numpy as np
import torch
from mmd_prior_adaptation_common import ROOT, setup, bind, write
from mmd_paired_common import configure, sha
from mmd_corrected_akule_runtime import compose, native_free
from mmd_official_root_runtime import synchronize
from mmd_official_search_equivalence import bind_official_plan, PROVENANCE
from mmd_final_clean_runtime import NullLog, Deadline, GPUComponents
from mmd_mainpaper_common import NativeEvaluator, metrics
from mmd.common import smooth_trajs
from mmd.common.experiences import PathBatchExperience
from mmd.config.mmd_params import MMDParams
from torch_robotics.torch_utils.seed import fix_random_seed

STUDY = ROOT/'results/mmd_n20_shared_g_adaptation_20261006'
OUT = STUDY/'OFFICIAL_BACKEND_EQUIVALENCE'
PANEL = STUDY/'G_COMPLETE_PARETO_VALIDATION/REDUCED_VALIDATION_PANEL.json'
G = STUDY/'G_TRAINING/G_a_003000.pt'
ARMS = ('MMD_ROOT', 'AKULE_DENSE', 'AKULE_SPARSE')


def foreign():
    x = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                       capture_output=True,text=True,check=True).stdout
    return [int(s) for s in x.splitlines() if s.strip().isdigit() and int(s)!=os.getpid()]


class Calls:
    """Delegate to exact MPDEnsemble; record seed reuse, never add fallback."""
    def __init__(self, low, agent, audit):
        self.low=low; self.agent=agent; self.audit=audit
    def __getattr__(self,name): return getattr(self.low,name)
    def __call__(self,*args,**kw):
        rec={'agent':self.agent,'phase':self.audit['phase'],
             'constraints':len(kw.get('constraints_l') or []),
             'bank_seed_hit':kw.get('experience') is not None,
             'fresh_from_noise':kw.get('experience') is None,
             'diffusion_call':True}
        started=time.perf_counter()
        try:
            out=self.low(*args,**kw)
            rec['free_candidates']=int(out.trajs_final_free_idxs.numel())
            if self.audit['phase']=='root':
                self.audit['pre'][self.agent]=out.trajs_iters[-1,int(out.idx_best_traj)].detach().clone() if out.idx_best_traj is not None else None
            return out
        finally:
            rec['seconds']=time.perf_counter()-started
            self.audit['calls'].append(rec)


class OpenList(list):
    def __init__(self,audit):super().__init__();self.audit=audit
    def append(self,value):
        self.audit['generated_nodes']+=1
        super().append(value)


def bank_stats(banks, originals, indices, arm):
    result=[]
    for a,bank in enumerate(banks):
        b=bank.detach().cpu().contiguous();original=originals[a].detach().cpu()
        mask=torch.ones(64,dtype=torch.bool);mask[indices[a]]=False
        equal=torch.equal(b[mask],original[mask])
        result.append({'agent':a,'shape':list(b.shape),
            'distinct_candidates':len(torch.unique(b.reshape(64,-1),dim=0)),
            'original_distinct_candidates':len(torch.unique(original.reshape(64,-1),dim=0)),
            'nonselected_63_bitwise_preserved':equal,
            'selected_index':int(indices[a]),
            'original_bank_sha256':hashlib.sha256(original.numpy().tobytes()).hexdigest()})
    assert all(x['shape']==[64,64,4] and x['nonselected_63_bitwise_preserved'] for x in result)
    assert all(x['distinct_candidates']==64 for x in result), 'Candidate collapse entering official search'
    return result


def run(engine,proto,scene,arm,directory,*,test_used=False,unary_mode='prioritized'):
    directory.mkdir(parents=True,exist_ok=True)
    saved=directory/'RESULT.json'
    if saved.exists(): return json.loads(saved.read_text())
    assert not foreign(), 'Contended GPU before request'
    with contextlib.redirect_stdout(NullLog()):planner=bind_official_plan(configure(proto,scene))
    audit={'phase':'root','pre':{},'calls':[],'conflict_seconds':0.,'conflict_calls':0,
           'ct_started':0,'ct_completed':0,'generated_nodes':0}
    timers=GPUComponents();arrays={};bank_in=[];originals=[];indices=[]
    planner.low_level_planner_l=[Calls(l,a,audit) for a,l in enumerate(planner.low_level_planner_l)]
    original_get=planner.get_conflicts
    def get_conflicts(state):
        begin=time.perf_counter();answer=original_get(state)
        audit['conflict_seconds']+=time.perf_counter()-begin;audit['conflict_calls']+=1
        if 'root_conflicts' not in audit:
            audit['root_conflicts']=len(answer)
            audit['root_unique_pairs']=len({tuple(sorted(c.agent_ids)) for c in answer})
        return answer
    planner.get_conflicts=get_conflicts
    original_expand=planner.expand
    def expand(state):
        audit['ct_started']+=1
        original_expand(state)
        audit['ct_completed']+=1
    planner.expand=expand
    planner.open_l=OpenList(audit)
    def transform(planner,root):
        synchronize(engine.device);audit['unary_seconds']=time.perf_counter()-t0
        originals.extend([b.detach().clone() for b in root.path_bl])
        indices.extend([int(i) for i in root.ix_best_path_in_batch_l])
        native=torch.stack([audit['pre'][a] for a in range(scene['population'])],1)
        physical=native.clone();physical[...,2:]*=5/64
        arrays['unary']=physical
        if arm!='MMD_ROOT':
            with torch.no_grad():proposal,support,_=compose(engine,scene,physical,
                'dense' if arm=='AKULE_DENSE' else 'sparse',timers)
            audit['actual_r_evaluations']=int(support.sum())
            arrays['proposal']=proposal
            synchronize(engine.device);st=time.perf_counter();audit['phase']='static'
            free=native_free(planner.reference_task.tasks[0],proposal)
            invalid=[a for a in range(scene['population']) if a not in free]
            paths=proposal.permute(1,0,2).clone();paths[...,2:]*=64/5
            paths=smooth_trajs(paths)
            for a in invalid:
                # Official bank-seeded local planner only. No RRT, retry,
                # empty-prior switch or duplicate-bank replacement.
                out=planner.low_level_planner_l[a](planner.start_state_pos_l[a],
                    planner.goal_state_pos_l[a],constraints_l=[],
                    experience=PathBatchExperience(root.path_bl[a]))
                if not out.trajs_final_free_idxs.numel():
                    raise RuntimeError('Native targeted cleanup returned no solution')
                paths[a]=out.trajs_final[int(out.idx_best_traj)]
            audit['static_agents']=invalid
            synchronize(engine.device);audit['static_seconds']=time.perf_counter()-st
            for a,path in enumerate(paths):root.path_bl[a]=root.path_bl[a].clone();root.path_bl[a][indices[a]]=path
            arrays['post_static_native']=paths.permute(1,0,2)
        else:
            audit['static_agents']=[];audit['static_seconds']=0.
        bank_in.extend([b.detach().clone() for b in root.path_bl])
        audit['phase']='search'
        synchronize(engine.device);audit['repair_begin']=time.perf_counter()
        return root
    fix_random_seed(MMDParams.seed)
    previous=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(Deadline()))
    synchronize(engine.device);t0=time.perf_counter();signal.setitimer(signal.ITIMER_REAL,60)
    paths=[];status='UNKNOWN';ct=0;conflicts=None
    try:
        with contextlib.redirect_stdout(NullLog()),contextlib.redirect_stderr(NullLog()):
            initial=None
            if unary_mode=='batched_independent':
                from mmd_batched_independent_unary import generate_batched
                from mmd.planners.multi_agent.cbs import SearchState
                generated=generate_batched(planner)
                for a,o in enumerate(generated.native_outputs):audit['pre'][a]=o.trajs_iters[-1,int(o.idx_best_traj)].detach().clone()
                initial=SearchState(list(generated.selected_indices),[o.trajs_final.clone() for o in generated.native_outputs],constraints={})
            paths,ct,state,conflicts=planner.plan(runtime_limit=60,initial_root=initial,root_transform=transform)
            status=state.name
    except Deadline:status='HARD_TIMEOUT'
    except RuntimeError as exc:
        if str(exc)=='Native targeted cleanup returned no solution':status='FAIL_STATIC_CLEANUP'
        elif str(exc).startswith('Batched official unary: no native-free'):status='FAIL_NO_SOLUTION'
        else:raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    synchronize(engine.device);t1=time.perf_counter()
    assert not foreign(), 'Contended GPU at return; retain attempt and report'
    # All external diagnostics and artifact writing occur after t1.
    banks=bank_stats(bank_in,originals,indices,arm) if bank_in else []
    ev=NativeEvaluator(scene['family']);final=None;success=False;metric=None
    if len(paths)==scene['population']:
        final=torch.stack(paths,1).detach().cpu().numpy();final[...,2:]*=5/64
        native_check=ev.check(final,scene,certificate=status=='SUCCESS')
        success=status=='SUCCESS' and native_check['native_feasible'] and t1-t0<=60
        metric=metrics(final,scene)
    else:native_check=None
    calls=audit['calls'];search_calls=[c for c in calls if c['phase']=='search']
    repair_seconds=t1-audit['repair_begin'] if audit.get('repair_begin') else None
    low=sum(c['seconds'] for c in search_calls)
    row={'scene_id':scene['configuration_id'],'scene_input_sha256':scene['input_sha256'],
        'family':scene['family'],'N':scene['population'],'arm':arm,'native_status':status,'native_success_60':success,
        'native_check':native_check,'metrics':metric,'complete_plan_seconds':t1-t0,
        'repair_only_seconds':repair_seconds,'unary_seconds':audit.get('unary_seconds'),
        **timers.values(),'static_seconds':audit.get('static_seconds'),
        'static_repaired_agents':audit.get('static_agents',[]),
        'root_conflicts':audit.get('root_conflicts'),'root_unique_pairs':audit.get('root_unique_pairs'),
        'ct_expansions':ct if status!='HARD_TIMEOUT' else audit['ct_completed'],
        'ct_expansions_started':audit['ct_started'],'generated_CT_nodes':max(0,audit['generated_nodes']-1),
        'search_low_level_calls':len(search_calls),
        'bank_seed_hits':sum(c['bank_seed_hit'] for c in search_calls),
        'bank_seed_misses':sum(not c['bank_seed_hit'] for c in search_calls),
        'fresh_constrained_from_noise_calls':sum(c['fresh_from_noise'] for c in search_calls),
        'constrained_local_diffusion_calls':sum(c['bank_seed_hit'] for c in search_calls),
        'pure_cache_hits':0,'low_level_seconds':low,
        'mean_low_level_call_seconds':low/len(search_calls) if search_calls else 0.,
        'conflict_detection_seconds':audit['conflict_seconds'],
        'other_search_host_seconds':repair_seconds-low-audit['conflict_seconds'] if repair_seconds else None,
        'calls':calls,'bank_audit':banks,'stage_D':{},'test_used':test_used,'unary_mode':unary_mode,
        'search_source_provenance':{k:v for k,v in PROVENANCE.items() if k!='root_hook_source'}}
    for stage,value in arrays.items():
        q=value.detach().cpu().numpy().copy()
        if stage.endswith('_native'):q[...,2:]*=5/64
        row['stage_D'][stage]=metrics(q,scene)['D']
        arrays[stage]=q
    if final is not None:arrays['final']=final
    if 'actual_r_evaluations' in audit:
        actual=audit['actual_r_evaluations'];dense=scene['population']*(scene['population']-1)
        row.update(actual_r_evaluations=actual,dense_r_evaluations=dense,
            true_r_savings=1-actual/dense,u_degree_mean=actual/scene['population'],
            u_normalized_degree=actual/dense)
    started=time.perf_counter()
    for name,value in arrays.items():np.savez_compressed(directory/(name+'.npz'),physical=value)
    row['serialization_seconds']=time.perf_counter()-started
    write(saved,row)
    return row


def summarize(rows):
    groups={}
    for family in ('highways','conveyor'):
        groups[family]={}
        for arm in ARMS:
            r=[x for x in rows if x['family']==family and x['arm']==arm]
            if not r:continue
            good=[x for x in r if x['native_success_60']]
            avg=lambda key:statistics.mean(x[key] for x in r if x.get(key) is not None) if any(x.get(key) is not None for x in r) else None
            groups[family][arm]={'scenes':len(r),'success':len(good),
                **{k:avg(k) for k in ['root_conflicts','root_unique_pairs','ct_expansions','generated_CT_nodes',
                'search_low_level_calls','bank_seed_hits','bank_seed_misses','fresh_constrained_from_noise_calls',
                'constrained_local_diffusion_calls','mean_low_level_call_seconds','low_level_seconds',
                'conflict_detection_seconds','other_search_host_seconds','repair_only_seconds',
                'complete_plan_seconds','unary_seconds','U','R','G','static_seconds','serialization_seconds']},
                'successful_Plan':statistics.mean(x['complete_plan_seconds'] for x in good) if good else None,
                'D':statistics.mean(x['metrics']['D'] for x in good) if good else None,
                'distinct_candidates_min':min([b['distinct_candidates'] for x in r for b in x['bank_audit']],default=None)}
    write(OUT/'RESULTS.json',{'complete':len(rows)==60,'groups':groups,'raw':rows,'test_used':False,'training':False})
    lines=['# Matched official-backend N20 validation','','All arms use the same official search, prioritized unary root and candidate banks. Repair-only is the nested timer after root preparation; complete Plan pays for it. Search counts are means over all requests. Plan/D in this table use native-successful requests.','','| Map | Arm | Root conflicts | CT | LL calls | Bank seeds | Fresh from noise | Local diffusion | Repair s | Plan s | Success | D |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'N/A' if x is None else f'{x:.3f}'
    for f,v in groups.items():
        for arm,g in v.items():lines.append('| '+f+' | '+arm+' | '+' | '.join(fmt(g[k]) for k in ['root_conflicts','ct_expansions','search_low_level_calls','bank_seed_hits','fresh_constrained_from_noise_calls','constrained_local_diffusion_calls','repair_only_seconds','successful_Plan'])+f" | {g['success']}/{g['scenes']} | {fmt(g['D'])} |")
    (OUT/'MATCHED_TABLE.md').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    scenes=json.loads(PANEL.read_text())['scenes'];assert len(scenes)==20 and all(s['population']==20 for s in scenes)
    write(OUT/'CONTRACT.json',{'panel_sha256':sha(PANEL),'G_sha256':sha(G),'scenes':scenes,
        'arms':ARMS,'cap':60,'training':False,'test_used':False,
        'root_generation':'Official CBS.plan prioritized soft-constraint root, same seed/banks in every arm.',
        'search_and_lowlevel':'Official CBS.plan search and CBS.expand; exact MPDEnsemble without RRT/retry overrides.',
        'bank_hit_definition':'Experience bank supplied as local-diffusion seed. No cache-only acceptance exists in released MPDEnsemble. Every constrained seed hit still invokes local diffusion.',
        'static_cleanup':'Only native-invalid agents; official bank-seeded MPDEnsemble with empty constraints; remaining 63 original bank slots preserved.',
        'timing':'Synchronized complete perf_counter includes paid unary and cleanup; nested repair-only begins after root preparation and before official cost/conflicts/search.',
        'provenance':PROVENANCE})
    engine,priors,protos,_=setup();engine.signed.g.load_state_dict(torch.load(G,map_location=engine.device,weights_only=False)['g'])
    for f in ('highways','conveyor'):
        scene=next(s for s in scenes if s['family']==f)
        with contextlib.redirect_stdout(NullLog()),contextlib.redirect_stderr(NullLog()):
            warm=configure(protos[f],scene)
            warm.low_level_planner_l[0](warm.start_state_pos_l[0],warm.goal_state_pos_l[0],constraints_l=None,experience=None)
            bank=warm.low_level_planner_l[0].recent_call_data.trajs_final
            warm.low_level_planner_l[0](warm.start_state_pos_l[0],warm.goal_state_pos_l[0],constraints_l=[],experience=PathBatchExperience(bank))
            bind(engine,priors,scene)
            cached=ROOT/'results/mmd_prior_interaction_adaptation_20261006/operational_unary'/scene['configuration_id']/'official_unary.npz'
            q=torch.as_tensor(np.load(cached)['pre_smoothing_physical'],device=engine.device)
            with torch.no_grad():compose(engine,scene,q,'dense');compose(engine,scene,q,'sparse')
    rows=[]
    (OUT/'STATUS').write_text('RUNNING_VALIDATION_ONLY_NO_TRAINING_NO_TEST\n')
    for scene in scenes:
        for arm in ARMS:
            bind(engine,priors,scene)
            row=run(engine,protos[scene['family']],scene,arm,OUT/'SCENES'/arm/scene['configuration_id'])
            rows.append(row);summarize(rows)
            print(json.dumps({'completed':len(rows),'total':60,'arm':arm,'map':scene['family'],
                'status':row['native_status'],'seconds':row['complete_plan_seconds']}),flush=True)
        # Exact bank identity in causal arms, established after timing.
        cell=rows[-3:]
        hashes=[[b['original_bank_sha256'] for b in r['bank_audit']] for r in cell]
        if all(hashes):assert hashes[0]==hashes[1]==hashes[2], 'Root-generation bank mismatch across causal arms'
    (OUT/'STATUS').write_text('COMPLETE_BACKEND_EQUIVALENCE_VALIDATION_REPORT_GATE_NO_TRAINING_NO_TEST\n')


if __name__=='__main__':main()
