"""Vectorized official independent MPDEnsemble; no inter-robot constraints.

Only the sample axis and hard endpoint conditions are expanded. Released
diffusion/guidance, native filtering, per-robot stats/least-cost selection,
normalizer and smoothing are reused. Candidate banks remain native m/s.
"""
import copy
import ast
import inspect
import textwrap
import time
from types import SimpleNamespace,MethodType
import torch
from mmd_official_root_runtime import OfficialRoot,synchronize
from mmd.common import smooth_trajs

ROBOTS_PER_CHUNK=20


def accept_batched_conditions(model):
    """Adapt only released endpoint expansion; retain its inference body."""
    original=inspect.unwrap(type(model).run_inference)
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    function=tree.body[0]
    function.decorator_list=[]
    replaced=0
    for node in ast.walk(function):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
            if ast.unparse(node.value).startswith('einops.repeat(v,'):
                node.value=ast.parse('v.clone() if v.ndim == 2 else einops.repeat(v, "d -> b d", b=n_samples)',mode='eval').body
                replaced+=1
    assert replaced==1
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<official-batched-endpoints>','exec'),namespace)
    model.run_inference=MethodType(torch.no_grad()(namespace[function.name]),model)


def finish_robot(low,normalized_chain):
    ensemble={}
    for key,chain in normalized_chain.items():
        values=low.task.get_traj_unnormalized(key,low.datasets,chain)
        ensemble[key]=low.task.get_stats(key,*values,0.,save_data=False)
    r=low.task.combine_trajs(ensemble)
    out=SimpleNamespace(**r)
    out.trajs_final_free_idxs=r['trajs_final_free_idxs']
    if not out.trajs_final_free_idxs.numel():
        raise RuntimeError('Batched official unary: no native-free candidate for a robot')
    out.idx_best_traj=int(r['idx_best_traj'])
    out.trajs_final=smooth_trajs(r['trajs_iters'][-1])
    return out


def generate_batched(planner,robots_per_chunk=ROBOTS_PER_CHUNK):
    start=time.perf_counter();outputs=[]
    count=planner.num_agents
    for offset in range(0,count,robots_per_chunk):
        ids=list(range(offset,min(count,offset+robots_per_chunk)))
        first=getattr(planner.low_level_planner_l[ids[0]],'low',planner.low_level_planner_l[ids[0]])
        shared=[*first.models.values(),*first.datasets,first.task,first.robot]
        low=copy.deepcopy(first,{id(x):x for x in shared})
        accept_batched_conditions(low.model)
        assert low.num_samples==64 and len(low.models)==1
        assert not low.cross_conds
        low.num_samples=64*len(ids)
        low.hard_conds={key:{t:torch.cat([planner.low_level_planner_l[a].hard_conds[key][t].reshape(1,-1).expand(64,-1)
                          for a in ids],dim=0) for t in first.hard_conds[key]} for key in first.hard_conds}
        chains,_,_=low.run_constrained_inference([])
        for slot in range(len(ids)):
            output=finish_robot(low,{key:value[:,slot*64:(slot+1)*64] for key,value in chains.items()})
            outputs.append(output)
    synchronize(planner.tensor_args['device'])
    pre=torch.stack([o.trajs_iters[-1,int(o.idx_best_traj)] for o in outputs],1).clone()
    returned=torch.stack([o.trajs_final[int(o.idx_best_traj)] for o in outputs],1).clone()
    banks=[o.trajs_final.clone() for o in outputs]
    for value in (pre,returned,*banks):value[...,2:]*=5/64
    return OfficialRoot(pre,returned,banks,outputs,[int(o.idx_best_traj) for o in outputs],
                        [o.trajs_final_free_idxs.clone() for o in outputs],time.perf_counter()-start)
