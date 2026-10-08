"""Run one frozen scene with the benchmark's original inference pathway."""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=['weave', 'basic', 'dense', 'shelf', 'room', 'highways', 'conveyor', 'scaledweave'], default='weave')
    parser.add_argument('--n', type=int)
    parser.add_argument('--mode', choices=['unary', 'sparse', 'dense'], default='sparse')
    parser.add_argument('--scene', type=Path, help='Saved scene JSON; overrides --n')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--repair', action='store_true', help='Run the benchmark-specific downstream planner (CUDA)')
    parser.add_argument('--output', type=Path, default=Path('outputs/demo'))
    args = parser.parse_args()
    import numpy as np
    import torch
    import canonical_n28_runtime as runtime
    from canonical_n28_rollout import rollout
    n = args.n or (28 if args.env == 'weave' else (40 if args.env == 'scaledweave' else 3))
    path = (args.scene or ROOT / f'benchmarks/scenes/{args.env}_n{n}.json').resolve()
    if not path.is_file():
        parser.error(f'Saved scene unavailable: {path.name}; supply --scene or choose a released population')
    scene = json.loads(path.read_text())
    n = len(scene['starts'])
    scene.setdefault('population', n)
    scene.setdefault('parent_geometry_id', scene.get('layout_id', scene.get('geometry_hash', scene['input_sha256'])))
    scene.setdefault('rollout_seed', int(scene.get('noise_seed', 0)) % (2**63-1))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.json').exists():
        parser.error('Output already contains a run; choose a fresh --output directory')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable. SMD proposal inference also supports --device cpu.')
    if args.repair and args.device != 'cuda':
        parser.error('The frozen downstream planner requires CUDA')
    maps = args.env in ('highways', 'conveyor')
    if args.env in ('weave', 'scaledweave'):
        if args.device != 'cuda': parser.error('The frozen Weave runtime requires CUDA')
        engine = runtime.Engine()
        if args.env == 'scaledweave':
            if args.repair: parser.error('The released ScaledWeave runtime is proposal-only')
            from mmd_structured_bridge import load_base, NativeVelocityCodec, limits_from_official_trajectories
            from diffuser.models.mpd_v2 import MPDUnaryAdapter
            from scaled_weave_frame import ScaledWeaveFrame
            from akule.scaled_binding import model_scene
            diffusion, _, _ = load_base('nowait', ROOT / 'external/mmd/data_trained_models', device=args.device)
            engine.unary = MPDUnaryAdapter(diffusion).eval().requires_grad_(False)
            engine.codec = NativeVelocityCodec(limits_from_official_trajectories('nowait'), args.device, native_velocity_is_mps=False)
            engine.signed.g.load_state_dict(torch.load(ROOT / 'checkpoints/scaledweave/G.pt', map_location=args.device, weights_only=True)['g'])
            physical_scene = scene
            frame = ScaledWeaveFrame(n)
            scene = model_scene(scene, frame)
    else:
        from mmd_zero_shot_planner import load_final_engine
        engine, unary, _ = load_final_engine(args.device)
        if maps:
            from mmd_paired_common import prototype
            from mmd_structured_bridge import NativeVelocityCodec
            from diffuser.models.mpd_v2 import MPDUnaryAdapter
            from mmd_final_clean_runtime import NullLog
            with contextlib.redirect_stdout(NullLog()):
                proto = prototype(args.env, args.device, out / 'setup')
            diffusion = getattr(proto.models[0], '_orig_mod', proto.models[0])
            normal = proto.datasets[0].normalizer.normalizers['traj']
            engine.unary = MPDUnaryAdapter(diffusion).eval().requires_grad_(False)
            engine.codec = NativeVelocityCodec({'minimum': normal.mins.tolist(), 'maximum': normal.maxs.tolist()}, args.device, native_velocity_is_mps=True)
            for name, model in [('G', engine.signed.g), ('U', engine.u)]:
                model.load_state_dict(torch.load(ROOT / f'checkpoints/maps/{name}.pt', map_location=args.device, weights_only=True)[name.lower()])
        else:
            from akule.smd_binding import bind_final_unary
            bind_final_unary(engine, unary, scene)
    engine.population = n
    engine.split = {'train': [scene]}
    def sync():
        if args.device == 'cuda': torch.cuda.synchronize()
    sync(); begun = time.perf_counter()
    record = {'environment': args.env, 'population': n, 'mode': args.mode,
              'scene_id': scene['configuration_id'], 'input_sha256': scene['input_sha256'],
              'stage': 'complete_planning' if args.repair else 'proposal',
              'pathway': 'selected_independent_candidates_t0' if maps else 'full_reverse_diffusion'}
    if maps and args.repair:
        from audit_mmd_a3000_official_backend import run
        arm = {'unary': 'MMD_ROOT', 'sparse': 'AKULE_SPARSE', 'dense': 'AKULE_DENSE'}[args.mode]
        record['planner'] = run(engine, proto, scene, arm, out / 'repair', test_used=True, unary_mode='batched_independent')
        record['success'] = record['planner']['native_success_60']
    else:
        if maps:
            from mmd_paired_common import configure
            from mmd_batched_independent_unary import generate_batched
            from mmd_corrected_akule_runtime import compose
            from mmd.config.mmd_params import MMDParams
            from torch_robotics.torch_utils.seed import fix_random_seed
            fix_random_seed(MMDParams.seed)
            root = generate_batched(configure(proto, scene))
            physical = root.selected_pre_smoothing
            if args.mode != 'unary':
                with torch.no_grad(): physical, _, _ = compose(engine, scene, physical, args.mode)
            physical = physical[None]
        else:
            with torch.no_grad(): physical = rollout(engine, scene, args.mode)['output']
        sync(); record['proposal_seconds'] = time.perf_counter()-begun
        values = physical.detach().cpu().numpy()
        if args.env == 'scaledweave':
            from scaled_weave_validate import validate
            values = frame.grouped_to_physical(values)
            scene = physical_scene
            record['scaled_validation'] = validate(values[0], scene)
        if values.shape != (1,64,n,4) or not np.isfinite(values).all():
            raise RuntimeError('Invalid generated trajectory')
        np.savez_compressed(out / 'proposal.npz', physical=values, input_sha256=scene['input_sha256'])
        if args.env != 'scaledweave':
            record['sampled_metrics'] = runtime.q.metrics(values, scene['starts'], scene['goals'])
            if maps:
                from mmd_mainpaper_common import NativeEvaluator
                record['native_proposal_check'] = NativeEvaluator(args.env).check(values[0], scene)
            elif args.env != 'weave':
                from common_strict_safety import smd_native_success
                record['native_proposal_collision_free'] = smd_native_success(values[0], scene['obstacles'])
        if args.repair:
            import importlib.util
            if args.env == 'weave':
                import canonical_mmd_repair as backend
            else:
                backend = runtime.load('akule_smd_repair', ROOT / 'integrations/smd_runtime/scripts/canonical_mmd_repair.py')
                from smd_repair_endpoint_adapter import bind_smd_root_admission
                bind_smd_root_admission()
                engine.native_instance_name = 'EnvEmpty2DRobotPlanarDiskRandom'
                engine.native_model_id = 'EnvEmptyNoWait2D-RobotPlanarDisk'
            engine.setup_seconds = 0.
            engine.mixer = type('FrozenG', (), {'checkpoint_path': 'checkpoints/G.pt' if args.env=='weave' else 'checkpoints/smd/g.pt', 'checkpoint_sha256': ''})()
            engine.u.checkpoint_path = 'checkpoints/U.pt' if args.env=='weave' else 'checkpoints/smd/u.pt'
            engine.u.checkpoint_sha256 = ''
            engine.rollout = lambda _scene, _method, **kw: (physical, [], {'proposal_reused_without_recomputation': True})
            runtime.OUT = out
            (out / 'logs').mkdir(exist_ok=True)
            backend.CAP = backend.NATIVE_CAP = backend.WATCHDOG = max(.01, (240 if args.env=='weave' else 900)-record['proposal_seconds'])
            native = backend.Native(engine, backend='XECBS')
            if args.env != 'weave':
                from smd_population_endpoint_adapter import bind_smd_population_endpoint_precheck
                bind_smd_population_endpoint_precheck(native, scene)
            repair_scene = dict(scene, noise_seed=int(scene.get('noise_seed',scene['rollout_seed']))%(2**32))
            record['planner'] = native.request(repair_scene, args.mode, out / 'repair')
            record['success'] = bool(record['planner'].get('smd_native_success',record['planner'].get('complete_success',False)))
    sync(); record['wall_seconds_including_repair_setup'] = time.perf_counter()-begun
    (out / 'scene.json').write_text(json.dumps(scene, indent=2)+'\n')
    (out / 'metrics.json').write_text(json.dumps(runtime.ready(record), indent=2)+'\n')
    print(json.dumps(runtime.ready(record), indent=2))


if __name__ == '__main__': main()
