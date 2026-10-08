import hashlib
import json

def model_scene(sample, frame):
    scene = dict(sample, starts=frame.positions_to_model(sample["starts"]).tolist(),
                 goals=frame.positions_to_model(sample["goals"]).tolist(),
                 rollout_seed=sample["noise_seed"],
                 obstacles=dict(kind="circles", items=[]),
                 workspace=[[-1., -1.], [1., 1.]])
    scene["input_sha256"] = hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest()
    return scene
