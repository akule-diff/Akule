import torch
from diffuser.models.smd_topological_mpd import SpatialMapField
from diffuser.models.mpd_v2 import grouped_to_mpd_per_agent, mpd_per_agent_to_grouped

def bind_final_unary(engine, model, scene):
    field = SpatialMapField(scene["obstacles"], device=engine.device, clearance_scale=.2)
    starts = torch.as_tensor(scene["starts"], device=engine.device, dtype=torch.float32)
    goals = torch.as_tensor(scene["goals"], device=engine.device, dtype=torch.float32)
    encoded = model.encode_map(field, starts, goals)

    def mapped_base(x, timestep, hard):
        batch, _, agents, _ = x.shape
        flat = grouped_to_mpd_per_agent(x)
        eps_flat = model(flat, timestep.repeat_interleave(agents),
                         model.context_from_state(flat, field, encoded))
        eps = mpd_per_agent_to_grouped(eps_flat, batch, agents)
        mean = engine.unary.reverse_mean(x, eps, timestep, hard)
        c1 = engine.unary.diffusion.posterior_mean_coef1[timestep][:, None, None, None]
        c2 = engine.unary.diffusion.posterior_mean_coef2[timestep][:, None, None, None]
        return engine.codec.decode(engine.unary.apply_hard_conditions((mean-c2*x)/c1, hard)), c1, c2

    engine.base_with_grad = mapped_base
