import torch

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)


@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True, state_grad_mask=False, 
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)
            
        if scale_grad_by_std:
            grad = model_var * grad

        if state_grad_mask:
            grad[:, :, model.action_dim:] = 0.0
        grad[t < t_stopgrad] = 0

        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y, torch.zeros_like(y)

@torch.no_grad()
def n_step_guided_p_sample_decouple(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True, state_grad_mask=False, policy_step=True
):
    x_action, x_state = x[:, :, :model.action_dim], x[:, :, model.action_dim:]
    x_part = x_action if policy_step else x_state
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x_part.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    if policy_step:
        for _ in range(n_guide_steps):
            with torch.enable_grad():
                y, grad = guide.gradients(x, cond, t, policy_step=True)

            if scale_grad_by_std:
                grad = model_var * grad

            if state_grad_mask:
                grad[:, :, model.action_dim:] = 0.0
            grad[t < t_stopgrad] = 0

            x = x + scale * grad
            x = apply_conditioning(x, cond, model.action_dim)
    else:
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t, policy_step=False)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t, policy_step=policy_step)

    # no noise when t == 0
    noise = torch.randn_like(x_part)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y, torch.zeros_like(y)

@torch.no_grad()
def n_step_guided_and_constrained_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True, cost_guide = None, cost_threshold = 0, cost_grad_weight=1.0, state_grad_mask=False
):
    assert cost_guide is not None
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for k in range(n_guide_steps):
        
        with torch.enable_grad():
            cost_y, cost_grad = cost_guide.gradients(x, cond, t)

        with torch.enable_grad():
            r_y, r_grad = guide.gradients(x, cond, t)
        
        #grad = r_grad - (cost_y.unsqueeze(-1).unsqueeze(-1).expand_as(cost_grad) > cost_threshold) * cost_grad * cost_grad_weight

        #step_ratio = (model.n_timesteps - t[0]) / model.n_timesteps
        # c_weight = (cost_y-cost_threshold) / cost_threshold
        # c_weight = cost_grad_weight * c_weight.clamp(0, 1.0).exp().unsqueeze(-1).unsqueeze(-1).expand_as(r_grad)
        # grad = r_grad - (cost_grad*c_weight*(cost_y>cost_threshold).unsqueeze(-1).unsqueeze(-1).expand_as(r_grad))

        c_weight = cost_grad_weight * (cost_y>cost_threshold).unsqueeze(-1).unsqueeze(-1).expand_as(r_grad)
        grad = r_grad - (cost_grad*c_weight)

        #grad = torch.where(cost_y.unsqueeze(-1).unsqueeze(-1).expand_as(r_grad) > cost_threshold, -cost_grad*c_weight, r_grad)
        #grad = r_grad
        #print(scale, torch.sum(cost_y>cost_threshold).cpu().item(), t[0].cpu().item())
        #print(cost_y)
        #print(torch.sum(cost_y>cost_threshold).cpu().item(), end=" " if t[0].cpu().item()>0 else "\n")
        #print("t=", t[0].cpu().item(), torch.sum(cost_y>cost_threshold).cpu().item(), cost_y.mean().cpu().item(), cost_y[:5].detach().cpu().numpy())

        if scale_grad_by_std:
            grad = model_var * grad

        if state_grad_mask:
            grad[:, :, model.action_dim:] = 0.0
        grad[t < t_stopgrad] = 0

        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

        # if cost_y.mean() < cost_threshold and k>=1:
        #     break

    print(torch.sum(cost_y>cost_threshold).cpu().item(), end=" " if t[0].cpu().item()>0 else "\n\n")

    if torch.sum(cost_y<=cost_threshold).cpu().item()==0: #if all trajectory break the constraint, select one with minimum cost 
        r_y = -cost_y                                           
    else:
        r_y = torch.where(cost_y > cost_threshold, r_y-1e8, r_y)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, r_y, cost_y
