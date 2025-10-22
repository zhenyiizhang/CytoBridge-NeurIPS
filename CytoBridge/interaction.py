import torch, math

def cal_interaction(z, lnw, interaction_potential, m=16, threshold=1000, use_mass = True):
    w = torch.exp(lnw)
    n = w.shape[0]
    w = w * n
    batch_size, embed_dim = z.shape
    device = z.device
    perm = torch.randperm(batch_size, device=device)
    z_shuffled = z[perm]
    if batch_size % m == 0:
        num_full_groups = batch_size // m
        remainder = batch_size % m
    else:
        if batch_size < m:
            num_full_groups = 0
            remainder = batch_size
        else:
            num_full_groups = batch_size // m - 1
            remainder = batch_size % m + m
    net_force = torch.zeros_like(z)
    if num_full_groups > 0:
        groups_z = z_shuffled[:num_full_groups * m].view(num_full_groups, m, embed_dim)
        idx = torch.triu_indices(m, m, offset=1, device=device)
        num_pairs = idx.shape[1]
        diffs = groups_z[:, idx[0], :] - groups_z[:, idx[1], :]
        mask = (diffs.norm(dim=-1) <= threshold).float().unsqueeze(-1)
        diffs_flat = diffs.reshape(num_full_groups * num_pairs, embed_dim)
        diffs_flat.requires_grad_(True)
        potentials = interaction_potential(diffs_flat)
        grad_potentials = torch.autograd.grad(
            outputs=potentials,
            inputs=diffs_flat,
            grad_outputs=torch.ones_like(potentials),
            create_graph=True,
        )[0]
        f_pairs_flat = -grad_potentials
        f_pairs = f_pairs_flat.view(num_full_groups, num_pairs, embed_dim)
        f_pairs = f_pairs * mask
        mass_groups = w[perm[:num_full_groups * m]].view(num_full_groups, m, 1)
        force_groups = torch.zeros(num_full_groups, m, embed_dim, device=device)
        i_idx = idx[0].unsqueeze(0).expand(num_full_groups, -1)
        j_idx = idx[1].unsqueeze(0).expand(num_full_groups, -1)
        mass_j = mass_groups.gather(1, j_idx.unsqueeze(-1))
        if use_mass:
            contrib_i = f_pairs * mass_j
        else:
            contrib_i = f_pairs
        force_groups.scatter_add_(1, i_idx.unsqueeze(-1).expand(-1, -1, embed_dim), contrib_i)
        mass_i = mass_groups.gather(1, i_idx.unsqueeze(-1))
        if use_mass:
            contrib_j = -f_pairs * mass_i
        else:
            contrib_j = -f_pairs
        force_groups.scatter_add_(1, j_idx.unsqueeze(-1).expand(-1, -1, embed_dim), contrib_j)
        force_groups = force_groups / (m - 1)
        net_force[perm[:num_full_groups * m]] = force_groups.reshape(num_full_groups * m, embed_dim)
    if remainder > 0:
        group_z = z_shuffled[num_full_groups * m:]
        idx_rem = torch.triu_indices(remainder, remainder, offset=1, device=device)
        diffs = group_z[idx_rem[0]] - group_z[idx_rem[1]]
        mask_rem = (diffs.norm(dim=-1) <= threshold).float().unsqueeze(-1)
        diffs.requires_grad_(True)
        potentials = interaction_potential(diffs)
        grad_potentials = torch.autograd.grad(
            outputs=potentials,
            inputs=diffs,
            grad_outputs=torch.ones_like(potentials),
            create_graph=True,
        )[0]
        f_pairs = -grad_potentials
        f_pairs = f_pairs * mask_rem
        mass_group = w[perm[num_full_groups * m:]].view(remainder, 1)
        force_group = torch.zeros(remainder, embed_dim, device=device)
        i_idx_rem = idx_rem[0]
        j_idx_rem = idx_rem[1]
        mass_j = mass_group[j_idx_rem]
        if use_mass:
            contrib_i = f_pairs * mass_j
        else:
            contrib_i = f_pairs
        force_group.scatter_add_(0, i_idx_rem.unsqueeze(-1).expand(-1, embed_dim), contrib_i)
        mass_i = mass_group[i_idx_rem]
        if use_mass:
            contrib_j = -f_pairs * mass_i
        else:
            contrib_j = -f_pairs
        force_group.scatter_add_(0, j_idx_rem.unsqueeze(-1).expand(-1, embed_dim), contrib_j)
        force_group = force_group / (remainder - 1)
        net_force[perm[num_full_groups * m:]] = force_group
    return net_force

def euler_sdeint(sde, initial_state, dt, ts):
    device = initial_state[0].device
    t0 = ts[0].item()
    tf = ts[-1].item()
    current_state = initial_state
    current_time = t0
    output_states = []
    ts_list = ts.tolist()
    next_output_idx = 0
    while current_time <= tf + 1e-8:
        if current_time >= ts_list[next_output_idx] - 1e-8:
            output_states.append(current_state)
            next_output_idx += 1
            if next_output_idx >= len(ts_list):
                break
        t_tensor = torch.tensor([current_time], device=device)
        f_z, f_lnw = sde.f(t_tensor, current_state)
        noise_z = torch.randn_like(current_state[0]) * math.sqrt(dt)
        g_z = sde.g(t_tensor, current_state[0])
        new_z = current_state[0] + f_z * dt + g_z * noise_z
        new_lnw = current_state[1] + f_lnw * dt 
        current_state = (new_z, new_lnw)
        current_time += dt
    while len(output_states) < len(ts_list):
        output_states.append(current_state)
    traj_z = torch.stack([state[0] for state in output_states], dim=0)
    traj_lnw = torch.stack([state[1] for state in output_states], dim=0)
    return traj_z, traj_lnw

def euler_sdeint_split(sde, initial_state, dt, ts, noise_std = 0.01):
    device = initial_state[0].device
    t0 = ts[0].item()
    tf = ts[-1].item()
    current_state = initial_state
    current_time = t0
    output_states = []
    ts_list = ts.tolist()
    next_output_idx = 0
    w_prev = torch.exp(current_state[1])
    while current_time <= tf + 1e-8:
        t_tensor = torch.tensor([current_time], device=device)
        f_z, f_lnw = sde.f(t_tensor, current_state)
        noise_z = torch.randn_like(current_state[0]) * math.sqrt(dt)
        g_z = sde.g(t_tensor, current_state[0])
        new_z = current_state[0] + f_z * dt + g_z * noise_z
        new_lnw = current_state[1] + f_lnw * dt
        current_time += dt
        if current_time >= ts_list[next_output_idx] - 1e-8:
            w_next = torch.exp(new_lnw)
            r = w_next / w_prev
            next_z = []
            next_lnw = []
            for j in range(current_state[0].shape[0]):
                if r[j] >= 1:
                    r_floor = torch.floor(r[j])
                    m_j = int(r_floor) + (1 if torch.rand(1, device=device) < (r[j] - r_floor) else 0)
                    for _ in range(m_j):
                        noise = torch.normal(0, noise_std, size=new_z[j].shape, device=device)
                        perturbed_x = new_z[j] + noise
                        next_z.append(perturbed_x.unsqueeze(0))
                        next_lnw.append(new_lnw[j].unsqueeze(0))
                else:
                    if torch.rand(1, device=device) < r[j]:
                        next_z.append(new_z[j].unsqueeze(0))
                        next_lnw.append(new_lnw[j].unsqueeze(0))
            if next_z:
                new_z = torch.cat(next_z, dim=0)
                new_lnw = torch.log(torch.ones(new_z.shape[0], 1) / initial_state[0].shape[0]).to(device)
            else:
                new_z = torch.empty(0, current_state[0].shape[1], device=device)
                new_lnw = torch.empty(0, 1, device=device)
            current_state = (new_z, new_lnw)
            output_states.append(current_state)
            next_output_idx += 1
            w_prev = torch.exp(new_lnw)
            if next_output_idx >= len(ts_list):
                break
        else:
            current_state = (new_z, new_lnw)
    while len(output_states) < len(ts_list):
        output_states.append(current_state)
    traj_z = [state[0] for state in output_states]
    traj_lnw = [state[1] for state in output_states]
    return traj_z, traj_lnw