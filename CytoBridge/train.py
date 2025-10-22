__all__ = ['train']

import os, sys, json, math, itertools
import pandas as pd, numpy as np
import warnings

# from tqdm import tqdm
from tqdm.notebook import tqdm

import torch

from .utils import sample, generate_steps
from .losses import MMD_loss, OT_loss1, OT_loss2, Density_loss, Local_density_loss
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint2
from CytoBridge.models import velocityNet, growthNet, scoreNet, dediffusionNet, indediffusionNet, FNet, ODEFunc2, ODEFunc3, FNet_interaction, ODEFunc2_interaction, ODEFunc_interaction, ODEFunc2_interaction_energy
from CytoBridge.utils import group_extract, sample, to_np, generate_steps, cal_mass_loss, cal_mass_loss_reduce, parser, _valid_criterions
import geomloss
from geomloss import SamplesLoss
from CytoBridge.interaction import cal_interaction

import matplotlib.pyplot as plt 


def train_un1_reduce(
    model, df, groups, optimizer, n_batches=20, 
    criterion=MMD_loss(),
    use_cuda=False,
    sample_size=(100, ),
    sample_with_replacement=False,
    local_loss=True,
    global_loss=False,
    hold_one_out=False,
    hold_out='random',
    apply_losses_in_time=True,
    top_k = 5,
    hinge_value = 0.01,
    use_density_loss=False,
    use_local_density=False,
    lambda_density = 1.0,
    autoencoder=None, 
    use_emb=False,
    use_gae=False,
    use_gaussian:bool=False, 
    add_noise:bool=False, 
    noise_scale:float=0.1,
    device=None,
    logger=None,
    use_pinn=False,
    use_penalty=True,
    lambda_energy=0.01,
    lambda_pinn=1,
    lambda_ot=0.1,
    lambda_mass=1,
    relative_mass=None,
    initial_size=None,
    reverse:bool = False,
    best_model_path=None,
    pca_transform=None,
    use_mass = True,
    global_mass = False,
):
    steps = generate_steps(groups)

    if reverse:
        groups = groups[::-1]
        steps = generate_steps(groups)

    batch_losses = []
    globe_losses = []
    if hold_one_out and hold_out in groups:
        groups_ho = [g for g in groups if g != hold_out]
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho) if hold_out not in [t0, t1]}
    else:
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
        
    density_fn = Density_loss(hinge_value)

    if use_cuda:
        model = model.cuda()
    
    model.train()
    model.to(device)
    step=0
    print('begin local loss')
    min_ot_loss = float('inf')

    for batch in tqdm(range(n_batches)):
        if local_loss and not global_loss:
            i_mass=1
            lnw0 = torch.log(torch.ones(sample_size[0],1) / (sample_size[0])).to(device)
            data_t0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
            data_t0.to(device)

            batch_loss = []
            if hold_one_out:
                groups = [g for g in groups if g != hold_out]
                steps = generate_steps(groups)
            for step_idx, (t0, t1) in enumerate(steps):  
                if hold_out in [t0, t1] and hold_one_out:
                    continue                              
                optimizer.zero_grad()
                
                data_t0.to(device)
                size1=(df[df['samples']==t1].values.shape[0],)
                data_t1 = sample(df, t1, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
                data_t1.to(device)
                time = torch.Tensor([t0, t1]).cuda() if use_cuda else torch.Tensor([t0, t1])
                time.to(device)
                if add_noise:
                    data_t0 += noise(data_t0) * noise_scale
                    data_t1 += noise(data_t1) * noise_scale
                if autoencoder is not None and use_gae:
                    data_t0 = autoencoder.encoder(data_t0)
                    data_t1 = autoencoder.encoder(data_t1)

                if autoencoder is not None and use_emb:        
                    data_tp, data_t1 = autoencoder.encoder(data_tp), autoencoder.encoder(data_t1)

                relative_mass_now = relative_mass[i_mass]
                m0 = torch.zeros_like(lnw0).to(device)
                data_t0=data_t0.to(device)
                data_t1=data_t1.to(device)
                initial_state_energy = (data_t0, lnw0, m0)
                t=time.to(device)
                t.requires_grad=True
                data_t0.requires_grad=True
                lnw0.requires_grad=True
                
                x_t, lnw_t, m_t=odeint(ODEFunc2(model, use_mass = use_mass),initial_state_energy,t,options=dict(step_size=0.1),method='euler')
                lnw_t_last = lnw_t[-1]
                m_t_last = m_t[-1]
                if use_mass:
                    mu = torch.exp(lnw_t_last)
                else:
                    mu = torch.ones(data_t0.shape[0],1)
                mu = mu / mu.sum()
                nu = torch.ones(data_t1.shape[0],1)
                nu = nu / nu.sum()
                mu = mu.squeeze(1)
                nu=nu.squeeze(1)
                loss_ot = criterion(x_t[-1], data_t1, mu, nu)
                i_mass=i_mass+1
                global_mass_loss = torch.norm(torch.sum(torch.exp(lnw_t_last)) - relative_mass_now, p=2)**2
                local_mass_loss = cal_mass_loss_reduce(data_t1, x_t[-1], lnw_t_last, relative_mass_now, sample_size[0],dim_reducer=pca_transform)
                if global_mass:
                    mass_loss = local_mass_loss + global_mass_loss
                else:
                    mass_loss = local_mass_loss
                lnw0=lnw_t_last.detach()
                data_t0=x_t[-1].detach()
            
                print('Otloss')
                print(loss_ot)
                print('mass loss')
                print(mass_loss)
                print('energy loss')
                print(m_t_last.mean())
                loss=(lambda_ot*loss_ot+lambda_mass*mass_loss + lambda_energy * m_t_last.mean())

                if use_density_loss:                
                    density_loss = density_fn(data_t0, data_t1, top_k=top_k)
                    density_loss = density_loss.to(loss.device)
                    loss += lambda_density * density_loss
                    print('density loss')
                    print(density_loss)

                if apply_losses_in_time and local_loss:
                    loss.backward()
                    optimizer.step()
                    model.norm=[]
                local_losses[f'{t0}:{t1}'].append(loss.item())
                batch_loss.append(loss)
            current_ot_loss = loss_ot.item()
            
            if current_ot_loss < min_ot_loss:
                min_ot_loss = current_ot_loss
                torch.save(model.state_dict(), best_model_path)
                print(f'New minimum otloss found: {min_ot_loss}. Model saved.')
        
            batch_loss = torch.Tensor(batch_loss).float()
            if use_cuda:
                batch_loss = batch_loss.cuda()
            
            if not apply_losses_in_time:
                batch_loss.backward()
                optimizer.step()

            ave_local_loss = torch.mean(batch_loss)
            sum_local_loss = torch.sum(batch_loss)            
            batch_losses.append(ave_local_loss.item())
                     
    print_loss = globe_losses if global_loss else batch_losses 
    if logger is None:      
        tqdm.write(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    else:
        logger.info(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    return local_losses, batch_losses, globe_losses


def train_un1_reduce_interaction(
    model, df, groups, optimizer, n_batches=20, 
    criterion=MMD_loss(),
    use_cuda=False,
    sample_size=(100, ),
    sample_with_replacement=False,
    local_loss=True,
    global_loss=False,
    hold_one_out=False,
    hold_out='random',
    apply_losses_in_time=True,
    top_k = 5,
    hinge_value = 0.01,
    use_density_loss=False,
    use_local_density=False,
    lambda_density = 1.0,
    autoencoder=None, 
    use_emb=False,
    use_gae=False,
    use_gaussian:bool=False, 
    add_noise:bool=False, 
    noise_scale:float=0.1,
    device=None,
    logger=None,
    use_pinn=False,
    use_penalty=True,
    lambda_energy=0.01,
    lambda_pinn=1,
    lambda_ot=0.1,
    lambda_mass=1,
    relative_mass=None,
    initial_size=None,
    reverse:bool = False,
    best_model_path=None,
    pca_transform=None,
    use_mass = True,
    num_rbf = 8,
    thre = 0.5,
    num_particles = 16,
    use_spatial = False,
):
    steps = generate_steps(groups)

    if reverse:
        groups = groups[::-1]
        steps = generate_steps(groups)

    batch_losses = []
    globe_losses = []
    if hold_one_out and hold_out in groups:
        groups_ho = [g for g in groups if g != hold_out]
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho) if hold_out not in [t0, t1]}
    else:
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
        
    density_fn = Density_loss(hinge_value)

    if use_cuda:
        model = model.cuda()
    
    model.train()
    model.to(device)
    
    step=0
    print('begin local loss')
    min_ot_loss = float('inf')

    for batch in tqdm(range(n_batches)):
        
        if local_loss and not global_loss:
            i_mass=1
            lnw0 = torch.log(torch.ones(sample_size[0],1) / (sample_size[0])).to(device)
            data_t0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)

            batch_loss = []
            if hold_one_out:
                groups = [g for g in groups if g != hold_out]
                steps = generate_steps(groups)
                
            for step_idx, (t0, t1) in enumerate(steps):
                data_t0.to(device)
                initial_energy = torch.zeros(sample_size[0],1).to(device)

                if hold_out in [t0, t1] and hold_one_out:
                    continue
                optimizer.zero_grad()
                
                size1=(df[df['samples']==t1].values.shape[0],)
                data_t1 = sample(df, t1, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
                m0 = (torch.zeros(sample_size[0],1) / (initial_size)).to(device)  
                data_t1.to(device)
                time = torch.Tensor([t0, t1]).cuda() if use_cuda else torch.Tensor([t0, t1])
                time.to(device)
                if add_noise:
                    data_t0 += noise(data_t0) * noise_scale
                    data_t1 += noise(data_t1) * noise_scale
                if autoencoder is not None and use_gae:
                    data_t0 = autoencoder.encoder(data_t0)
                    data_t1 = autoencoder.encoder(data_t1)

                if autoencoder is not None and use_emb:        
                    data_tp, data_t1 = autoencoder.encoder(data_tp), autoencoder.encoder(data_t1)

                relative_mass_now = relative_mass[i_mass]
                data_t0=data_t0.to(device)
                data_t1=data_t1.to(device)

                initial_state_energy = (data_t0, lnw0, initial_energy)
                t=time.to(device)
                t.requires_grad=True
                data_t0.requires_grad=True
                lnw0.requires_grad=True
                
                x_t, lnw_t, energy=odeint2(ODEFunc2_interaction_energy(model, use_mass, thre, num_particles),initial_state_energy,t,options=dict(step_size=0.1),method='euler')
                lnw_t_last = lnw_t[-1]
                energy_last = energy[-1]
                if use_mass:
                    mu = torch.exp(lnw_t_last)
                else:
                    mu = torch.ones(data_t0.shape[0],1)
                mu = mu / mu.sum()
                nu = torch.ones(data_t1.shape[0],1)
                nu = nu / nu.sum()
                mu = mu.squeeze(1)
                nu=nu.squeeze(1)
                loss_ot = criterion(x_t[-1], data_t1, mu, nu)
                i_mass=i_mass+1
                mass_loss = cal_mass_loss_reduce(data_t1, x_t[-1], lnw_t_last, relative_mass_now, sample_size[0],dim_reducer=pca_transform)
                lnw0=lnw_t_last.detach()
                data_t0=x_t[-1].detach()
                
                print(f'Otloss {loss_ot.item()}')
                print(f'mass loss {mass_loss.item()}')
                print(f"energy loss {energy_last.mean().item()}")
                loss=(lambda_ot*loss_ot + lambda_mass*mass_loss + lambda_energy * energy_last.mean())

                if use_density_loss:                
                    density_loss = density_fn(data_t0, data_t1, top_k=top_k)
                    density_loss = density_loss.to(loss.device)
                    loss += lambda_density * density_loss
                    print('density loss')
                    print(density_loss)

                if apply_losses_in_time and local_loss:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    optimizer.step()
                    model.norm=[]
                local_losses[f'{t0}:{t1}'].append(loss.item())
                batch_loss.append(loss)
            current_ot_loss = loss_ot.item()

            if current_ot_loss < min_ot_loss:
                min_ot_loss = current_ot_loss
                torch.save(model.state_dict(), best_model_path)
                print(f'New minimum otloss found: {min_ot_loss}. Model saved.')
        
            batch_loss = torch.Tensor(batch_loss).float()
            if use_cuda:
                batch_loss = batch_loss.cuda()
            
            if not apply_losses_in_time:
                batch_loss.backward()
                optimizer.step()

            ave_local_loss = torch.mean(batch_loss)
            sum_local_loss = torch.sum(batch_loss)            
            batch_losses.append(ave_local_loss.item())
                     
    print_loss = globe_losses if global_loss else batch_losses 
    if logger is None:      
        tqdm.write(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    else:
        logger.info(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    return local_losses, batch_losses, globe_losses

from .utils import density1, trace_df_dz
import torch, torch.nn as nn

def train_all(
    model, df, groups, optimizer, n_batches=20, 
    criterion=MMD_loss(),
    use_cuda=False,

    sample_size=(100, ),
    sample_with_replacement=False,

    local_loss=True,
    global_loss=False,

    hold_one_out=False,
    hold_out='random',
    apply_losses_in_time=True,

    top_k = 5,
    hinge_value = 0.01,
    use_density_loss=False,
    use_local_density=False,

    lambda_density = 1.0,
    datatime0=None,
    device=None,
    autoencoder=None, 
    use_emb=False,
    use_gae=False,

    use_gaussian:bool=False, 
    add_noise:bool=False, 
    noise_scale:float=0.1,
    
    logger=None,
    use_pinn=False,
    sf2m_score_model=None,
    use_penalty=True,
    lambda_energy=0.01,
    lambda_pinn=1,
    lambda_ot=10,
    relative_mass=None,
    initial_size=None,
    reverse:bool = False,
    sigmaa=0.1,
    lambda_initial=None,
    lambda_mass = 1,
    use_mass = True,
    thre = 0.5,
    exp_dir = None,
    scheduler = None,
    log_mass = False,
    global_mass = True,
    num_particles = 16,
    sf2m_optimizer = None,
    use_spatial = False,
):

    def compute_integral(x, time, V, rho_net, num_samples=50, sigma=0.1, sigmaa=1.0):
        batch_size, dim = x.shape
        noise = torch.randn(batch_size, num_samples, dim, device=x.device)
        y = sigma * noise
        perterbed_x = x.unsqueeze(1) + sigma * noise
        perterbed_x_flat = perterbed_x.reshape(-1, dim)
        time_repeated = time.repeat(num_samples, 1)
        ss = rho_net(time_repeated, perterbed_x_flat)
        rrho = torch.exp(ss * 2 / (sigmaa ** 2))
        rrho = rrho.reshape(batch_size, num_samples, 1)
        y_flat = y.reshape(-1, dim)
        y_flat.requires_grad_(True)

        v = V(y_flat)
        if v.dim() > 1:
            v = v.squeeze(-1)
        grad_y = torch.autograd.grad(v.sum(), y_flat, create_graph=False)[0]
        F = -grad_y
        F = F.view(batch_size, num_samples, dim)
        norm_constant = (2 * math.pi) ** (dim / 2) * (sigma ** dim)
        q = torch.exp(-0.5 * (noise ** 2).sum(dim=-1)) / norm_constant
        contributions = (rrho * F) / q.unsqueeze(-1)
        integral_estimate = contributions.mean(dim=1)
        return integral_estimate

    if autoencoder is None and (use_emb or use_gae):
        use_emb = False
        use_gae = False
        warnings.warn('\'autoencoder\' is \'None\', but \'use_emb\' or \'use_gae\' is True, both will be set to False.')

    noise_fn = torch.randn if use_gaussian else torch.rand
    def noise(data):
        return noise_fn(*data.shape).cuda() if use_cuda else noise_fn(*data.shape)

    steps = generate_steps(groups)

    if reverse:
        groups = groups[::-1]
        steps = generate_steps(groups)

    batch_losses = []
    globe_losses = []
    if hold_one_out and hold_out in groups:
        groups_ho = [g for g in groups if g != hold_out]
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho) if hold_out not in [t0, t1]}
    else:
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
        
    density_fn = Density_loss(hinge_value)

    if use_cuda:
        model = model.cuda()
    
    model.train()
    step=0
    print('begin local loss')

    min_ot_loss = float('inf')

    for batch in tqdm(range(n_batches)):

        if local_loss and not global_loss:
            size0=(df[df['samples']==0].values.shape[0],)
            data_0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
            P0=0
            num=data_0.shape[0]
            time=torch.tensor([0.0]).to(device)
            time=time.expand(num,1)
            data_0=data_0.to(device)
            sf2m_score_model=sf2m_score_model.to(device)
            s2=sf2m_score_model(time, data_0)
            data_0.requires_grad_(True)

            i_mass=1
            lnw0 = torch.log(torch.ones(sample_size[0],1) / (sample_size[0])).to(device)
            m0 = (torch.zeros(sample_size[0],1) / (initial_size)).to(device)  
            data_t0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)

            batch_loss = []
            if hold_one_out:
                groups = [g for g in groups if g != hold_out]
                steps = generate_steps(groups)
            for step_idx, (t0, t1) in enumerate(steps):  
                if hold_out in [t0, t1] and hold_one_out:
                    continue                              
                optimizer.zero_grad()
                sf2m_optimizer.zero_grad()
                size1=(df[df['samples']==t1].values.shape[0],)
                data_t1 = sample(df, t1, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
                time = torch.Tensor([t0, t1]).cuda() if use_cuda else torch.Tensor([t0, t1])

                if add_noise:
                    data_t0 += noise(data_t0) * noise_scale
                    data_t1 += noise(data_t1) * noise_scale
                if autoencoder is not None and use_gae:
                    data_t0 = autoencoder.encoder(data_t0)
                    data_t1 = autoencoder.encoder(data_t1)

                if autoencoder is not None and use_emb:        
                    data_tp, data_t1 = autoencoder.encoder(data_tp), autoencoder.encoder(data_t1)

                relative_mass_now = relative_mass[i_mass]
                data_t0=data_t0.to(device)
                data_t1=data_t1.to(device)
                lnw0=lnw0.to(device)
                m0=m0.to(device)
                model=model.to(device)
                initial_state_energy = (data_t0, lnw0,m0)
                t=time.to(device)
                t.requires_grad=True
                data_t0.requires_grad=True
                lnw0.requires_grad=True
                m0.requires_grad=True
                x_t, lnw_t,m_t=odeint2(ODEFunc3(model,sf2m_score_model,sigmaa, use_mass, thre, num_particles),initial_state_energy,t,options=dict(step_size=0.1),method='euler')
                lnw_t_last = lnw_t[-1]
                m_t_last=m_t[-1]
                if use_mass:
                    mu = torch.exp(lnw_t_last)
                else:
                    mu = torch.ones(data_t0.shape[0],1, device = data_t0.device)
                mu = mu / mu.sum()
                nu = torch.ones(data_t1.shape[0],1, device = data_t0.device)
                nu = nu / nu.sum()
                mu = mu.squeeze(1)
                nu=nu.squeeze(1)
                if use_spatial:
                    loss_ot = criterion(x_t[-1], data_t1, mu, nu)
                else:
                    loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.1)
                    loss_ot = loss_fn(mu, x_t[-1],nu, data_t1,)
                i_mass=i_mass+1
                global_mass_loss = torch.norm(torch.sum(torch.exp(lnw_t_last)) - relative_mass_now, p=2)**2
                local_mass_loss = cal_mass_loss(data_t1, x_t[-1], lnw_t_last, relative_mass_now, sample_size[0])
                if global_mass:
                    mass_loss = global_mass_loss + local_mass_loss
                else:
                    mass_loss = local_mass_loss
                m0=m_t_last.clone().detach()
                lnw0=lnw_t_last.clone().detach()
                data_t0=x_t[-1].clone().detach()
            
                print(f'Otloss {loss_ot.item()}')
                print(f'mass loss {mass_loss.item()}')
                print(f'energy loss {m_t_last.mean().item()}')
                loss_ot=loss_ot.to(device)
                loss=(lambda_ot*loss_ot+lambda_mass*mass_loss+m_t_last.mean()* lambda_energy)
                print(f"total loss {loss}")

                if use_density_loss:                
                    density_loss = density_fn(data_t0, data_t1, top_k=top_k)
                    density_loss = density_loss.to(loss.device)
                    loss += lambda_density * density_loss
                    print('density loss')
                    print(density_loss)

                if use_pinn and batch % 50 == 0:
                    P1=0
                    nnum=data_t1.shape[0]
                    ttime=time[1]
                    vv, gg, _, _ = model(ttime, data_t1)
                    ttime=ttime.expand(nnum,1)
                    ss=sf2m_score_model(ttime, data_t1)
                    rrho = torch.exp(ss*2/sigmaa**2) 
                    rrho_t = torch.autograd.grad(outputs=rrho, inputs=ttime, grad_outputs=torch.ones_like(rrho),create_graph=True)[0]
                    vv_rho = vv * rrho
                    ddiv_v_rho = trace_df_dz(vv_rho, data_t1).unsqueeze(1)
                    if model.interaction_net.cutoff != 0:
                        interaction_estimate = compute_integral(data_t1, ttime, model.interaction_net, sf2m_score_model,sigmaa=sigmaa)
                        rho_interaction_estimate = interaction_estimate * rrho
                        div_interaction = trace_df_dz(rho_interaction_estimate, data_t1).unsqueeze(1)

                        if use_mass:
                            ppinn_loss = torch.abs(rrho_t + ddiv_v_rho - gg * rrho + div_interaction)
                        else:
                            ppinn_loss = torch.abs(rrho_t + ddiv_v_rho + div_interaction)
                    else:
                        if use_mass:
                            ppinn_loss = torch.abs(rrho_t + ddiv_v_rho - gg * rrho)
                        else:
                            ppinn_loss = torch.abs(rrho_t + ddiv_v_rho)
                    pppinn_loss = ppinn_loss

                    mean_pppinn_loss = torch.mean(pppinn_loss)

                    size0=(df[df['samples']==0].values.shape[0],)
                    data_0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
                    P0=0
                    num=data_0.shape[0]
                    time=torch.tensor([0.0]).to(device)
                    time=time.expand(num,1)
                    data_0=data_0.to(device)
                    sf2m_score_model=sf2m_score_model.to(device)
                    s2=sf2m_score_model(time, data_0)
                    data_0.requires_grad_(True)
                    density_values = density1(data_0,datatime0,device)
                    loss2=0
                    loss2=torch.mean((torch.exp(s2*2/sigmaa**2)-density_values)**2)
  
                    print('pinloss')
                    print(mean_pppinn_loss+loss2)
                    
                    loss += lambda_pinn * mean_pppinn_loss + lambda_initial*loss2

                if apply_losses_in_time and local_loss:
                    loss.backward()
                    optimizer.step()
                    if batch % 50 == 0:
                        sf2m_optimizer.step()
                    if scheduler != None:
                        scheduler.step()
                    model.norm=[]
                local_losses[f'{t0}:{t1}'].append(loss.item())
                batch_loss.append(loss)
            
            current_ot_loss = loss_ot.item()
            
            if current_ot_loss < min_ot_loss and exp_dir != None:
                min_ot_loss = current_ot_loss
                torch.save(sf2m_score_model.state_dict(), os.path.join(exp_dir, 'score_model_result'))
                torch.save(model.state_dict(), os.path.join(exp_dir, 'model_result'))
                print(f'New minimum otloss found: {min_ot_loss}. Model saved.')
        
            batch_loss = torch.Tensor(batch_loss).float()
            if use_cuda:
                batch_loss = batch_loss.cuda()
            
            if not apply_losses_in_time:
                torch.mean(batch_loss).backward()
                optimizer.step()
                if scheduler != None:
                    scheduler.step()

            ave_local_loss = torch.mean(batch_loss)
            sum_local_loss = torch.sum(batch_loss)            
            batch_losses.append(ave_local_loss.item())
                     
    print_loss = globe_losses if global_loss else batch_losses 
    if logger is None:      
        tqdm.write(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    else:
        logger.info(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    return local_losses, batch_losses, globe_losses

