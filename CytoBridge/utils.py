__all__ = ['group_extract', 'sample', 'to_np', 'generate_steps', 'set_seeds', 'config_hold_out', 'config_criterion',
           'get_groups_from_df', 'get_cell_types_from_df', 'get_sample_n_from_df', 'get_times_from_groups']

import numpy as np, pandas as pd
import torch
import random

def group_extract(df, group, index='samples', groupby='samples'):
    return df.groupby(groupby).get_group(group).set_index(index).values

def sample(data, group, size=(100, ), replace=False, to_torch=False, use_cuda=False, use_mps=False):
    sub = group_extract(data, group)
    idx = np.arange(sub.shape[0])
    sampled = sub[np.random.choice(idx, size=size, replace=replace)]
    if to_torch:
        sampled = torch.Tensor(sampled).float()
        if use_cuda:
            sampled = sampled.cuda()
        if use_mps:
            sampled = sampled.mps()
    return sampled

def to_np(data):
    return data.detach().cpu().numpy()

def generate_steps(groups):
    return list(zip(groups[:-1], groups[1:]))
    
def set_seeds(seed:int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def config_hold_out(df:pd.DataFrame, hold_out:str='random', hold_one_out:bool=False):
    DF = None
    if not hold_one_out:
        DF = df
        groups = sorted(df.samples.unique())
    elif hold_one_out is True and hold_out in groups:
        df_ho = df.drop(df[df['samples']==hold_out].index, inplace=False)
        DF = df_ho
        groups = sorted(df_ho.samples.unique())
    else:
        raise ValueError(f'group={hold_out} not in known groups {groups}')
    return DF, groups

from .losses import MMD_loss, OT_loss1
def config_criterion(criterion_name:str='ot', use_cuda:bool=False):
    _valid_criterion_names = 'ot mmd'.split()
    if criterion_name == 'mmd':
        criterion = MMD_loss()
    elif criterion_name == 'ot':
        criterion = OT_loss1(use_cuda=use_cuda)
    else:
        raise NotImplementedError(
            f'{criterion_name} not implemented.\n'
            f'Please use one of {_valid_criterion_names}'
        )
    return criterion

def get_groups_from_df(df, samples_key='samples', samples=None):
    try:
        groups = sorted(df[samples_key].unique())  
    except KeyError:
        if samples is not None:
            groups = sorted(np.unique(samples))  
        else:
            raise ValueError(
                f'DataFrame df has no key {samples_key} and backup list of samples'
                f' samples is None.'
            )
    return groups

def get_cell_types_from_df(df, cell_type_key=None, cell_types=None):
    if cell_types is None:
        try:
            if cell_type_key is None:
                cell_types = sorted(df.index.unique())
            else:
                cell_types = sorted(df[cell_type_key].unique())
        except KeyError:
            raise KeyError(
                f'DataFrame df has no key {cell_type_key} and backup list of cell types'
                ' cell_types is None'
            )
    return cell_types


def get_sample_n_from_df(
    df, n, samples_key='samples', samples=None,    
    groups=None,
    drop_index=False
):
    if groups is None:
        groups =  get_groups_from_df(df, samples_key, samples)
        
    try:
        counts_n = df.reset_index(drop=drop_index)[df[samples_key] == groups[n]]
    except KeyError:
        if samples is not None:
            counts_n = df.reset_index(drop=drop_index)[samples == groups[n]]
        else:
            raise ValueError(
                f'DataFrame df has no key {samples_key} and backup list of samples'
                f' samples is None.'
            )
    return counts_n

def get_times_from_groups(groups, where='start', start=0):
    _valid_where = 'start end'.split()
    if where not in _valid_where:
        raise ValueError(f'{where} not known. Should be one of {_valid_where}')

    times = groups
    if where == 'end':
        times = times[::-1]
    times = times[start:]
    return times

def cal_mass_loss(data_t1, x_t_last, lnw_t_last, relative_mass, batch_size):
    distances = torch.cdist(data_t1, x_t_last)
    _, indices = torch.min(distances, dim=1)
    weights = torch.exp(lnw_t_last).squeeze(1)
    count = torch.zeros_like(weights)
    for idx in indices:
        count[idx] += 1
    relative_count = count  / batch_size
    local_mass_loss = torch.norm(weights - relative_mass*relative_count, p=2)**2
    return local_mass_loss

def cal_mass_loss_reduce(data_t1, x_t_last, lnw_t_last, relative_mass, batch_size, dim_reducer=None):
    if dim_reducer is not None:
        data_t1_reduced = dim_reducer(data_t1.detach().cpu().numpy())
        x_t_last_reduced = dim_reducer(x_t_last.detach().cpu().numpy())
    else:
        data_t1_reduced = data_t1
        x_t_last_reduced = x_t_last
    
    data_t1_reduced = torch.as_tensor(data_t1_reduced)
    x_t_last_reduced = torch.as_tensor(x_t_last_reduced)
    
    distances = torch.cdist(data_t1_reduced, x_t_last_reduced)
    _, indices = torch.min(distances, dim=1)
    
    weights = torch.exp(lnw_t_last).squeeze(1)
    
    count = torch.zeros_like(weights)
    for idx in indices:
        count[idx] += 1
    
    relative_count = count / batch_size
    local_mass_loss = torch.norm(weights - relative_mass * relative_count, p=2)**2
    
    return local_mass_loss

from .losses import MMD_loss, OT_loss1, OT_loss2, Density_loss, Local_density_loss
from CytoBridge.constants import ROOT_DIR, DATA_DIR, NTBK_DIR, IMGS_DIR, RES_DIR

_valid_datasets = {
    'file': lambda file: np.load(file),
}

_valid_criterions = {
    'mmd': MMD_loss,
    'ot1': OT_loss1,
    'ot2': OT_loss2,
}

import argparse
import sys

parser = argparse.ArgumentParser(prog='CytoBridge Training', description='Train CytoBridge')

parser.add_argument(
    '--dataset', '-d', type=str, choices=list(_valid_datasets.keys()), required=False,
    help=(
        'Dataset of the experiment to use. '
        'If value is fullpath to a file then tries to load file. '
        'Note, if using your own file we assume it is a pandas '
        'dataframe which has a column named `samples` that correspond to '
        'the timepoints.'
    )
)

parser.add_argument(
    '--time-col', '-tc', type=str, choices='simulation_i step_ix sim_time'.split(), required=False,
    help='Time column of the dataset to use.'
)

parser.add_argument(
    '--name', '-n', type=str, required=True, default=None,
    help='Name of the experiment. If none is provided timestamp is used.'
)

parser.add_argument(
    '--output-dir', '-od', type=str, default=RES_DIR,
    help='Where experiments should be saved. The results directory will automatically be generated here.'
)

parser.add_argument(
    '--local-epochs', '-le', type=int, default=5,
    help='Number of epochs to use `local_loss` while training. These epochs occur first. Defaults to `5`.'
)
parser.add_argument(
    '--epochs', '-e', type=int, default=15,
    help='Number of epochs to use `global_loss` while training. Defaults to `15`.' 
)
parser.add_argument(
    '--local-post-epochs', '-lpe', type=int, default=5,
    help='Number of epochs to use `local_loss` after training. These epochs occur last. Defaults to `5`.'
)

parser.add_argument(
    '--criterion', '-c', type=str, choices=list(_valid_criterions.keys()), 
    default='ot1', required=False,
    help='a loss function, either `"mmd"` or `"emd"`. Defaults to `"mmd"`.'
)

parser.add_argument(
    '--batches', '-b', type=int, default=100,
    help='the number of batches from which to randomly sample each consecutive pair of groups.'
)

parser.add_argument(
    '--cuda', '--use-gpu', '-g', type=bool, 
    action=argparse.BooleanOptionalAction, default=True,     
    help='Whether or not to use CUDA. Defaults to `True`.'
)

parser.add_argument(
    '--sample-size', '-ss', type=int, default=100,     
    help='Number of points to sample during each batch. Defaults to `100`.'
)

parser.add_argument(
    '--sample-with-replacement', '-swr', type=bool, 
    action=argparse.BooleanOptionalAction, default=False,     
    help='Whether or not to sample with replacement. Defaults to `True`.'
)

parser.add_argument(
    '--hold-one-out', '-hoo', type=bool, 
    action=argparse.BooleanOptionalAction, default=True,
    help='Defaults to `True`. Whether or not to randomly hold one time pair e.g. t_1 to t_2 out when computing the global loss.'
)

parser.add_argument(
    '--hold-out', '-ho', type=str, default='random',
    help='Defaults to `"random"`. Which time point to hold out when calculating the global loss.'
)

parser.add_argument(
    '--apply-losses-in-time', '-it', type=bool, 
    action=argparse.BooleanOptionalAction, default=True,
    help='Defaults to `True`. Applies the losses and does back propagation as soon as a loss is calculated. See notes for more detail.'
)

parser.add_argument(
    '--top-k', '-k', type=int, default=5,
    help='the k for the k-NN used in the density loss'
)

parser.add_argument(
    '--hinge-value', '-hv', type=float, default=0.01,
    help='hinge value for density loss function. Defaults to `0.01`.'
)

parser.add_argument(
    '--use-density-loss', '-udl', type=bool, 
    action=argparse.BooleanOptionalAction, default=True,
    help='Defaults to `True`. Whether or not to add density regularization.'
)

parser.add_argument(
    '--use-local-density', '-uld', type=bool, 
    action=argparse.BooleanOptionalAction, default=False,
    help='Defaults to `False`. Whether or not to use local density.'
)

parser.add_argument(
    '--lambda-density', '-ld', type=float, default=1.0,
    help='The weight for density loss. Defaults to `1.0`.'
)

parser.add_argument(
    '--lambda-density-local', '-ldl', type=float, default=1.0,
    help='The weight for local density loss. Defaults to `1.0`.'
)

parser.add_argument(
    '--lambda-local', '-ll', type=float, default=0.2,
    help='the weight for average local loss.  Note `lambda_local + lambda_global = 1.0`. Defaults to `0.2`.'
)

parser.add_argument(
    '--lambda-global', '-lg', type=float, default=0.8,
    help='the weight for global loss. Note `lambda_local + lambda_global = 1.0`. Defaults to `0.8`.'
)

parser.add_argument(
    '--model-layers', '-ml', type=int, nargs='+', default=[64],
    help='Layer sizes for ode model'
)

parser.add_argument(
    '--use-geo', '-ug', type=bool, default=False,
    action=argparse.BooleanOptionalAction,
    help='Whether or not to use a geodesic embedding'
)

parser.add_argument(
    '--geo-layers', '-gl', type=int, nargs='+', default=[32],
    help='Layer sizes for geodesic embedding model'
)
parser.add_argument(
    '--geo-features', '-gf', type=int, default=5,
    help='Number of features for geodesic model.'
)

parser.add_argument(
    '--n-points', '-np', type=int, default=100,
    help='number of trajectories to generate for plot. Defaults to `100`.'
)

parser.add_argument(
    '--n-trajectories', '-nt', type=int, default=30,
    help='number of trajectories to generate for plot. Defaults to `30`.'
)

parser.add_argument(
    '--n-bins', '-nb', type=int, default=100,
    help='number of bins to use for generating trajectories. Higher make smoother trajectories. Defaults to `100`.'
)

parser.add_argument(
    '--num_particles', '-npm', type=int, default=16,
    help='number of particles to use for the simulation. Defaults to `16`.'
)

def trace_df_dz(f, z):
    sum_diag = 0.0
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0][:, i]
    return sum_diag

import math
import warnings
from functools import partial
from typing import Optional

import numpy as np
import ot as pot
import torch

class OTPlanSampler:

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        warn: bool = True,
    ) -> None:
        if method == "exact":
            self.ot_fn = pot.emd
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn

    def get_map(self, x0, x1):
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map(self, pi, batch_size, replace=True):
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, replace=True):
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]

    def sample_plan_with_labels(self, x0, x1, y0=None, y1=None, replace=True):
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )

    def sample_trajectory(self, X):
        times = X.shape[1]
        pis = []
        for t in range(times - 1):
            pis.append(self.get_map(X[:, t], X[:, t + 1]))

        indices = [np.arange(X.shape[0])]
        for pi in pis:
            j = []
            for i in indices[-1]:
                j.append(np.random.choice(pi.shape[1], p=pi[i] / pi[i].sum()))
            indices.append(np.array(j))

        to_return = []
        for t in range(times):
            to_return.append(X[:, t][indices[t]])
        to_return = np.stack(to_return, axis=1)
        return to_return

def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret

import math
import warnings
from typing import Union

import torch

def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

class ConditionalFlowMatcher:

    def __init__(self, sigma: Union[float, int] = 0.0):
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)

class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):

    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1

class TargetConditionalFlowMatcher(ConditionalFlowMatcher):

    def compute_mu_t(self, x0, x1, t):
        del x0
        t = pad_t_like_x(t, x1)
        return t * x1

    def compute_sigma_t(self, t):
        return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0, x1, t, xt):
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)

class SchrodingerBridgeConditionalFlowMatcher(ConditionalFlowMatcher):

    def __init__(self, sigma: Union[float, int] = 1.0, ot_method="exact"):
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")
        super().__init__(sigma)
        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma**2)

    def compute_sigma_t(self, t):
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(self, x0, x1, t, xt):
        t = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        return ut

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1

class VariancePreservingConditionalFlowMatcher(ConditionalFlowMatcher):

    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        del xt
        t = pad_t_like_x(t, x0)
        return math.pi / 2 * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)

from torchdiffeq import odeint_adjoint as odeint
from .models import velocityNet, growthNet, scoreNet, dediffusionNet, indediffusionNet, FNet, ODEFunc2, ODEFunc2_interaction_energy

def generate_state_trajectory(X, n_times, batch_size,f_net,time, device):
    lnw0 = torch.log(torch.ones(batch_size,1) / (batch_size)).float().to(device)
    x0 = torch.from_numpy(X[0]).float().to(device)
    trajectory = [x0]
    for t_start in range(n_times - 1):
        t_mid = torch.Tensor([time[t_start], time[t_start+1]]).float().to(device)
        m0 = torch.zeros_like(lnw0).to(device)
        initial_state_energy = (trajectory[-1], lnw0, m0)
        xtt, _, _=odeint(ODEFunc2(f_net),initial_state_energy,t_mid,options=dict(step_size=0.01),method='euler')
        trajectory.append(xtt[-1].detach())
    return trajectory

def generate_state_trajectory_interaction(X, n_times, batch_size,f_net,time, device):
    lnw0 = torch.log(torch.ones(batch_size,1) / (batch_size)).float().to(device)
    x0 = torch.from_numpy(X[0]).float().to(device)
    x0.requires_grad=True
    trajectory = [x0]
    for t_start in range(n_times - 1):
        t_mid = torch.Tensor([time[t_start], time[t_start+1]]).float().to(device)
        trajectory[-1].requires_grad=True
        lnw0.requires_grad = True
        m0 = torch.zeros_like(lnw0).to(device)
        initial_state_energy = (trajectory[-1], lnw0, m0)
        
        xtt, _, _=odeint(ODEFunc2_interaction_energy(f_net),initial_state_energy,t_mid,options=dict(step_size=0.01),method='euler')
        trajectory.append(xtt[-1].detach())
    return trajectory

def get_batch(FM, X, trajectory,batch_size, n_times, return_noise=False):
    ts = []
    xts = []
    uts = []
    noises = []
    

    for t_start in range(n_times - 1):
        x0 = trajectory[t_start]
        x1 = trajectory[t_start + 1]
        
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(
                x0, x1, return_noise=return_noise
            )
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)

    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises
    return t, xt, ut

def density1(x,datatime0,device):
    """Density function for Multimodal Gaussian."""
    mu = datatime0.to(device)  # Shape: [num_samples, 10]
    num_gaussian = mu.shape[0]  # Number of Gaussian components
    dim = mu.shape[1]  # Dimensionality (10)
    
    # Define a fixed covariance matrix (0.4 * I)
    sigma_matrix = 0.4 * torch.eye(dim).type(torch.float32).to(device)
    
    # Initialize density values to zero
    p_unn = torch.zeros([x.shape[0]], dtype=torch.float32).to(device)
    
    # Sum the densities from each Gaussian component
    for i in range(num_gaussian):
        m = torch.distributions.MultivariateNormal(mu[i, :], sigma_matrix)
        p_unn += 2 * torch.exp(m.log_prob(x))
    
    # Average the density values
    p_n = p_unn / num_gaussian
    return p_n
def get_batch_size(FM, X, trajectory, batch_size, n_times, return_noise=False, hold_one_out=False, hold_out=None, device = 'cpu'):
    ts = []
    xts = []
    uts = []
    noises = []

    if hold_one_out:
        if hold_out == 'random':
            raise ValueError("hold_out='random' is not supported, please specify a concrete time step index.")
        trajectory = [data for idx, data in enumerate(trajectory) if idx != hold_out]
        n_times = len(trajectory)

    for t_start in range(n_times - 1):
        x0 = trajectory[t_start]
        x1 = trajectory[t_start + 1]
        indices0 = np.random.choice(len(x0), size=batch_size, replace=False)
        indices1 = np.random.choice(len(x1), size=batch_size, replace=False)
        
        x0 = x0[indices0]
        x1 = x1[indices1]
        
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
            
        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)

    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises
    
    return t, xt, ut




