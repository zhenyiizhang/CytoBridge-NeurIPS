import os
import random
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import anndata as ad
import scanpy as sc
from tqdm import tqdm
from scipy.spatial import distance_matrix
from sklearn.gaussian_process.kernels import RBF
from sklearn.manifold import MDS
from torchdiffeq import odeint_adjoint as odeint

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from CytoBridge.losses import MMD_loss, OT_loss1, OT_loss2, Density_loss, Local_density_loss
from CytoBridge.utils import (
    group_extract, sample, to_np, generate_steps, cal_mass_loss, parser, _valid_criterions,
    OTPlanSampler, ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher, SchrodingerBridgeConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher, generate_state_trajectory, get_batch, get_batch_size
)
from CytoBridge.train import (
    train_un1_reduce,
    train_un1_reduce_interaction, train_all
)
from CytoBridge.models import (
    velocityNet, growthNet, scoreNet, dediffusionNet, indediffusionNet,
    FNet_interaction, ODEFunc2, scoreNet2
)
from CytoBridge.constants import ROOT_DIR, DATA_DIR, NTBK_DIR, IMGS_DIR, RES_DIR
from CytoBridge.exp import setup_exp
from CytoBridge.eval import generate_plot_data
from CytoBridge.interaction import cal_interaction, euler_sdeint

dim = 10
f_net = FNet_interaction(in_out_dim=dim, hidden_dim=400, n_hiddens=2, activation='leakyrelu', thre=2)

sys.argv = [
    'CytoBridge Training',
    '-d', 'file',
    '-c', 'ot1',
    '-n', 'emt',
]

args = parser.parse_args()
opts = vars(args)

print(opts)

device = torch.device('cuda')

df = pd.read_csv(DATA_DIR + '/emt.csv')
df = df.iloc[:, :11]

if not os.path.isdir(opts['output_dir']):
    os.makedirs(opts['output_dir'])
exp_dir, logger = setup_exp(opts['output_dir'], opts, opts['name'])

logger.info(f'Loading dataset')

groups = sorted(df.samples.unique())
steps = generate_steps(groups)
logger.info(f'Defining model')
use_geo = opts['use_geo']
model_layers = opts['model_layers']
model_features = len(df.columns) - 1

logger.info(f'Defining optimizer and criterion')
optimizer = torch.optim.Adam(f_net.parameters(), 0.0001)
params_v_net = f_net.v_net.parameters()
params_g_net = f_net.g_net.parameters()

optimizer1 = torch.optim.Adam(params_v_net)
optimizer2 = torch.optim.Adam(params_g_net)

opts['criterion'] = 'ot1'
criterion = OT_loss1(which='emd')

logger.info(f'Extracting parameters')
use_cuda = torch.cuda.is_available() and opts['cuda']
sample_size = (opts['sample_size'], )
sample_with_replacement = opts['sample_with_replacement']
apply_losses_in_time = opts['apply_losses_in_time']

n_local_epochs = opts['local_epochs']
n_epochs = opts['epochs']
n_post_local_epochs = opts['local_post_epochs']
n_batches = opts['batches']

hold_one_out = False
hold_out = 4

hinge_value = opts['hinge_value']
top_k = opts['top_k']
lambda_density = opts['lambda_density']
lambda_density_local = opts['lambda_density_local']
use_density_loss = opts['use_density_loss']
use_local_density = opts['use_local_density']

lambda_local = opts['lambda_local']
lambda_global = opts['lambda_global']

n_points = opts['n_points']
n_trajectories = opts['n_trajectories']
n_bins = opts['n_bins']

local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
batch_losses = []
globe_losses = []

f_net = f_net.to(device)
initial_size = df[df['samples']==0].x1.shape[0]

if hold_one_out:
    df_mass = df[df['samples'] != hold_out]
    sample_sizes = df_mass.groupby('samples').size()
    ref0 = sample_sizes / sample_sizes.iloc[0]
    relative_mass = torch.tensor(ref0.values)
else:
    sample_sizes = df.groupby('samples').size()
    ref0 = sample_sizes / sample_sizes.iloc[0]
    relative_mass = torch.tensor(ref0.values)
print(relative_mass)
sample_size = (initial_size,)

print('Pretraining growth net')
if n_local_epochs > 0:
    logger.info(f'Beginning pretraining')
    for epoch in tqdm(range(1), desc='Pretraining Epoch'):
        l_loss, b_loss, g_loss = train_un1_reduce(
            f_net, df, groups, optimizer, 100,
            criterion=criterion, use_cuda=use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=apply_losses_in_time,
            hold_one_out=hold_one_out, hold_out=hold_out,
            hinge_value=hinge_value, lambda_ot=1, lambda_mass=0.01, lambda_energy=0.0,
            use_pinn=False, use_penalty=False, use_density_loss=False, lambda_density=10,
            top_k=top_k, sample_size=sample_size, relative_mass=relative_mass, initial_size=initial_size,
            sample_with_replacement=sample_with_replacement, logger=logger, device=device, best_model_path=exp_dir+'/best_model',
            pca_transform=None, reverse=False, global_mass=False
        )

print('Pretraining Velocity and Interaction')
f_net.load_state_dict(torch.load(os.path.join(exp_dir+'/best_model'), map_location=torch.device('cpu')))
f_net.to(device)

for param in f_net.g_net.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(f_net.parameters(), lr=1e-5)

if n_local_epochs > 0:
    logger.info(f'Beginning pretraining')
    for epoch in tqdm(range(1), desc='Pretraining Epoch'):
        l_loss, b_loss, g_loss = train_un1_reduce_interaction(
            f_net, df, groups, optimizer, 100,
            criterion=criterion, use_cuda=use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=apply_losses_in_time,
            hold_one_out=hold_one_out, hold_out=hold_out,
            hinge_value=hinge_value, lambda_ot=1, lambda_mass=0, lambda_energy=0.0,
            use_pinn=False, use_penalty=False, use_density_loss=False, lambda_density=10,
            top_k=top_k, sample_size=sample_size, relative_mass=relative_mass, initial_size=initial_size,
            sample_with_replacement=sample_with_replacement, logger=logger, device=device, best_model_path=exp_dir+'/best_model',
            pca_transform=None, reverse=False, thre=1000
        )

f_net.load_state_dict(torch.load(os.path.join(exp_dir+'/best_model'), map_location=torch.device('cpu')))
f_net.to(device)

print('Pretrianing score')
device = 'cuda'
f_net.load_state_dict(torch.load(os.path.join(exp_dir+'/best_model'), map_location=torch.device('cpu')))
f_net.to(device)

print("DataFrame shape:", df.shape)
print("DataFrame columns:", df.columns)
n = dim
samples = df['samples'].values
column_names = [f'x{i}' for i in range(1, n + 1)]

obsm_data = df[column_names].values
print("obsm_data shape:", obsm_data.shape)

adata = ad.AnnData(obs=pd.DataFrame(index=samples))
adata.obsm['X_pca'] = obsm_data
adata_loaded = adata
print(adata_loaded)

adata.obs['samples'] = df['samples'].values
sc.pl.scatter(adata, basis="pca", color="samples")

n_times = len(adata.obs["samples"].unique())
print(n_times)
X = [
    adata.obsm["X_pca"][adata.obs["samples"] == t]
    for t in range(n_times)
]

batch_size = df[df['samples']==0].x1.shape[0]
sigma = 0.1
time = torch.Tensor(groups)
SF2M = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
sf2m_score_model = scoreNet2(in_out_dim=dim, hidden_dim=128, activation='leakyrelu').float().to(device)
sf2m_optimizer = torch.optim.AdamW(
    list(sf2m_score_model.parameters()), 1e-4
)
trajectory = generate_state_trajectory(X, n_times, batch_size, f_net, time, device)

batch_size = 512
max_norm_ut = torch.tensor(0.0)
lambda_penalty = 1
for i in tqdm(range(1001)):
    sf2m_optimizer.zero_grad()
    t, xt, ut, eps = get_batch_size(SF2M, X, trajectory, batch_size, n_times, return_noise=True)
    t = torch.unsqueeze(t, 1)
    lambda_t = SF2M.compute_lambda(t % 1)
    value_st = sf2m_score_model(t, xt)
    st = sf2m_score_model.compute_gradient(t, xt)
    positive_st = torch.relu(value_st)
    penalty = lambda_penalty * torch.max(positive_st)

    score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
    if i % 100 == 0:
        print(torch.max(positive_st))
        print(f"{i}:  {score_loss.item():0.2f}")
    loss = score_loss + penalty

    loss.backward()
    sf2m_optimizer.step()

torch.save(sf2m_score_model.state_dict(), os.path.join(exp_dir, 'score_model'))

print('Train all')
sf2m_score_model.load_state_dict(torch.load(os.path.join(exp_dir, 'score_model'), map_location=torch.device('cuda')))
sf2m_score_model.to(device)
f_net.load_state_dict(torch.load(os.path.join(exp_dir+'/best_model'), map_location=torch.device('cuda')))
f_net.to(device)

device = 'cuda'
optimizer = torch.optim.Adam(list(f_net.parameters())+list(sf2m_score_model.parameters()), 1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
datatime0 = torch.zeros(df[df['samples']==0].x1.shape[0], dim)
datatime0 = torch.tensor(df[df['samples']==0].iloc[:,1:].values).to(device)
for param in f_net.parameters():
    param.requires_grad = True

for param in sf2m_score_model.parameters():
    param.requires_grad = True

if n_local_epochs > 0:
    logger.info(f'Beginning Training')
    for epoch in tqdm(range(1), desc='Training Epoch'):
        l_loss, b_loss, g_loss = train_all(
            f_net, df, groups, optimizer, 200,
            criterion=criterion, use_cuda=use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=apply_losses_in_time,
            hold_one_out=hold_one_out, hold_out=hold_out, sf2m_score_model=sf2m_score_model,
            hinge_value=hinge_value, datatime0=datatime0, device=device, lambda_initial=0.1,
            use_pinn=True, use_penalty=True, use_density_loss=False, lambda_density=10,
            top_k=top_k, sample_size=sample_size, relative_mass=relative_mass, initial_size=initial_size,
            sample_with_replacement=sample_with_replacement, logger=logger, sigmaa=sigma, lambda_pinn=100,
            exp_dir=exp_dir, thre=1000, lambda_ot=10, lambda_mass=10, lambda_energy=0.01, log_mass=True
        )

torch.save(sf2m_score_model.state_dict(), os.path.join(exp_dir, 'score_model_result_final'))
torch.save(f_net.state_dict(), os.path.join(exp_dir, 'model_result_final'))