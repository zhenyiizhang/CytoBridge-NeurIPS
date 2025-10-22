import os, random, argparse, pandas as pd, numpy as np, seaborn as sns
from tqdm import tqdm
import torch, torch.nn as nn
import sys
import torch.optim as optim
import torchsde
import importlib
import anndata as ad
import scanpy as sc

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)

from scipy.spatial import distance_matrix
from sklearn.gaussian_process.kernels import RBF
from sklearn.manifold import MDS
from torchdiffeq import odeint

from CytoBridge.losses import MMD_loss, OT_loss1, OT_loss2, Density_loss, Local_density_loss 
from CytoBridge.utils import (
    group_extract, sample, to_np, generate_steps, cal_mass_loss, parser, _valid_criterions,
    OTPlanSampler, ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher, SchrodingerBridgeConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher, generate_state_trajectory, get_batch
)
from CytoBridge.train import train_un1_reduce_interaction, train_un1_reduce, train_all
from CytoBridge.models import (
    velocityNet, growthNet, scoreNet, dediffusionNet, indediffusionNet, FNet,
    ODEFunc2, FNet_interaction, ODEFunc2_interaction, ODEFunc_interaction, scoreNet2
)
from CytoBridge.constants import ROOT_DIR, DATA_DIR, NTBK_DIR, IMGS_DIR, RES_DIR
from CytoBridge.exp import setup_exp
from CytoBridge.eval import generate_plot_data, generate_plot_data_interaction
from CytoBridge.interaction import cal_interaction, euler_sdeint

f_net = FNet_interaction(in_out_dim=2, hidden_dim=400, n_hiddens=2, activation='leakyrelu', thre = 0.5)

sys.argv = [
    'CytoBridge Training',
    '-d', 'file',
    '-c', 'ot1',
    '-n', 'Attract',
]

args = parser.parse_args()
opts = vars(args)
print(opts)

num_particles = int(opts.get('num_particles', 16))  # default to 16 if not set
print(num_particles)

device = torch.device('cuda')
device

df=pd.read_csv(DATA_DIR + '/simulation_attract.csv')

if not os.path.isdir(opts['output_dir']):
    os.makedirs(opts['output_dir'])
exp_dir, logger = setup_exp(opts['output_dir'], opts, opts['name'])

groups = sorted(df.samples.unique())
steps = generate_steps(groups)
use_geo = opts['use_geo']
model_layers = opts['model_layers']
model_features = len(df.columns) - 1

optimizer = torch.optim.AdamW(f_net.parameters())

opts['criterion']='ot1'
criterion =  _valid_criterions[opts['criterion']]()

use_cuda = torch.cuda.is_available() and opts['cuda']
sample_size = 500
sample_with_replacement = opts['sample_with_replacement' ]
apply_losses_in_time = opts['apply_losses_in_time']

n_local_epochs = opts['local_epochs']
n_epochs = opts['epochs']
n_post_local_epochs = opts['local_post_epochs']
n_batches = opts['batches']

hinge_value = opts['hinge_value']
top_k = opts['top_k']
lambda_density = opts['lambda_density']
lambda_density_local = opts['lambda_density_local']
use_density_loss = opts['use_density_loss']
use_local_density = opts['use_local_density']
    
lambda_local = opts['lambda_local']
lambda_global = opts['lambda_global']

n_points=opts['n_points']
n_trajectories=opts['n_trajectories'] 
n_bins=opts['n_bins']
    
local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
batch_losses = []
globe_losses = []

f_net=f_net.to(device)
f_net

initial_size=df[df['samples']==0].x1.shape[0]
initial_size

sample_size = (df[df['samples']==0.0].values.shape[0],)
sample_size=(400,)

hold_one_out = False
hold_out = 1

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

if n_local_epochs > 0:
    logger.info(f'Beginning pretraining')
    for epoch in tqdm(range(1), desc='Pretraining Epoch'):
        l_loss, b_loss, g_loss = train_un1_reduce(
            f_net, df, groups, optimizer,200, 
            criterion = criterion, use_cuda = use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=apply_losses_in_time,
            hold_one_out=hold_one_out, hold_out=hold_out, 
                hinge_value=hinge_value, lambda_ot=1, lambda_mass=0.01, lambda_energy=0.0,
                use_pinn=False, use_penalty=False,use_density_loss=False,lambda_density=10,
            top_k = top_k, sample_size = sample_size,relative_mass=relative_mass,initial_size=initial_size,
            sample_with_replacement = sample_with_replacement, logger=logger, device=device,best_model_path=exp_dir+'/best_model',
        )

f_net.load_state_dict(torch.load(os.path.join(exp_dir+'/best_model'),map_location=torch.device('cpu')))
f_net.to(device)

optimizer = torch.optim.Adam(f_net.parameters(), lr = 1e-5)

for param in f_net.g_net.parameters():
    param.requires_grad = False

if n_local_epochs > 0:
    logger.info(f'Beginning pretraining')
    for epoch in tqdm(range(1), desc='Pretraining Epoch'):
        l_loss, b_loss, g_loss = train_un1_reduce_interaction(
            f_net, df, groups, optimizer,100, 
            criterion = criterion, use_cuda = use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=apply_losses_in_time,
            hold_one_out=hold_one_out, hold_out=hold_out, 
                hinge_value=hinge_value, lambda_ot=1, lambda_mass=0, lambda_energy=0.000,
                use_pinn=False, use_penalty=False,use_density_loss=False,lambda_density=10,
            top_k = top_k, sample_size = sample_size,relative_mass=relative_mass,initial_size=initial_size,
            sample_with_replacement = sample_with_replacement, logger=logger, device=device,best_model_path=exp_dir+'/best_model',
            num_particles=num_particles,
        )

f_net.load_state_dict(torch.load(os.path.join(exp_dir+'/best_model'),map_location=torch.device('cpu')))
f_net.to(device)

f_net.load_state_dict(torch.load(os.path.join(exp_dir+'/best_model'),map_location=torch.device('cpu')))
f_net.to(device)
print("DataFrame shape:", df.shape)
print("DataFrame columns:", df.columns)
n=2
samples = df['samples'].values
column_names = [f'x{i}' for i in range(1, n + 1)]
obsm_data = df[column_names].values
print("obsm_data shape:", obsm_data.shape)
adata = ad.AnnData(obs=pd.DataFrame(index=samples))
adata.obsm['X_pca'] = obsm_data
adata_loaded = adata
print(adata_loaded)

adata.obs['samples']=df['samples'].values

n_times = len(adata.obs["samples"].unique())
print(n_times)
X = [
    adata.obsm["X_pca"][adata.obs["samples"] == t]
    for t in range(n_times)
]

batch_size = df[df['samples']==0].x1.shape[0]
sigma = 0.05
dim = 2
time = torch.Tensor(groups)
SF2M = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
sf2m_score_model=scoreNet2(in_out_dim=dim, hidden_dim=128,  activation='leakyrelu').float().to(device)
sf2m_optimizer = torch.optim.AdamW(
    list(sf2m_score_model.parameters()), 1e-4
)
trajectory = generate_state_trajectory(X, n_times,batch_size, f_net, time, device)

batch_size = df[df['samples']==0].x1.shape[0]
dim = 2
time = torch.Tensor(groups)
SF2M = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
sf2m_score_model=scoreNet2(in_out_dim=dim, hidden_dim=128,  activation='leakyrelu').float().to(device)
sf2m_optimizer = torch.optim.AdamW(
    list(sf2m_score_model.parameters()), 1e-4
)
trajectory = generate_state_trajectory(X, n_times,batch_size, f_net, time, device)

max_norm_ut = torch.tensor(0.0)
lambda_penalty=1
for i in tqdm(range(1001)):
    sf2m_optimizer.zero_grad()
    t, xt, ut,eps = get_batch(SF2M, X, trajectory,batch_size, n_times, return_noise=True)
    t=torch.unsqueeze(t,1)
    lambda_t = SF2M.compute_lambda(t % 1)
    value_st=sf2m_score_model(t, xt)
    st = sf2m_score_model.compute_gradient(t, xt)
    positive_st = torch.relu(value_st)
    penalty = lambda_penalty * torch.max(positive_st)

    score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
    if i % 100 == 0:
        print(torch.max(positive_st))
        print(f"{i}:  {score_loss.item():0.2f}")
    loss = score_loss+penalty

    loss.backward()
    sf2m_optimizer.step()

torch.save(sf2m_score_model.state_dict(), os.path.join(exp_dir, 'score_model'))

sf2m_score_model.load_state_dict(torch.load(os.path.join(exp_dir, 'score_model'),map_location=torch.device('cuda')))
sf2m_score_model.to(device)
f_net.load_state_dict(torch.load(os.path.join(exp_dir+'/best_model'),map_location=torch.device('cuda')))
f_net.to(device)

datatime0=torch.zeros(df[df['samples']==0].x1.shape[0],2)
datatime0[:,0]=torch.tensor(df[df['samples']==0].x1)
datatime0[:,1]=torch.tensor(df[df['samples']==0].x2)

from CytoBridge.train import train_all
device='cuda'
optimizer = torch.optim.AdamW(list(f_net.parameters()),1e-5)
sf2m_optimizer = torch.optim.AdamW(list(sf2m_score_model.parameters()),1e-5)

for param in f_net.parameters():
    param.requires_grad = True

for param in sf2m_score_model.parameters():
    param.requires_grad = True

if n_local_epochs > 0:
    logger.info(f'Beginning Training')
    for epoch in tqdm(range(1), desc='Training Epoch'):
        l_loss, b_loss, g_loss, = train_all(
            f_net, df, groups, optimizer,200,
            criterion = criterion, use_cuda = use_cuda,
            local_loss=True, global_loss=False, apply_losses_in_time=apply_losses_in_time,
            hold_one_out=hold_one_out, hold_out=hold_out, sf2m_score_model=sf2m_score_model,
                hinge_value=hinge_value,datatime0=datatime0,device=device, lambda_initial=0.0,
                use_pinn=True, use_penalty=True,use_density_loss=False,lambda_density=10,
            top_k = top_k, sample_size = sample_size,relative_mass=relative_mass,initial_size=initial_size,
            sample_with_replacement = sample_with_replacement, logger=logger, sigmaa=sigma,lambda_pinn=100,
             exp_dir = exp_dir, lambda_mass = 10, lambda_ot = 10, use_mass = True,sf2m_optimizer=sf2m_optimizer,
        )
        
        for k, v in l_loss.items():  
            local_losses[k].extend(v)
        batch_losses.extend(b_loss)
        globe_losses.extend(g_loss)

torch.save(sf2m_score_model.state_dict(), os.path.join(exp_dir, 'score_model_result'))
torch.save(f_net.state_dict(), os.path.join(exp_dir, 'model_result'))

data=torch.tensor(df[df['samples']==0].values,dtype=torch.float32).requires_grad_()
data_t0 = data[:, 1:].to(device).requires_grad_()
print(data_t0.shape)
x0=data_t0.to(device)

#run_idx = 5
seed_list = [1,4,8,32,256]
for run_idx in range(1,6):

    SEED = seed_list[run_idx - 1]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)

    from CytoBridge.interaction import euler_sdeint
    from CytoBridge.interaction import cal_interaction, euler_sdeint
    import joblib
    class SDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self, ode_drift, g, score, interaction, input_size=(3, 32, 32), sigma=1.0):
            super().__init__()
            self.drift = ode_drift
            self.score = score
            self.input_size = input_size
            self.sigma = sigma
            self.interaction = interaction
            self.g_net = g

        # Drift
        def f(self, t, y):
            z, lnw = y
            drift=self.drift(t, z)
            dlnw = self.g_net(t, z)
            num = z.shape[0]
            t = t.expand(num, 1)  # 保持 t 的梯度信息并扩展其形状
            return (drift+cal_interaction(z, lnw, self.interaction, m=num_particles)+self.score.compute_gradient(t, z), dlnw) #+self.score.compute_gradient(t, z)

        # Diffusion
        def g(self, t, y):
            return torch.ones_like(y)*self.sigma
        
    # if x0.shape[0] >= 200:
    #indices = torch.randperm(x0.shape[0])[:2000]
    #x0_subset = x0[indices].to(device)
    # else:
    # # 如果 x0 不足 100 个，就直接全部使用
    x0_subset = x0.to(device)

    x0_subset = x0_subset.to(device)
    lnw0 = torch.log(torch.ones(x0_subset.shape[0], 1) / x0_subset.shape[0]).to(device)
    initial_state = (x0_subset, lnw0)

    # 定义 SDE 对象
    sde = SDE(f_net.v_net, 
            f_net.g_net, 
            sf2m_score_model, 
            f_net.interaction_net, 
            input_size=(50,), 
            sigma=sigma)

    # 定义时间点，假设总共积分 200 步
    ts = torch.linspace(0, n_times - 1, 100, device=device)

    # 手写 SDE 积分
    sde_traj, traj_lnw = euler_sdeint(sde, initial_state, dt=0.1, ts=ts)
    # 若需要转移到 CPU 上：
    sde_traj, traj_lnw = sde_traj.cpu(), traj_lnw.cpu()


    sample_number = 10  # 例如，采样10个
    sample_indices = random.sample(range(sde_traj.size(1)), sample_number)
    sampled_sde_trajec = sde_traj[:, sample_indices, :]
    sampled_sde_trajec.shape
    sampled_sde_trajec = sampled_sde_trajec.tolist()
    sampled_sde_trajec = np.array(sampled_sde_trajec, dtype=object)
    np.save(exp_dir+f'/scRNA_sde_trajec_025_our_post_plot_run_{run_idx}.npy', sampled_sde_trajec)

    from CytoBridge.interaction import euler_sdeint, euler_sdeint_split

    ts_points=torch.tensor([0.0,1.0,2.0, 3.0, 4.0]).to(device)
    ts_points

    print(time)
    sde_point, traj_lnw = euler_sdeint(sde, initial_state, dt=0.1, ts=ts_points)

    traj_point, _ = euler_sdeint_split(sde, initial_state, dt=0.1, ts=time, noise_std = 0.0)

    weight = torch.exp(traj_lnw)
    weight_normed = weight/weight.sum(dim = 1, keepdim = True)

    sde_point_np = sde_point.detach().cpu().numpy()
    sde_point_list = sde_point_np.tolist()
    sde_point_array = np.array(sde_point_list, dtype=object)

    sde_point_split_numpy = []
    for i in traj_point:
        sde_point_split_numpy.append(i.cpu().detach().numpy())
    sde_point_split_array = np.array(sde_point_split_numpy, dtype=object)
    np.save(exp_dir+f'/scRNA_sde_point_our_simu2d_025_post_run_{run_idx}.npy', sde_point_array)
    np.save(exp_dir+f'/scRNA_sde_weight_our_simu2d_025_post_run_{run_idx}.npy', weight.detach().cpu().numpy())
    np.save(exp_dir+f'/scRNA_sde_point_split_our_simu2d_025_post_run_{run_idx}.npy', sde_point_split_array)

