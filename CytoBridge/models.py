import torch, torch.nn as nn
from CytoBridge.interaction import cal_interaction
import math
import joblib
import torch.nn.init as init

class velocityNet(nn.Module):
    # input x, t to get v= dx/dt
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation='Tanh', use_spatial = False):
        super().__init__()
        Layers = [in_out_dim+1]
        for i in range(n_hiddens):
            Layers.append(hidden_dim)
        Layers.append(in_out_dim)
        
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        
        self.use_spatial = use_spatial

        if use_spatial:
            # Spatial velocity
            self.spatial_net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
            )
            self.spatial_out = nn.Linear(Layers[-2], 2)

            # Gene velocity
            self.gene_net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
            )
            self.gene_out = nn.Linear(Layers[-2], in_out_dim - 2)
        else:
            self.net = nn.ModuleList(
                [nn.Sequential(
                    nn.Linear(Layers[i], Layers[i + 1]),
                    self.activation,
                )
                    for i in range(len(Layers) - 2)
                ]
            )
            self.out = nn.Linear(Layers[-2], Layers[-1])

    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        #print(num)
        t = t.expand(num, 1)  # Maintain gradient information of t and expand its shape
        #print(t)
        state  = torch.cat((t,x),dim=1)
        #print(state)
        if self.use_spatial:
            ii = 0
            for layer in self.spatial_net:
                if ii == 0:
                    x = layer(state)
                else:
                    x = layer(x)
                ii =ii+1
            spatial_x = self.spatial_out(x)

            ii = 0
            for layer in self.gene_net:
                if ii == 0:
                    x = layer(state)
                else:
                    x = layer(x)
                ii =ii+1
            gene_x = self.gene_out(x)
            x = torch.cat([spatial_x, gene_x], dim = 1)
        else:
            ii = 0
            for layer in self.net:
                if ii == 0:
                    x = layer(state)
                else:
                    x = layer(x)
                ii =ii+1
            x = self.out(x)
        return x

class growthNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        num = x.shape[0]
        t = t.expand(num, 1)
        state  = torch.cat((t,x),dim=1)
        return self.net(state)

class scoreNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        num = x.shape[0]
        t = t.expand(num, 1)
        state  = torch.cat((t,x),dim=1)
        return self.net(state)

class dediffusionNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        num = x.shape[0]
        t = t.expand(num, 1)
        state  = torch.cat((t,x),dim=1)
        return self.net(state)

class indediffusionNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        num = x.shape[0]
        t = t.expand(num, 1)
        return self.net(t)

class InteractionModel_vanilla(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        self.net = nn.Sequential(
            nn.Linear(in_out_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, x):
        zero_input = torch.zeros(1, x.shape[-1], device=x.device, dtype=x.dtype)
        baseline = self.net(zero_input)
        potential = self.net(x) + self.net(-x) - 2 * baseline
        return potential

class InteractionModel(nn.Module):
    def __init__(self, x_dim, hidden_dim, activation='Tanh', num_rbf=16, cutoff=1, dim_reduce = False):
        super().__init__()
        self.num_rbf = num_rbf
        if activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation type: {}".format(activation))
        self.rbf_expansion = ExpNormalSmearing(cutoff=cutoff, num_rbf = self.num_rbf, trainable= True)
        self.net = nn.Sequential(
            nn.Linear(self.num_rbf, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1)
        )
        self.cutoff = cutoff
        self.eps = 1e-6
        self.dim_reduce = dim_reduce
        if self.dim_reduce:
            self.pca = nn.Linear(x_dim, 10, bias=False)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
    def forward(self, x_t):
        if self.cutoff == 0:
            return 0 * x_t.sum()
        if self.dim_reduce:
            x_t = self.pca(x_t)
        dis = self.compute_distance(x_t)
        dis_exp = self.rbf_expansion(dis[dis != 0])
        potential = self.net(dis_exp)
        return potential
    def compute_distance(self, x):
        return torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + self.eps)

class InteractionModel_spatial(nn.Module):
    def __init__(self, x_dim, hidden_dim, activation='Tanh', num_rbf=16, cutoff=1, dim_reduce = False):
        super().__init__()
        self.num_rbf = num_rbf
        if activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation type: {}".format(activation))
        self.spatial_rbf_expansion = ExpNormalSmearing(cutoff=cutoff, num_rbf = self.num_rbf, trainable= True)
        self.express_rbf_expansion = ExpNormalSmearing(cutoff=15, num_rbf = self.num_rbf, trainable= True)
        self.net = nn.Sequential(
            nn.Linear(self.num_rbf*2, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1)
        )
        self.cutoff = cutoff
        self.eps = 1e-6
        self.dim_reduce = dim_reduce
        if self.dim_reduce:
            self.pca = nn.Linear(x_dim, 10, bias=False)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
    def forward(self, x_t):
        if self.cutoff == 0:
            return 0 * x_t.sum()
        if self.dim_reduce:
            x_t = self.pca(x_t)
        spatial_dis = self.compute_distance(x_t[:,:2])
        express_dis = self.compute_distance(x_t[:,2:])
        spatial_dis_exp = self.spatial_rbf_expansion(spatial_dis[spatial_dis <= self.cutoff])
        express_dis_exp = self.express_rbf_expansion(express_dis[spatial_dis <= self.cutoff])
        dis_concat = torch.cat([spatial_dis_exp, express_dis_exp], dim=1)
        potential = self.net(dis_concat)
        return potential
    def compute_distance(self, x):
        return torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + self.eps)

class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff
    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs

class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, cutoff_sr = 0.0, num_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff = cutoff
        self.cutoff_sr = cutoff_sr
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.alpha = 1.0
        self.cutoff_fn = CosineCutoff(cutoff)
        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)
    def _initial_params(self):
        if self.cutoff == 0:
            return torch.zeros(self.num_rbf), torch.tensor(0.0)
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        end_value = torch.exp(torch.scalar_tensor(-self.cutoff_sr))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (end_value - start_value)) ** -2] * self.num_rbf)
        return means, betas
    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)
    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)

class FNet_interaction(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation, num_rbf = 8, thre = 0.5, dim_reduce = False, spatial = False):
        super(FNet_interaction, self).__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.v_net = velocityNet(in_out_dim, hidden_dim, n_hiddens, activation, use_spatial = spatial)
        self.g_net = growthNet(in_out_dim, hidden_dim, activation)
        self.s_net = scoreNet(in_out_dim, hidden_dim, activation)
        self.d_net = indediffusionNet(in_out_dim, hidden_dim, activation)
        if spatial:
            self.interaction_net = InteractionModel_spatial(in_out_dim, hidden_dim, activation, num_rbf = num_rbf, cutoff = thre, dim_reduce = dim_reduce)
        else:
            self.interaction_net = InteractionModel(in_out_dim, hidden_dim, activation, num_rbf = num_rbf, cutoff = thre, dim_reduce = dim_reduce)
    def forward(self, t, z):
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)
            v = self.v_net(t, z).float()
            g = self.g_net(t, z).float()
            s = self.s_net(t, z).float()
            d = self.d_net(t, z).float()
        return v, g, s, d

class FNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation):
        super(FNet, self).__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.v_net = velocityNet(in_out_dim, hidden_dim, n_hiddens, activation)
        self.g_net = growthNet(in_out_dim, hidden_dim, activation)
        self.s_net = scoreNet(in_out_dim, hidden_dim, activation)
        self.d_net = indediffusionNet(in_out_dim, hidden_dim, activation)
    def forward(self, t, z):
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)
            v = self.v_net(t, z).float()
            g = self.g_net(t, z).float()
            s = self.s_net(t, z).float()
            d = self.d_net(t, z).float()
        return v, g, s, d

class ODEFunc2_interaction_energy(nn.Module):
    def __init__(self, f_net, use_mass = True, thre = 0.5, num_particles = 16):
        super(ODEFunc2_interaction_energy, self).__init__()
        self.f_net = f_net
        self.use_mass = use_mass
        self.thre = thre
        self.num_particles = num_particles
        self.interaction_potential=f_net.interaction_net
    def forward(self, t, state):
        z, lnw, _= state
        v, g, _, _ = self.f_net(t, z)
        dz_dt = v
        dlnw_dt = g
        net_force = cal_interaction(z, lnw, self.interaction_potential, threshold = self.thre, use_mass = self.use_mass, m = self.num_particles).float()
        w = torch.exp(lnw)
        if self.use_mass:
            dm_dt = (torch.norm(v, p=2,dim=1).unsqueeze(1)**2 + g**2) * w
        else:
            dm_dt = torch.norm(v, p=2,dim=1).unsqueeze(1)**2
        return dz_dt.float()+net_force.float(), dlnw_dt.float(), dm_dt.float()

class ODEFunc2_interaction(nn.Module):
    def __init__(self, f_net):
        super(ODEFunc2_interaction, self).__init__()
        self.f_net = f_net
        self.interaction_potential=f_net.interaction_net
    def forward(self, t, state):
        z, lnw,= state
        z.requires_grad_(True)
        lnw.requires_grad_(True)
        t.requires_grad_(True)
        v, g, _, _ = self.f_net(t, z)
        dz_dt = v
        dlnw_dt = g
        net_force = cal_interaction(z, lnw, self.interaction_potential).float()
        w = torch.exp(lnw)
        dm_dt = (torch.norm(v, p=2,dim=1).unsqueeze(1)**2 + g**2) * w
        return dz_dt.float()+net_force.float(), dlnw_dt.float()

class ODEFunc_interaction(nn.Module):
    def __init__(self, f_net):
        super(ODEFunc_interaction, self).__init__()
        self.v_net = f_net.v_net
        self.interaction_potential=f_net.interaction_net
    def forward(self, t, z):
        dz_dt = self.v_net(t, z)
        batch_size, embed_dim = z.shape
        device = z.device
        perm = torch.randperm(batch_size, device=device)
        z_shuffled = z[perm]
        if batch_size % 2 == 0:
            num_pairs = batch_size // 2
            num_triples = 0
        else:
            num_pairs = (batch_size - 3) // 2
            num_triples = 1
        net_force = torch.zeros_like(z)
        if num_pairs > 0:
            pairs_z = z_shuffled[:num_pairs * 2].view(num_pairs, 2, embed_dim)
            diff_pairs = pairs_z[:, 0] - pairs_z[:, 1]
            diff_pairs.requires_grad_(True)
            potentials_pairs = self.interaction_potential(diff_pairs)
            grad_potential = torch.autograd.grad(
                outputs=potentials_pairs,
                inputs=diff_pairs,
                grad_outputs=torch.ones_like(potentials_pairs),
                create_graph=True,
            )[0]
            force_pairs = -grad_potential
            force_assigned = torch.zeros(num_pairs * 2, embed_dim, device=device)
            force_assigned[0::2] = force_pairs
            force_assigned[1::2] = -force_pairs
            net_force[perm[:num_pairs * 2]] = force_assigned
        if num_triples > 0:
            triple_z = z_shuffled[num_pairs * 2:].view(1, 3, embed_dim)
            diff_triple = triple_z.unsqueeze(1) - triple_z.unsqueeze(0)
            diff_triple_flat = diff_triple.view(-1, embed_dim)
            diff_triple_flat.requires_grad_(True)
            potentials_triple = self.interaction_potential(diff_triple_flat)
            grad_potential = torch.autograd.grad(
                outputs=potentials_triple,
                inputs=diff_triple_flat,
                grad_outputs=torch.ones_like(potentials_triple),
                create_graph=True,
            )[0]
            force_triple_flat = -grad_potential
            force_matrix = force_triple_flat.view(3, 3, embed_dim)
            eye = torch.eye(3, device=device).unsqueeze(-1)
            force_matrix = force_matrix * (1 - eye)
            force_triple = force_matrix.sum(dim=1) * 2
            net_force[perm[num_pairs * 2:]] = force_triple
        return dz_dt.float()+net_force.float()

class ODEFunc2(nn.Module):
    def __init__(self, f_net, use_mass = True):
        super(ODEFunc2, self).__init__()
        self.f_net = f_net
        self.use_mass = use_mass
    def forward(self, t, state):
        z, lnw, _= state
        v, g, _, _ = self.f_net(t, z)
        dz_dt = v
        dlnw_dt = g
        w = torch.exp(lnw)
        if self.use_mass:
            dm_dt = (torch.norm(v, p=2,dim=1).unsqueeze(1)**2 + g**2) * w
        else:
            dm_dt = torch.norm(v, p=2,dim=1).unsqueeze(1)**2
        return dz_dt.float(), dlnw_dt.float(), dm_dt.float()

class ODEFunc(nn.Module):
    def __init__(self, v_net):
        super(ODEFunc, self).__init__()
        self.v_net = v_net
    def forward(self, t, z):
        dz_dt = self.v_net(t, z)
        return dz_dt.float()

class scoreNet2(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        state  = torch.cat((t,x),dim=1)
        return self.net(state)
    def compute_gradient(self, t, x):
        x = x.requires_grad_(True)
        output = self.forward(t, x)
        gradient = torch.autograd.grad(outputs=output, inputs=x,
                                       grad_outputs=torch.ones_like(output),
                                       create_graph=True)[0]
        return gradient

import torch
import torch.nn as nn

class scoreNet2_res(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, num_layers = 10, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        self.num_layers = num_layers
        self.input_layer = nn.Linear(in_out_dim + 1, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)
    def forward(self, t, x):
        state = torch.cat((t, x), dim=1)
        z = self.activation(self.input_layer(state))
        for i in range(self.num_layers):
            dz = self.activation(self.hidden_layers[i](z))
            z = z + dz
        out = self.output_layer(z)
        return out
    def compute_gradient(self, t, x):
        x = x.requires_grad_(True)
        output = self.forward(t, x)
        gradient = torch.autograd.grad(outputs=output, inputs=x,
                                       grad_outputs=torch.ones_like(output),
                                       create_graph=True)[0]
        return gradient

class ODEFunc3(nn.Module):
    def __init__(self, f_net,sf2m_score_model,sigma, use_mass, thre, num_particles):
        super(ODEFunc3, self).__init__()
        self.f_net = f_net
        self.interaction_potential = f_net.interaction_net
        self.sf2m_score_model = sf2m_score_model
        self.sigma=sigma
        self.use_mass = use_mass
        self.thre = thre
        self.num_particles = num_particles
    def forward(self, t, state):
        z, lnw, m = state
        w = torch.exp(lnw)
        z.requires_grad_(True)
        lnw.requires_grad_(True)
        m.requires_grad_(True)
        t.requires_grad_(True)
        v, g, _, _ = self.f_net(t, z)
        v.requires_grad_(True)
        g.requires_grad_(True)
        time=t.expand(z.shape[0],1)
        time.requires_grad_(True)
        s=self.sf2m_score_model(time,z)
        dz_dt = v
        dlnw_dt = g
        z=z.requires_grad_(True)
        grad_s = torch.autograd.grad(outputs=s, inputs=z,grad_outputs=torch.ones_like(s),create_graph=True)[0]
        norm_grad_s = torch.norm(grad_s, dim=1).unsqueeze(1).requires_grad_(True)
        net_force = cal_interaction(z, lnw, self.interaction_potential, threshold = self.thre, use_mass = self.use_mass, m = self.num_particles).float()
        if self.use_mass:
            if self.interaction_potential.cutoff != 0:
                dm_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) ** 2 / (2) + 
                        (norm_grad_s ** 2) / 2 + torch.norm(v, p=2, dim=1).unsqueeze(1) * torch.norm(grad_s, p=2, dim=1).unsqueeze(1) + g ** 2) * w
            else:
                dm_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) ** 2 / (2) + 
                 (norm_grad_s ** 2) / 2 -
                 (1 / 2 * self.sigma ** 2 *g + s* g) + g ** 2) * w
        else:
            dm_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) ** 2 / (2) + 
                    (norm_grad_s ** 2) / 2 + torch.norm(v, p=2, dim=1).unsqueeze(1) * torch.norm(grad_s, p=2, dim=1).unsqueeze(1))
        return dz_dt.float()+net_force.float(), dlnw_dt.float(), dm_dt.float()
