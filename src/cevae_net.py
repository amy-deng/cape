import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import bernoulli, normal
import sys
from torch import optim



class p_x_z(nn.Module):

    def __init__(self, dim_in=20, nh=3, dim_h=20, dim_out_bin=19, dim_out_con=6):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out_bin = dim_out_bin
        self.dim_out_con = dim_out_con

        # dim_in is dim of latent space z
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh-1)])
        # output layer defined separate for continuous and binary outputs
        self.output_bin = nn.Linear(dim_h, dim_out_bin)
        # for each output an mu and sigma are estimated
        self.output_con_mu = nn.Linear(dim_h, dim_out_con)
        self.output_con_sigma = nn.Linear(dim_h, dim_out_con)
        self.softplus = nn.Softplus()

    def forward(self, z_input):
        z = F.elu(self.input(z_input))
        for i in range(self.nh-1):
            z = F.elu(self.hidden[i](z))
        # for binary outputs:
        x_bin_p = torch.sigmoid(self.output_bin(z))
        x_bin = bernoulli.Bernoulli(x_bin_p)
        # for continuous outputs
        mu, sigma = self.output_con_mu(z), self.softplus(self.output_con_sigma(z))
        x_con = normal.Normal(mu, sigma)

        if (z != z).all():
            print(z,'z')
            raise ValueError('p(x|z) forward contains NaN')

        return x_bin, x_con


class p_t_z(nn.Module):

    def __init__(self, dim_in=20, nh=1, dim_h=20, dim_out=1):
        super().__init__()
        self.nh = nh
        self.dim_out = dim_out

        self.input = nn.Linear(dim_in, dim_h)
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # for binary outputs:
        out_p = torch.sigmoid(self.output(x))

        out = bernoulli.Bernoulli(out_p)
        return out


class p_y_zt(nn.Module):
    def __init__(self, dim_in=20, nh=3, dim_h=20, dim_out=1, bi_outcome=True):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out
        self.bi_outcome = bi_outcome
        # Separated forwards for different t values, TAR
        self.input_t0 = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden_t0 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.mu_t0 = nn.Linear(dim_h, dim_out)

        self.input_t1 = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden_t1 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.mu_t1 = nn.Linear(dim_h, dim_out)

    def forward(self, z, t):
        # Separated forwards for different t values, TAR
        x_t0 = F.elu(self.input_t0(z))
        for i in range(self.nh):
            x_t0 = F.elu(self.hidden_t0[i](x_t0))
        
        x_t1 = F.elu(self.input_t1(z))
        for i in range(self.nh):
            x_t1 = F.elu(self.hidden_t1[i](x_t1))
        
        if self.bi_outcome:
            mu_t0 = torch.sigmoid(self.mu_t0(x_t0))
            mu_t1 = torch.sigmoid(self.mu_t1(x_t1))
            tmp_v = torch.where(t>0,mu_t1,mu_t0)
            y = bernoulli.Bernoulli(tmp_v)
        else:
            mu_t0 = F.elu(self.mu_t0(x_t0))
            mu_t1 = F.elu(self.mu_t1(x_t1))
            # set mu according to t value
            y = normal.Normal((1-t)*mu_t0 + t * mu_t1, 1) #\hat v is fixed to 1 here
        return y


####### Inference model / Encoder #######

class q_t_x(nn.Module):

    def __init__(self, dim_in=25, nh=1, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of data x
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # for binary outputs:
        out_p = torch.sigmoid(self.output(x))
        out = bernoulli.Bernoulli(out_p)
        return out


class q_y_xt(nn.Module):

    def __init__(self, dim_in=25, nh=3, dim_h=20, dim_out=1, bi_outcome=True):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out
        self.bi_outcome = bi_outcome
        # dim_in is dim of data x
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        # separate outputs for different values of t
        self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)

    def forward(self, x, t):
        # Unlike model network, shared parameters with separated heads
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # only output weights separated
        mu_t0 = self.mu_t0(x)
        mu_t1 = self.mu_t1(x)
        # set mu according to t, sigma set to 1
        
        if self.bi_outcome:
            mu_t0 = torch.sigmoid(mu_t0)
            mu_t1 = torch.sigmoid(mu_t1)
            y = bernoulli.Bernoulli((1-t)*mu_t0 + t * mu_t1)
        else:
            y = normal.Normal((1-t)*mu_t0 + t * mu_t1, 1)
        return y


class q_z_tyx(nn.Module):
    def __init__(self, dim_in=25+1, nh=3, dim_h=20, dim_out=20):
        super().__init__()
        # dim in is dim of x + dim of y
        # dim_out is dim of latent space z
        # save required vars
        self.nh = nh
        # Shared layers with separated output layers
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])

        self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)
        self.sigma_t0 = nn.Linear(dim_h, dim_out)
        self.sigma_t1 = nn.Linear(dim_h, dim_out)
        self.softplus = nn.Softplus()

    def forward(self, xy, t):
        x = F.elu(self.input(xy)) # causal NaN
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        mu_t0 = self.mu_t0(x)
        mu_t1 = self.mu_t1(x)
        sigma_t0 = self.softplus(self.sigma_t0(x))
        sigma_t1 = self.softplus(self.sigma_t1(x))
        # Set mu and sigma according to t
        mean = (1-t)*mu_t0 + t * mu_t1
        std = (1-t)*sigma_t0 + t * sigma_t1
        z = normal.Normal(mean, std)
        return z

class CEVAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, binfeats, contfeats, device, bi_outcome=True):
        super().__init__()
        self.device = device
        self.binfeats = binfeats
        self.contfeats = contfeats
        x_dim = self.binfeats + self.contfeats
        self.p_x_z_dist = p_x_z(dim_in=z_dim, nh=1, dim_h=h_dim, dim_out_bin=self.binfeats,
                       dim_out_con=self.contfeats)#.to(args.device) # 3
        self.p_t_z_dist = p_t_z(dim_in=z_dim, nh=1, dim_h=h_dim, dim_out=1)#.to(args.device)
        self.p_y_zt_dist = p_y_zt(dim_in=z_dim, nh=1, dim_h=h_dim, dim_out=1, bi_outcome=bi_outcome)#.to(args.device)# 3
        self.q_t_x_dist = q_t_x(dim_in=x_dim, nh=1, dim_h=h_dim, dim_out=1)#.to(args.device)
        # t is not feed into network, therefore not increasing input size (y is fed).
        self.q_y_xt_dist = q_y_xt(dim_in=x_dim, nh=1, dim_h=h_dim, dim_out=1, bi_outcome=bi_outcome)#.to(args.device)# 3
        self.q_z_tyx_dist = q_z_tyx(dim_in=self.binfeats + self.contfeats + 1, nh=1, dim_h=h_dim,# 3
                            dim_out=z_dim)#.to(args.device)
        self.p_z_dist = normal.Normal(torch.zeros(z_dim).to(device), torch.ones(z_dim).to(device))


    def forward(self, x_train, y_train, t_train):
        xy = torch.cat((x_train, y_train), 1)
        z_infer = self.q_z_tyx_dist(xy=xy, t=t_train)
        z_infer_sample = z_infer.sample()
        if (z_infer_sample != z_infer_sample).all(): #NaN
            z_infer_sample = z_infer.rsample()
        else:
            pass
        # p(x|z)
        x_bin, x_con = self.p_x_z_dist(z_infer_sample)
        l1 = x_bin.log_prob(x_train[:, :self.binfeats]).sum(1)
        if self.contfeats:
            l2 = x_con.log_prob(x_train[:, -self.contfeats:]).sum(1)
        else:
            l2 = 0.
        # p(t|z)
        t = self.p_t_z_dist(z_infer_sample)
        l3 = t.log_prob(t_train).squeeze()
         # p(y|t,z)
        y = self.p_y_zt_dist(z_infer_sample, t_train)
        # print(y.mean)
        l4 = y.log_prob(y_train).squeeze()
        # REGULARIZATION LOSS
        # p(z) - q(z|x,t,y)
        l5 = (self.p_z_dist.log_prob(z_infer_sample) - z_infer.log_prob(z_infer_sample)).sum(1)
        # q(t|x)
        t_infer = self.q_t_x_dist(x_train)
        l6 = t_infer.log_prob(t_train).squeeze()
        # q(y|x,t)
        y_infer = self.q_y_xt_dist(x_train, t_train)
        l7 = y_infer.log_prob(y_train).squeeze()
        # print(l1.shape,'l1',l2.shape,'l2',l3.shape,'l3',l4.shape,'l4',l5.shape,'l5',l6.shape,'l6',l7.shape,'l7')
        loss_sum = -torch.mean(l1 + l2 + l3 + l4 + l5 + l6 + l7)
        # print(l1 + l2 + l3 + l4 + l5 + l6 + l7)
        y0, y1 = get_y0_y1(self.p_y_zt_dist, self.q_y_xt_dist, self.q_z_tyx_dist, x_train, t_train, self.device)
        y = torch.where(t_train.view(-1) > 0, y1.view(-1), y0.view(-1))
        return loss_sum, y, y0, y1

    def init_qz_func(self, y, t, x):
        self.q_z_tyx_dist = init_qz(self.q_z_tyx_dist, self.p_z_dist, y, t, x, self.device)



def init_qz(qz, pz, y,t,x, device):
    """
    Initialize qz towards outputting standard normal distributions
    - with standard torch init of weights the gradients tend to explode after first update step
    """
    # x = 
    idx = list(range(x.shape[0]))
    print(x.shape,y.shape,t.shape,len(idx))
    np.random.shuffle(idx)
    optimizer = optim.Adam(qz.parameters(), lr=0.001)

    for i in range(50):
        batch = np.random.choice(idx, 1)
        if str(device) == 'cpu':
            x_train, y_train, t_train = torch.FloatTensor(x[batch]), torch.FloatTensor(y[batch]), \
                                    torch.FloatTensor(t[batch])
        else:
            x_train, y_train, t_train = torch.cuda.FloatTensor(x[batch]), torch.cuda.FloatTensor(y[batch]), \
                                    torch.cuda.FloatTensor(t[batch])
        xy = torch.cat((x_train, y_train), 1)

        z_infer = qz(xy=xy, t=t_train)

        # KL(q_z|p_z) mean approx, to be minimized
        # KLqp = (z_infer.log_prob(z_infer.mean) - pz.log_prob(z_infer.mean)).sum(1)
        # Analytic KL
        KLqp = (-torch.log(z_infer.stddev) + 1/2*(z_infer.variance + z_infer.mean**2 - 1)).sum(1)

        objective = KLqp
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        if KLqp != KLqp:
            raise ValueError('KL(pz,qz) contains NaN during init')

    return qz


def get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, x_train, t_train, device, L=1):
    y_infer = q_y_xt_dist(x_train.float(), t_train.float())
    # use inferred y
    xy = torch.cat((x_train.float(), y_infer.mean), 1)  # TODO take mean?
    z_infer = q_z_tyx_dist(xy=xy, t=t_train.float())
    # Manually input zeros and ones
    y0 = p_y_zt_dist(z_infer.mean, torch.zeros(t_train.shape).to(device)).mean  # TODO take mean?
    y1 = p_y_zt_dist(z_infer.mean, torch.ones(t_train.shape).to(device)).mean  # TODO take mean?
    return y0, y1


class MultiTaskCEVAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, binfeats, contfeats, device, bi_outcome=True):
        super().__init__()
        self.models = nn.ModuleList([CEVAE(x_dim, h_dim, z_dim, binfeats, contfeats, device, bi_outcome) for i in range(20)])
        self.device = device

    def forward(self,x,y,t):
        batch_size, seq_len, n_loc, n_feat = x.size()
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(batch_size * n_loc, -1)
        y = y.view( batch_size * n_loc, 1)
        t = t.view(batch_size * n_loc, 20,1)
        total_loss = 0
        for i in range(len(self.models)):
            loss, y_pred, y0, y1 = self.models[i](x, y, t[:,i])
            total_loss += loss
        total_loss = total_loss/20
        return total_loss, y_pred, y0, y1

    def init(self,x,y,t):
        batch_size, seq_len, n_loc, n_feat = x.size()
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(batch_size * n_loc, -1)
        y = y.view( batch_size * n_loc, 1)
        t = t.view(batch_size * n_loc, 20)
        for i in range(len(self.models)):
            self.models[i].q_z_tyx_dist = init_qz(self.models[i].q_z_tyx_dist, self.models[i].p_z_dist, y, t[:,i], x, self.device)
