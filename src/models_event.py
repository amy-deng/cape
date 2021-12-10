import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import torch
import torch.nn.functional as F
from torch.distributions import bernoulli, normal
from layers import *
from utils import *
import math
 
 
 
 
class cape_cau(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.data = data
        self.p1 = args.p1
        self.geo_mx = data.geo_mx.to(args.device)
        self.n_treatment = 1
        self.balance = args.balance
        self.i_t = args.i_t
        self.h_dim = args.h_dim 
        self.device = args.device 
        device = args.device
        layers=3
        self.layers = layers
        residual_channels=args.h_dim // 2
        dilation_channels=args.h_dim // 2
        skip_channels=args.h_dim // 2
        end_channels=args.h_dim // 2
        kernel_size=2
        out_dim = 1
        in_dim = data.f
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        num_nodes = data.m
        self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.task_dim = args.h_dim // 2

        self.task_embedding = nn.Parameter(torch.Tensor(self.task_dim,1))
        nn.init.xavier_uniform_(self.task_embedding, gain=nn.init.calculate_gain('relu'))
        
        self.supports_len =1
        receptive_field = 1
        additional_scope = kernel_size - 1
        new_dilation = 1
        for i in range(layers):
            self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                out_channels=dilation_channels,
                                                kernel_size=(1,kernel_size),dilation=new_dilation))

            self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                out_channels=dilation_channels,
                                                kernel_size=(1, kernel_size), dilation=new_dilation))

            self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                out_channels=skip_channels,
                                                kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channels))
            new_dilation *=2
            receptive_field += additional_scope
            additional_scope *= 2
        self.gconv = gcn(dilation_channels,residual_channels,args.dropout,support_len=self.supports_len)

        self.hyp_layer0 = nn.Sequential(
                            nn.Conv2d(in_channels=skip_channels+self.task_dim,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        )
        self.hyp_layer1 = nn.Sequential(
                            nn.Conv2d(in_channels=skip_channels+self.task_dim,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        )
        self.receptive_field = receptive_field
        self.p = args.p
        self.wl = args.wl 
        self.n_c = args.n_c
        self.dropout = nn.Dropout(args.dropout)
        self.criterion = F.binary_cross_entropy
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) 
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, X, A, C, Y, Cc):
        inputs = X
        inputs = inputs.permute(0,3,2,1).contiguous()
        bs, features, n_nodes, n_timesteps = inputs.size()
        treatment = C.view(bs * n_nodes)
        in_len = inputs.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(inputs,(self.receptive_field-in_len,0,0,0))
        else:
            x = inputs
        x = self.start_conv(x)
        skip = 0
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        new_supports = self.supports + [adp]
        skip = 0
        for i in range(self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        x = self.gconv(skip, new_supports) 
        task_embedding = self.task_embedding.view(1, -1, 1, 1).repeat(bs, 1, n_nodes, 1)
        x = F.relu(x)
        z = torch.cat((task_embedding,x),1)
        y1 = self.hyp_layer1(z)
        y0 = self.hyp_layer0(z)
        y1 = y1.squeeze(-1).squeeze(1).view(-1)
        y0 = y0.squeeze(-1).squeeze(1).view(-1)
        y1 = torch.sigmoid(y1)
        y0 = torch.sigmoid(y0)
        y = torch.where(treatment > 0, y1, y0)
        Y = Y.view(-1)
        loss = self.criterion(y, Y) 
        z_balance = z.permute(0,2,1,3).contiguous().view(bs*n_nodes,-1)
        if self.balance == 'wass':
            imb, _ = wasserstein_ht(z_balance, treatment, self.p, device=self.device)
        else:
            imb = mmd2_lin(z_balance,treatment,self.p)
        loss += self.p1 * imb
        return loss, y, y0, y1 
  
# causal model + prediction model 
class cape(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.data = data
        self.n_c = args.n_c 
        self.balance = args.balance
        self.p1 = args.p1
        self.device = args.device 
        self.h_dim = args.h_dim
        device = args.device
        layers=3
        self.layers = layers
        residual_channels=args.h_dim // 2
        dilation_channels=args.h_dim // 2
        skip_channels=args.h_dim // 2
        end_channels=args.h_dim // 2
        kernel_size=2
        out_dim = 1
        in_dim = data.f
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        num_nodes = data.m
        self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.task_dim = args.h_dim // 2

        self.task_embedding = nn.Parameter(torch.Tensor(self.task_dim,self.n_c))
        nn.init.xavier_uniform_(self.task_embedding, gain=nn.init.calculate_gain('relu'))
        
        self.supports_len =1
        receptive_field = 1
        additional_scope = kernel_size - 1
        new_dilation = 1
        for i in range(layers):
            self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                out_channels=dilation_channels,
                                                kernel_size=(1,kernel_size),dilation=new_dilation))

            self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                out_channels=dilation_channels,
                                                kernel_size=(1, kernel_size), dilation=new_dilation))

            self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                out_channels=skip_channels,
                                                kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channels))
            new_dilation *=2
            receptive_field += additional_scope
            additional_scope *= 2
        self.gconv = gcn(dilation_channels,residual_channels,args.dropout,support_len=self.supports_len)

        self.hyp_layers0 = nn.ModuleList()
        self.hyp_layers1 = nn.ModuleList()
        self.dropout = nn.Dropout(args.dropout)
        for i in range(args.n_c):
            self.hyp_layers0.append(nn.Sequential(
                            nn.Conv2d(in_channels=skip_channels+self.task_dim,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)))
            self.hyp_layers1.append(nn.Sequential(
                            nn.Conv2d(in_channels=skip_channels+self.task_dim,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)))
        self.receptive_field = receptive_field
        self.u = args.u
        self.feat_map = PositionwiseFeedForward(data.f,self.h_dim)
        self.ite_map = nn.Linear(self.n_c,self.n_c)
        self.layer_norm2 = nn.LayerNorm(data.f)#,elementwise_affine=False)
        self.base_model = args.base_model
        if args.base_model == 'cola':
            self.pred_net = COLA(args,data)
        elif args.base_model == 'gwnet':
            self.pred_net = gwnet(args,data)
        self.criterion =  F.binary_cross_entropy_with_logits
        self.criterion_no_sig =  F.binary_cross_entropy

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) 
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, X, A, C, Y, C_vec):
        treatment_vec, Z, Ypred, _Y, Y0, Y1 = self.potential_outcome(X, A, C, Y, C_vec)
        emb = Z
        treat = treatment_vec
        if self.balance == 'wass':
            imb, _ = wasserstein_ht(emb, treat, 0.5, device=self.device)
        else:
            imb = mmd2_lin(emb,treat,0.5)
        loss = self.criterion(Ypred, _Y)
        loss += self.p1 * imb
        return loss, Y0, Y1

    def potential_outcome(self, X, A, C, Y, C_vec):
        inputs = X
        inputs = inputs.permute(0,3,2,1).contiguous()
        bs, features, n_nodes, n_timesteps = inputs.size()
        treatment_vec = C_vec.view(bs * n_nodes * self.n_c)
        in_len = inputs.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(inputs,(self.receptive_field-in_len,0,0,0))
        else:
            x = inputs
        x = self.start_conv(x)
        skip = 0
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        new_supports = self.supports + [adp]
        x_blocks = []
        skip = 0
        for i in range(self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        x = self.gconv(skip, new_supports)  #[bs, f, nloc, 1])
        x = x.repeat(1,1,1,self.n_c)

        task_embedding = self.task_embedding.view(1, -1, 1, self.n_c).repeat(bs, 1, n_nodes, 1)

        x = F.relu(x)
        z = torch.cat((task_embedding,x),1)
        pred0, pred1 = [], []
        for i in range(self.n_c):
            z_task = z[:,:,:,i:i+1]
            y1 = self.hyp_layers1[i](z_task)
            y0 = self.hyp_layers0[i](z_task)
            pred0.append(y0)
            pred1.append(y1)
        Y0 = torch.cat(pred0,-1).view(-1) 
        Y1 = torch.cat(pred1,-1).view(-1) 
        z = z.permute(0,2,3,1).contiguous().view(bs*n_nodes*self.n_c,-1)
        Ypred = torch.where(treatment_vec > 0, Y1, Y0) # event prediction should not based on this   
        Y = Y.view(bs*n_nodes,1).repeat(1,self.n_c).view(-1)
        return treatment_vec, z, Ypred, Y, Y0, Y1 
 
    def forward_event(self, X, A, C, Y, C_vec): 
        with torch.no_grad():
            treatment_vec, Z, Ypred, _Y, Y0, Y1  = self.potential_outcome(X, A, C, Y, C_vec)
        batch_size, seq_len, n_loc, n_feat = X.size()
        Y0 = Y0.view(batch_size, 1, n_loc, self.n_c)
        Y1 = Y1.view(batch_size, 1, n_loc, self.n_c)

        x = X.view(batch_size*seq_len,n_loc,n_feat)
        x_orig = x
        x = self.feat_map(x)
        x = x.view(X.size())
        x_orig = x_orig.view(X.size())
        Y1 = torch.sigmoid(Y1)
        Y0 = torch.sigmoid(Y0)
        ite = Y1-Y0
        ite = self.ite_map(ite)
        ite = ite.repeat(1,seq_len,1,1)

        ite = torch.cat((ite,torch.zeros(batch_size, seq_len, n_loc, n_feat-self.n_c).to(self.device)),dim=-1)
        w_ite = torch.sigmoid(ite)
        x = self.layer_norm2(x * w_ite + x_orig)
        
        loss, y  = self.pred_net(x, Y) 
        y = y.view(batch_size,1, n_loc, 1)
        y_repeat = y.repeat(1,1,1,self.n_c)
        max_y = torch.max(Y1, Y0)
        min_y = torch.min(Y1, Y0)
        loss_constrain = torch.relu(min_y-y_repeat) + torch.relu(y_repeat-max_y)
        loss += self.u * loss_constrain.mean()
        return loss, y 


'''base models'''

''' spatial temporal models '''
class COLA(nn.Module):  
    def __init__(self, args, data): 
        super().__init__()
        self.input_dim = data.f 
        self.m = data.m  
        self.w = args.window
        self.h = args.horizon
        self.adj = data.geo_mx.to_dense().to(args.device)
        self.dropout = args.dropout
        self.n_hidden = args.h_dim
        half_hid = int(self.n_hidden/2)
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.b1 = Parameter(torch.Tensor(half_hid))
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.act = F.elu 
        self.Wb = Parameter(torch.Tensor(self.m,self.m))
        self.wb = Parameter(torch.Tensor(1))
        self.k = 10
        self.conv = nn.Conv1d(data.f, self.k, self.w)
        long_kernal = self.w//2
        self.conv_long = nn.Conv1d(data.f, self.k, long_kernal, dilation=2)
        long_out = self.w-2*(long_kernal-1)
        self.n_spatial = 10  
        self.conv1 = GraphConvLayer((1+long_out)*self.k, self.n_hidden) # self.k
        self.conv2 = GraphConvLayer(self.n_hidden, self.n_spatial)
        self.rnn = nn.GRU(input_size=self.input_dim, hidden_size=self.n_hidden, num_layers=1, dropout=args.dropout, batch_first=True, bidirectional=False)
     
        hidden_size = 1 * self.n_hidden
        self.out = nn.Linear(hidden_size + self.n_spatial, 1)  

        self.residual_window = 0
        self.ratio = 1.0
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1) 

        self.criterion = F.binary_cross_entropy_with_logits
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, X, Y):
        '''
        Args:  x: (batch, time_step, m, f)  
            feat: [batch, window, dim, m]
        Returns: (batch, m)
        ''' 
        b, w, m, f = X.size()
        orig_x = X 
        x = X.permute(0, 2, 1, 3).contiguous().view(-1, X.size(1), f) 
        r_out, hc = self.rnn(x, None)
        last_hid = r_out[:,-1,:]
        last_hid = last_hid.view(-1,self.m, self.n_hidden)
        out_temporal = last_hid  # [b, m, 20]
        hid_rpt_m = last_hid.repeat(1,self.m,1).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous m
        hid_rpt_w = last_hid.repeat(1,1,self.m).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous w one window data
        a_mx = self.act( hid_rpt_m @ self.W1.t()  + hid_rpt_w @ self.W2.t() + self.b1 ) @ self.V + self.bv # row, all states influence one state 
        a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12, out=None)
        r_l = []
        r_long_l = []
        h_mids = orig_x
        for i in range(self.m):
            h_tmp = h_mids[:,:,i].permute(0,2,1).contiguous() 
            r = self.conv(h_tmp) # [32, 10/k, 1]
            r_long = self.conv_long(h_tmp)
            r_l.append(r)
            r_long_l.append(r_long)
        r_l = torch.stack(r_l,dim=1)
        r_long_l = torch.stack(r_long_l,dim=1)
        r_l = torch.cat((r_l,r_long_l),-1)
        r_l = r_l.view(r_l.size(0),r_l.size(1),-1)
        r_l = torch.relu(r_l)
        adjs = self.adj.repeat(b,1)
        adjs = adjs.view(b,self.m, self.m)
        c = torch.sigmoid(a_mx @ self.Wb + self.wb)
        a_mx = adjs * c + a_mx * (1-c) 
        adj = a_mx 
        x = r_l  
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        out_spatial = F.relu(self.conv2(x, adj))
        out = torch.cat((out_spatial, out_temporal),dim=-1)
        out = self.out(out)
        out = torch.squeeze(out)

        if (self.residual_window > 0):
            z = orig_x[:, -self.residual_window:, :]; #Step backward # [batch, res_window, m]
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window); #[batch*m, res_window]
            z = self.residual(z); #[batch*m, 1]
            z = z.view(-1,self.m); #[batch, m]
            out = out * self.ratio + z; #[batch, m]
        Y = Y.view(-1)
        y = out.view(-1)
        loss = self.criterion(y, Y) 
        y = torch.sigmoid(y)
        return loss, y
        

''' GWNET Graph WaveNet for Deep Spatial-Temporal Graph Modeling '''
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gwnet(nn.Module):
    def __init__(self, args, data, in_feat=0):
        super(gwnet, self).__init__()
        device = args.device
        supports=None 
        gcn_bool=True
        addaptadj=True
        aptinit=None
        if in_feat == 0:
            in_dim=data.f
        else:
            in_dim = in_feat
        out_dim = 1
        residual_channels=args.h_dim // 4
        dilation_channels=args.h_dim // 4
        skip_channels=args.h_dim // 4
        end_channels=args.h_dim // 4
        kernel_size=2
        blocks=4
        layers=4
        dropout = args.dropout
        self.dropout = args.dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = True
        num_nodes = data.m

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.criterion = F.binary_cross_entropy_with_logits

    def forward(self, inputs, Y):
        # inputs shape is (bs, features, n_nodes, n_timesteps)
        #  [32, 20, 47] b,w,m,f
        inputs = inputs.permute(0,3,2,1).contiguous()
        in_len = inputs.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(inputs,(self.receptive_field-in_len,0,0,0))
        else:
            x = inputs
        x = self.start_conv(x)
        skip = 0
        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]
        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*inputs*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        out = x.squeeze(-1).squeeze(1)
        y = out.view(-1)
        Y = Y.view(-1)
        loss = self.criterion(y, Y) 
        y = torch.sigmoid(y)
        return loss, y 
        

 