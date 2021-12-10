import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn.functional as F
from layers import *
from utils import *
 
 
class GCN_DECONF(nn.Module):
    def __init__(self, args, nfeat, nhid, dropout, n_in=1, n_out=1, cuda=False, binary=True):
        super(GCN_DECONF, self).__init__()
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.use_cuda = cuda
        self.p1 = args.p1
        nfeat = nfeat*args.window
        if cuda:
            self.gc = [GraphConvLayer(nfeat, nhid).cuda()]
            for i in range(n_in - 1):
                self.gc.append(GraphConvLayer(nhid, nhid).cuda())
        else:
            self.gc = [GraphConvLayer(nfeat, nhid)]
            for i in range(n_in - 1):
                self.gc.append(GraphConvLayer(nhid, nhid))
        
        self.n_in = n_in
        self.n_out = n_out

        if cuda:
            self.out_t00 = [nn.Linear(nhid,nhid).cuda() for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid,nhid).cuda() for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid,1).cuda()
            self.out_t11 = nn.Linear(nhid,1).cuda()
        else:
            self.out_t00 = [nn.Linear(nhid,nhid) for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid,nhid) for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid,1)
            self.out_t11 = nn.Linear(nhid,1)

        self.dropout = dropout

        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, 1)

        if cuda:
            self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()
        self.binary = binary
        if binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # xavier_normal xavier_uniform_
            else:
                # nn.init.zeros_(p.data)
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, t, Y, idx_list=None):
        # print(x.shape, adj.shape, t.shape,'+++++++++++')
        batch_size, seq_len, n_loc, n_feat = x.size()
        # X = X.view(X.size(0), -1)
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(batch_size, n_loc, -1)
        adj = adj[:,0]
        # print(x.shape,adj.shape)
        rep = F.relu(self.gc[0](x, adj))
        rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep, adj))
            rep = F.dropout(rep, self.dropout, training=self.training)

        for i in range(self.n_out):
            y00 = F.relu(self.out_t00[i](rep))
            y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep))
            y10 = F.dropout(y10, self.dropout, training=self.training)
        
        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)
        y = torch.where(t > 0,y1,y0)
        p1 = self.pp_act(self.pp(rep)).view(-1)

        rep = rep.view(-1,rep.size(-1))
        if self.use_cuda:
            rep_t1, rep_t0 = rep[(t > 0).nonzero().cuda()], rep[(t < 1).nonzero().cuda()]
        else:
            rep_t1, rep_t0 = rep[(t > 0).nonzero()], rep[(t < 1).nonzero()]
        try:
            dist, _ = wasserstein(rep_t1, rep_t0, cuda=self.use_cuda) # torch.Size([942, 1, 128]) torch.Size([1232, 1, 128]
        except:
            dist = 0.

        if idx_list is not None:
            y0 = y0[idx_list]
            y1 = y1[idx_list]
            y = y[idx_list]
            Y = Y.view(-1)[idx_list]
        loss = self.criterion(y, Y.view(-1), reduction='mean') + self.p1 * dist
        if self.binary:
            y = torch.sigmoid(y)
            y0 = torch.sigmoid(y0)
            y1 = torch.sigmoid(y1)
        return loss, y, y0, y1
 
class TARNet(nn.Module): 
    def __init__(self, args, in_feat, rep_hid, hyp_hid, rep_layer=2, hyp_layer=2, binary=True, p=0.5, dropout=0.2, device=torch.device('cpu')): 
        super().__init__() 
        self.p = p
        self.device = device
        self.hyp_layer = hyp_layer
        self.rep_layer_fst = nn.Linear(in_feat*args.window, rep_hid)
        self.rep_bn_fst = nn.BatchNorm1d(rep_hid)
    
        self.rep_layers = nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(rep_layer-1)])
        self.rep_bns = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(rep_layer-1)])

        self.hyp_layer_fst0 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst0 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-2)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-2)])
        self.hyp_out0 = nn.Linear(rep_hid, 1) 

        self.hyp_layer_fst1 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst1 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers1= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-2)])
            self.hyp_bns1 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-2)])
        self.hyp_out1 = nn.Linear(rep_hid, 1) 
        self.dropout = nn.Dropout(p=dropout)
        self.binary = binary
        if binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss
  
    def forward(self, X, C, Y): 
        batch_size, seq_len, n_loc, n_feat = X.size()
        # X = X.view(X.size(0), -1)
        X = X.permute(0,2,1,3).contiguous()
        X = X.view(batch_size * n_loc, -1)
        X = X.view(-1, X.size(-1))
        Y = Y.view(-1)
        h = self.dropout(F.relu(self.rep_bn_fst(self.rep_layer_fst(X))))
        for fc, bn in zip(self.rep_layers, self.rep_bns):
            h = self.dropout(F.relu(bn(fc(h))))

        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))

        h1 = self.dropout(F.relu(self.hyp_bn_fst1(self.hyp_layer_fst1(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers1, self.hyp_bns1):
                h1 = self.dropout(F.relu(bn(fc(h1))))

        y0 = self.hyp_out0(h0).view(-1)
        y1 = self.hyp_out1(h1).view(-1)
        C_1d = C.view(-1)
        y = torch.where(C_1d > 0, y1, y0)
        loss = self.criterion(y, Y, reduction='none')
        weight = C_1d/(2*self.p) + (1-C_1d)/(2*(1-self.p))
        loss = torch.mean(loss * weight)
        if self.binary:
            y = torch.sigmoid(y)
            y0 = torch.sigmoid(y0)
            y1 = torch.sigmoid(y1)
        return loss, y, y0, y1 

class CFR_MMD(nn.Module): 
    def __init__(self, args, in_feat, rep_hid, hyp_hid, rep_layer=2, hyp_layer=2, binary=True, p=0.5, dropout=0.2, device=torch.device('cpu')): 
        super().__init__() 
        self.p = p
        self.p1 = args.p1
        self.device = device
        self.hyp_layer = hyp_layer
        self.rep_layer_fst = nn.Linear(in_feat*args.window, rep_hid)
        self.rep_bn_fst = nn.BatchNorm1d(rep_hid)
    
        self.rep_layers = nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(rep_layer-1)])
        self.rep_bns = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(rep_layer-1)])

        self.hyp_layer_fst0 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst0 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-2)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-2)])
        self.hyp_out0 = nn.Linear(rep_hid, 1) 

        self.hyp_layer_fst1 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst1 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers1= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-2)])
            self.hyp_bns1 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-2)])
        self.hyp_out1 = nn.Linear(rep_hid, 1) 
        self.dropout = nn.Dropout(p=dropout)
        self.binary = binary
        if binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss
  
    def forward(self, X, C, Y): 
        batch_size, seq_len, n_loc, n_feat = X.size()
        # X = X.view(X.size(0), -1)
        X = X.permute(0,2,1,3).contiguous()
        X = X.view(batch_size * n_loc, -1)
        Y = Y.view(-1)

        h = self.dropout(F.relu(self.rep_bn_fst(self.rep_layer_fst(X))))
        for fc, bn in zip(self.rep_layers, self.rep_bns):
            h = self.dropout(F.relu(bn(fc(h))))

        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))

        h1 = self.dropout(F.relu(self.hyp_bn_fst1(self.hyp_layer_fst1(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers1, self.hyp_bns1):
                h1 = self.dropout(F.relu(bn(fc(h1))))

        y0 = self.hyp_out0(h0).view(-1)
        y1 = self.hyp_out1(h1).view(-1)
 
        # try:
            # imb = mmd2_rbf(h,C,self.p)
        imb = mmd2_lin(h,C,self.p)
        # except:
            # imb = 0.
        # print(imb)
        C_1d = C.view(-1)
        weight = C_1d/(2*self.p) + (1-C_1d)/(2*(1-self.p))
        # weight = weight.to(self.device)
        y = torch.where(C_1d > 0, y1, y0)
        loss = self.criterion(y, Y, reduction='none')
        loss = torch.mean(loss * weight) + self.p1*imb
        if self.binary:
            y = torch.sigmoid(y)
            y0 = torch.sigmoid(y0)
            y1 = torch.sigmoid(y1)
        return loss, y, y0, y1 

class CFR_WASS(nn.Module): 
    def __init__(self, args, in_feat, rep_hid, hyp_hid, rep_layer=2, hyp_layer=2, binary=True, p=0.5, dropout=0.2, device=torch.device('cpu')): 
        super().__init__() 
        self.p = p
        self.p1 = args.p1
        self.device = device
        self.hyp_layer = hyp_layer
        self.rep_layer_fst = nn.Linear(in_feat*args.window, rep_hid)
        self.rep_bn_fst = nn.BatchNorm1d(rep_hid)
    
        self.rep_layers = nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(rep_layer-1)])
        self.rep_bns = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(rep_layer-1)])

        self.hyp_layer_fst0 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst0 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-2)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-2)])
        self.hyp_out0 = nn.Linear(rep_hid, 1) 

        self.hyp_layer_fst1 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst1 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers1= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-2)])
            self.hyp_bns1 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-2)])
        self.hyp_out1 = nn.Linear(rep_hid, 1) 
        self.dropout = nn.Dropout(p=dropout)
        self.binary = binary
        if binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss
  
    def forward(self, X, C, Y): 
        batch_size, seq_len, n_loc, n_feat = X.size()
        # X = X.view(X.size(0), -1)
        X = X.permute(0,2,1,3).contiguous()
        X = X.view(batch_size * n_loc, -1)
        Y = Y.view(-1)
        h = self.dropout(F.relu(self.rep_bn_fst(self.rep_layer_fst(X))))
        for fc, bn in zip(self.rep_layers, self.rep_bns):
            h = self.dropout(F.relu(bn(fc(h))))

        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))

        h1 = self.dropout(F.relu(self.hyp_bn_fst1(self.hyp_layer_fst1(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers1, self.hyp_bns1):
                h1 = self.dropout(F.relu(bn(fc(h1))))

        y0 = self.hyp_out0(h0).view(-1)
        y1 = self.hyp_out1(h1).view(-1)
        # try:
        imb, _ = wasserstein_ht(h,C,self.p,device=self.device)
        # except:
        #     imb = 0.
        # print(imb)
        C_1d = C.view(-1)
        y = torch.where(C_1d > 0, y1, y0)
        weight = C_1d/(2*self.p) + (1-C_1d)/(2*(1-self.p))
        # weight = weight.to(self.device)
        loss = self.criterion(y, Y, reduction='none')
        loss = torch.mean(loss * weight) + self.p1*imb
        if self.binary:
            y = torch.sigmoid(y)
            y0 = torch.sigmoid(y0)
            y1 = torch.sigmoid(y1)
        return loss, y, y0, y1 

class PDDM(nn.Module):
    def __init__(self, h_dim, dropout=0.2): 
        super().__init__() 
        self.nn_u = nn.Linear(h_dim,h_dim,bias=True) 
        self.nn_v = nn.Linear(h_dim,h_dim,bias=True) 
        self.nn_c = nn.Linear(h_dim*2,h_dim,bias=True) 
        self.bn = nn.LayerNorm(h_dim*2)
        self.nn_s = nn.Linear(h_dim,1,bias=False) 
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,z1,z2):
        u = torch.abs(z1-z2)
        v = torch.abs(z1+z2)/2.0
        u = u / (torch.norm(u,p=2)+1e-7)
        v = v / (torch.norm(v,p=2)+1e-7)
        # print(u.shape,'u',v.shape,'v')
        u1 = torch.relu(self.nn_u(u))
        v1 = torch.relu(self.nn_v(v))
        uv = torch.cat((u1,v1),dim=-1) # size h_dim*2
        uv = self.bn(uv)
        uv = self.dropout(uv)
        h = torch.relu(self.nn_c(uv))
        hat_S = self.nn_s(h)
        # print(hat_S.shape,'hat s')
        return hat_S.double()

 
def func_PDDM_dis(a,b):
    r = 0.75*torch.abs((a+b)/2.0-0.5) - torch.abs((a-b)/2.0) + 0.5
    return r.double()

def func_MPDM(zi,zm,zj,zk):
    tmp = ((zi+zm)/2.0-(zj+zk)/2.0)
    r = torch.pow(2, tmp).sum()
    return r.double()

class SITE(nn.Module):
    def __init__(self, args, in_feat, rep_hid, hyp_hid, rep_layer=2, hyp_layer=2, binary=True, dropout=0.2): 
        super().__init__() 
        self.p1 = args.p1
        self.p2 = args.p2
        self.hyp_layer = hyp_layer
        self.rep_layer_fst = nn.Linear(in_feat*args.window, rep_hid)
        self.rep_bn_fst = nn.BatchNorm1d(rep_hid)
    
        self.rep_layers = nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(rep_layer-1)])
        self.rep_bns = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(rep_layer-1)])

        self.hyp_layer_fst0 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst0 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers0= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-2)])
            self.hyp_bns0 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-2)])
        self.hyp_out0 = nn.Linear(rep_hid, 1) 

        self.hyp_layer_fst1 = nn.Linear(rep_hid, rep_hid)
        self.hyp_bn_fst1 = nn.BatchNorm1d(rep_hid)
        if hyp_layer > 2:
            self.hyp_layers1= nn.ModuleList([nn.Linear(rep_hid, rep_hid) for i in range(hyp_layer-2)])
            self.hyp_bns1 = nn.ModuleList([nn.BatchNorm1d(rep_hid) for i in range(hyp_layer-2)])
        self.hyp_out1 = nn.Linear(rep_hid, 1) 
        self.dropout = nn.Dropout(p=dropout)
  
        self.PDDM = PDDM(rep_hid,dropout) 
        self.bn = nn.BatchNorm1d(rep_hid)
        self.dropout = nn.Dropout(p=dropout)
        self.binary = binary
        if binary:
            self.criterion = F.binary_cross_entropy_with_logits
        else:
            self.criterion = F.mse_loss
  
    def forward(self, X, C, P, Y): 
        batch_size, seq_len, n_loc, n_feat = X.size()
        # X = X.view(X.size(0), -1)
        X = X.permute(0,2,1,3).contiguous()
        X = X.view(batch_size * n_loc, -1)
        Y = Y.view(-1)
        C = C.view(-1)
        P = P.view(-1)
        # print(X.shape,C.shape,P.shape,Y.shape)
        I_t = (C>0).nonzero().view(-1)#torch.where(C>0)[0]
        I_c = (C<1).nonzero().view(-1)#torch.where(C<1)[0]
        t_idx_map = dict(zip(range(len(I_t)),I_t.data.cpu().numpy()))
        c_idx_map = dict(zip(range(len(I_c)),I_c.data.cpu().numpy()))
        # print('c_idx_map',c_idx_map)
        # print(I_t.shape,I_c.shape,'I_c')
        prop_t = P[I_t].data.cpu()
        prop_c = P[I_c].data.cpu()
        # find x_i, x_j 
        index_i, index_j = find_middle_pair(prop_t, prop_c)
        # print('index_i, index_j',index_i, index_j)
        # find x_k, x_l
        index_k = torch.argmax(torch.abs(prop_c - prop_t[index_i])).item()
        index_l = find_nearest_point(prop_c, prop_c[index_k])
        # print('index_k, index_l',index_k, index_l)
        # find x_n, x_m
        index_m = torch.argmax(np.abs(prop_t - prop_c[index_j])).item()
        index_n = find_nearest_point(prop_t, prop_t[index_m,])
        # print('index_m, index_n',index_m, index_n)
        index_i = t_idx_map[index_i]
        index_j = c_idx_map[index_j]
        index_k = c_idx_map[index_k]
        index_l = c_idx_map[index_l]
        index_m = t_idx_map[index_m]
        index_n = t_idx_map[index_n]

        h = self.dropout(F.relu(self.rep_bn_fst(self.rep_layer_fst(X))))
        for fc, bn in zip(self.rep_layers, self.rep_bns):
            h = self.dropout(F.relu(bn(fc(h))))
        z = h
        h0 = self.dropout(F.relu(self.hyp_bn_fst0(self.hyp_layer_fst0(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers0, self.hyp_bns0):
                h0 = self.dropout(F.relu(bn(fc(h0))))

        h1 = self.dropout(F.relu(self.hyp_bn_fst1(self.hyp_layer_fst1(h))))
        if self.hyp_layer > 2:
            for fc, bn in zip(self.hyp_layers1, self.hyp_bns1):
                h1 = self.dropout(F.relu(bn(fc(h1))))

        y0 = self.hyp_out0(h0).view(-1)
        y1 = self.hyp_out1(h1).view(-1)
 
        y = torch.where(C > 0, y1, y0)
        loss_factual = self.criterion(y, Y)
        func_dis_los = nn.MSELoss(reduction='none')
        hat_S_kl = self.PDDM(z[index_k],z[index_l])
        S_kl = func_PDDM_dis(P[index_k],P[index_l]) 
        hat_S_mn = self.PDDM(z[index_m],z[index_n])
        S_mn = func_PDDM_dis(P[index_m],P[index_n]) 
        hat_S_km = self.PDDM(z[index_k],z[index_m])
        S_km = func_PDDM_dis(P[index_k],P[index_m]) 
        hat_S_im = self.PDDM(z[index_i],z[index_m])
        S_im = func_PDDM_dis(P[index_i],P[index_m]) 
        hat_S_jk = self.PDDM(z[index_j],z[index_k])
        S_jk = func_PDDM_dis(P[index_j],P[index_k]) 
        loss_PDDM = 0.2 * torch.sum(func_dis_los(hat_S_kl,S_kl) + func_dis_los(hat_S_mn,S_mn) + func_dis_los(hat_S_km,S_km) + func_dis_los(hat_S_im,S_im) + func_dis_los(hat_S_jk, S_jk))
        # print('loss_PDDM',loss_PDDM,type(loss_PDDM))
        loss_MPDM = func_MPDM(z[index_i],z[index_m],z[index_j],z[index_k])
        # print('loss_MPDM',loss_MPDM,type(loss_MPDM))
        loss = loss_factual + self.p1*loss_PDDM + self.p2*loss_MPDM
        if self.binary:
            y = torch.sigmoid(y)
            y0 = torch.sigmoid(y0)
            y1 = torch.sigmoid(y1)
        return loss, y, y0, y1 


class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
  