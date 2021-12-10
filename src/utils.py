import sys
import torch
import numpy as np
import random
import pickle
from math import sqrt
import scipy.sparse as sp
from torch.autograd import Variable
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
 
''' data loader '''
 
class EventDataLoaderStatic(object):
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window # 20
        self.h = args.horizon # 1
        self.hw = args.hw #
        self.rdm = args.rdm
        self.dataset = args.dataset
        self.i_t = args.i_t
        self.outcome = np.loadtxt(open("{}{}/outcome.txt".format(args.path,args.dataset)), delimiter=',')
        with open('{}{}/adj.pkl'.format(args.path,args.dataset), 'rb') as f:
            self.adj = pickle.load(f)
        geo_adj = np.loadtxt(open("{}{}/geoadj.txt".format(args.path,args.dataset)), delimiter=',')
        self.geo_mx = sparse_mx_to_torch_sparse_tensor(normalize(geo_adj)).to(args.device)
        if args.fixadj:
            print('static adj')
            self.adj = np.array([list(normalize_adj(geo_adj).toarray()) for adj in self.adj])
        else:
            print('mixed dynamic adj')
            self.adj = np.array([list(normalize_cosadj_w_geoadj(adj, geo_adj).toarray()) for adj in self.adj])
         
        with open('{}{}/feat-evt.pkl'.format(args.path,args.dataset), 'rb') as f:
            self.feature = pickle.load(f)
            self.feature = np.array([list(v.toarray()) for v in self.feature])
        self.n, self.m, self.f = self.feature.shape
        self.n_c = 20
        print('n =',self.n, 'm =', self.m, 'f =',self.f)
        print('outcome',self.outcome.shape)
        print('covariates',self.feature.shape,self.feature.max(),self.feature.min())
      
        self.outcome = np.where(self.outcome > 0., 1., 0.)
        ratio_events = self.outcome.mean(0)
        print(" % Occur =",[round(v,3) for v in ratio_events],self.outcome.mean())
        self._split(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        self.train_noise = args.train_noise
        self.test_noise = args.test_noise
        self.train_miss = args.train_miss
        self.test_miss = args.test_miss
     

    def _random_pairs(self,treatment,outcome,feature):
        feature_locs = []
        pairwise_dis_locs = []
        for i in range(self.m):
            pca = PCA(n_components=10)
            feature_new = pca.fit_transform(feature[:,:,i].reshape(feature.shape[0],-1))
            feature_locs.append(feature_new)
            p_dis = metrics.pairwise_distances(feature_new)
            pairwise_dis_locs.append(p_dis)

        sample_indices_locs = [] 
        for i in range(self.m):
            pairwise_dis = pairwise_dis_locs[i]
            indices = (treatment[:,i]==1).nonzero().view(-1).numpy()
            control_indices = (treatment[:,i]==0).nonzero().view(-1).numpy()
            sample_indices = []
            for idx in indices:
                selected_control_indices = []
                sorted_c_indices = pairwise_dis[idx].argsort()
                sorted_c_indices = [v for v in sorted_c_indices if v in control_indices and v not in selected_control_indices]
                if len(sorted_c_indices) > 1:
                    sample_indices += [idx, sorted_c_indices[1]]
                    selected_control_indices.append(sorted_c_indices[1])
                else:
                    break 
            sample_indices_locs += [v+len(treatment[:,i])*i for v in sample_indices]
        return sample_indices_locs
   
    def _split(self, train, valid, test): 
        self.train_set = range(self.P+self.h-1, train)
        self.valid_set = range(train, valid)
        self.test_set = range(valid, self.n)

        self.train = self._batchify(self.train_set) # torch.Size([179, 20, 47]) torch.Size([179, 47])
        self.val = self._batchify(self.valid_set)
        self.test = self._batchify(self.test_set)
 
        Cc = torch.cat((self.train[-1], self.val[-1], self.test[-1]), 0)
        self.p_vec =  Cc.mean(0).mean(0)
        self.p_all =  Cc.mean()

    def standarize(self):
        print('__standarize__')
        x_train_2d = self.train[3].view(-1,self.f)
        scaler = preprocessing.StandardScaler().fit(x_train_2d)
        x_train_2d = torch.from_numpy(scaler.transform(x_train_2d)) 
        self.train[3] = x_train_2d.view(self.train[3].shape).float()

        x_val_2d = self.val[3].view(-1,self.f)
        x_val_2d = torch.from_numpy(scaler.transform(x_val_2d)).float()
        self.val[3] = x_val_2d.view(self.val[3].shape) 

        x_test_2d = self.test[3].view(-1,self.f)
        x_test_2d = torch.from_numpy(scaler.transform(x_test_2d)) 
        self.test[3] = x_test_2d.view(self.test[3].shape).float()
        # print('test max min ',self.test[3].max(),self.test[3].min())

    def shuffle(self):
        [C_tr, Y_tr, A_tr, X_tr, Cc_tr] = self.train
        [C_va, Y_va, A_va, X_va, Cc_va] = self.val
        [C_te, Y_te, A_te, X_te, Cc_te] = self.test
        C = torch.cat((C_tr, C_va, C_te), 0)
        Y = torch.cat((Y_tr, Y_va, Y_te), 0)
        A = torch.cat((A_tr, A_va, A_te), 0)
        X = torch.cat((X_tr, X_va, X_te), 0)
        Cc = torch.cat((Cc_tr, Cc_va, Cc_te), 0)

        idx = list(range(C.size(0)))
        random.shuffle(idx)
        idx_tr = idx[:C_tr.size(0)]
        idx_va = idx[C_tr.size(0):C_tr.size(0)+C_va.size(0)]
        idx_te = idx[-C_te.size(0):]

        self.train = [C[idx_tr], Y[idx_tr], A[idx_tr], X[idx_tr], Cc[idx_tr]]
        self.val = [C[idx_va], Y[idx_va],  A[idx_va], X[idx_va], Cc[idx_va]]
        self.test = [C[idx_te], Y[idx_te],  A[idx_te], X[idx_te], Cc[idx_te]]
        self.raw_train_x = self.train[3]
        self.raw_val_x = self.val[3]
        self.raw_test_x = self.test[3]

        if self.rdm:
            C_trva = torch.cat((C[idx_tr], C[idx_va]), 0)
            Y_trva = torch.cat((Y[idx_tr], Y[idx_va]), 0)
            A_trva = torch.cat((A[idx_tr], A[idx_va]), 0)
            X_trva = torch.cat((X[idx_tr], X[idx_va]), 0)
            Cc_trva = torch.cat((Cc[idx_tr], Cc[idx_va]), 0)
            self.train_val = [C_trva, Y_trva, A_trva, X_trva, Cc_trva] 

            sel_test_idx = self._random_pairs(self.test[0],self.test[1],self.test[3])
            sel_val_idx = self._random_pairs(self.val[0],self.val[1],self.val[3])
            self.sel_idx = {'test':sel_test_idx,'val':sel_val_idx}

    def permute_feature(self): 
        if self.train_noise > 0:
            noise = np.random.poisson(self.train_noise,self.raw_train_x.shape)
            self.train[3] += torch.from_numpy(noise)
        if self.test_noise > 0:
            val_noise = np.random.poisson(self.test_noise,self.raw_val_x.shape)
            self.val[3] += torch.from_numpy(val_noise) 
            test_noise = np.random.poisson(self.test_noise,self.raw_test_x.shape)
            self.test[3] += torch.from_numpy(test_noise)
        if self.train_miss > 0:
            mask = (torch.FloatTensor(self.raw_train_x.size()).uniform_() > self.train_miss)*1.0
            self.train[3] *= mask 
        if self.test_miss > 0:
            mask = (torch.FloatTensor(self.raw_val_x.size()).uniform_() > self.test_miss)*1.0
            self.val[3] *= mask
            mask = (torch.FloatTensor(self.raw_test_x.size()).uniform_() > self.test_miss)*1.0
            self.test[3] *= mask

    def _batchify(self, idx_set): 
        n = len(idx_set)
        C = torch.zeros((n, self.m))
        Y = torch.zeros((n, self.m))
        A = torch.zeros((n, self.P, self.m, self.m))
        X = torch.zeros((n, self.P, self.m, self.f))
        Cc = torch.zeros((n, self.m, self.n_c))
 
        for i in range(0,n,self.hw):
            end = idx_set[i] - self.h + 1
            start = end - self.P 
            start_y = end  + self.h -1
            end_y = start_y + self.hw 
            curr_event = self.feature[start:end, :].sum(0)
            if start-self.P < 0:
                prev_event = np.zeros(curr_event.shape)
            else:
                prev_event = self.feature[start-self.P:end-self.P, :].sum(0)
            treatment = np.where(curr_event-prev_event>0, 1, 0)
            C[i] = torch.from_numpy(treatment[:,self.i_t])

            Cc[i,:] = torch.from_numpy(treatment) 
            A[i,:self.P,:]  = torch.from_numpy(self.adj[start:end, :]) 
            X[i,:self.P,:]  = torch.from_numpy(self.feature[start:end, :])
            tmp_y = self.outcome[start_y:end_y,:].sum(0)
            tmp_y = np.where(tmp_y > 0, 1., 0.)
            Y[i,:]  = torch.from_numpy(tmp_y)
        return [C, Y, A, X, Cc] 

    def get_batches(self, data, batch_size, shuffle=True):
        [C, Y, A, X, Cc] = data
        length = len(C)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            cc = C[excerpt,:]
            yy = Y[excerpt,:]
            aa = A[excerpt,:]
            xx = X[excerpt,:]
            ccc = Cc[excerpt,:] 
            if (self.cuda): 
                cc = cc.cuda()
                yy = yy.cuda()
                aa = aa.cuda()
                xx = xx.cuda()
                ccc = ccc.cuda()
            data = [Variable(cc), Variable(yy), Variable(aa),  Variable(xx), ccc]
            yield data
            start_idx += batch_size


class EventDataBasicLoaderStatic(object):
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window # 20
        self.h = args.horizon # 1
        self.hw = args.hw #
        self.rdm = args.rdm
        self.dataset = args.dataset
          
        self.outcome = np.loadtxt(open("{}{}/outcome.txt".format(args.path,args.dataset)), delimiter=',')
        with open('{}{}/adj.pkl'.format(args.path,args.dataset), 'rb') as f:
            self.adj = pickle.load(f)
         
        geo_adj = np.loadtxt(open("{}{}/geoadj.txt".format(args.path,args.dataset)), delimiter=',')
        if args.model == 'dndc': # dynamic adj
            print('mixed dynamic adj')
            self.adj = np.array([list(normalize_cosadj_w_geoadj(adj, geo_adj).toarray()) for adj in self.adj])
        else:
            print('static adj')
            self.adj = np.array([list(normalize_adj(geo_adj).toarray()) for adj in self.adj])
        
        with open('{}{}/feat-evt.pkl'.format(args.path,args.dataset), 'rb') as f:
            self.feature = pickle.load(f)
            self.feature = np.array([list(v.toarray()) for v in self.feature])
       
        self.n, self.m, self.f = self.feature.shape
        self.i_t = args.i_t
        print('n =',self.n, 'm =', self.m, 'f =',self.f)
        print('outcome',self.outcome.shape)
        print('covariates',self.feature.shape)

        
        self.outcome = np.where(self.outcome > 0., 1., 0.)
        self._split(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        
    def _random_pairs(self,treatment,outcome,feature):
        feature_locs = []
        pairwise_dis_locs = []
        for i in range(self.m):
            pca = PCA(n_components=10)
            feature_new = pca.fit_transform(feature[:,:,i].reshape(feature.shape[0],-1))
            feature_locs.append(feature_new)
            p_dis = metrics.pairwise_distances(feature_new)
            pairwise_dis_locs.append(p_dis)

        sample_indices_locs = [] 
        for i in range(self.m):
            pairwise_dis = pairwise_dis_locs[i]
            indices = (treatment[:,i]==1).nonzero().view(-1).numpy()
            control_indices = (treatment[:,i]==0).nonzero().view(-1).numpy()
            sample_indices = []
            for idx in indices:
                selected_control_indices = []
                sorted_c_indices = pairwise_dis[idx].argsort()
                sorted_c_indices = [v for v in sorted_c_indices if v in control_indices and v not in selected_control_indices]
                if len(sorted_c_indices) > 1:
                    sample_indices += [idx, sorted_c_indices[1]]
                    selected_control_indices.append(sorted_c_indices[1])
                else:
                    break 
            sample_indices_locs += [v+len(treatment[:,i])*i for v in sample_indices]
        return sample_indices_locs
   
    def _split(self, train, valid, test):

        self.train_set = range(self.P+self.h-1, train)
        self.valid_set = range(train, valid)
        self.test_set = range(valid, self.n)

        self.train = self._batchify(self.train_set) # torch.Size([179, 20, 47]) torch.Size([179, 47])
        self.val = self._batchify(self.valid_set)
        self.test = self._batchify(self.test_set)

        C = torch.cat((self.train[0], self.val[0], self.test[0]), 0)
        self.p = C.mean()
        print('treatment =',self.p)
    
    def shuffle(self):
        [C_tr, Y_tr, A_tr, X_tr] = self.train
        [C_va, Y_va, A_va, X_va] = self.val
        [C_te, Y_te, A_te, X_te] = self.test
        C = torch.cat((C_tr, C_va, C_te), 0)
        Y = torch.cat((Y_tr, Y_va, Y_te), 0)
        A = torch.cat((A_tr, A_va, A_te), 0)
        X = torch.cat((X_tr, X_va, X_te), 0)
        idx = list(range(C.size(0)))
        random.shuffle(idx)
        idx_tr = idx[:C_tr.size(0)]
        idx_va = idx[C_tr.size(0):C_tr.size(0)+C_va.size(0)]
        idx_te = idx[-C_te.size(0):]
        self.train = [C[idx_tr], Y[idx_tr], A[idx_tr], X[idx_tr]]
        self.val = [C[idx_va], Y[idx_va], A[idx_va], X[idx_va]]
        self.test = [C[idx_te], Y[idx_te], A[idx_te], X[idx_te]]

        sel_test_idx = self._random_pairs(self.test[0],self.test[1],self.test[3])
        sel_val_idx = self._random_pairs(self.val[0],self.val[1],self.val[3])
        self.sel_idx = {'test':sel_test_idx,'val':sel_val_idx}
        print("__shuffle__")
        self.standarize()

    def standarize(self):
        print('__standarize__')
        x_train_2d = self.train[3].view(-1,self.f)
        scaler = preprocessing.StandardScaler().fit(x_train_2d)
        x_train_2d = torch.from_numpy(scaler.transform(x_train_2d)) 
        self.train[3] = x_train_2d.view(self.train[3].shape).float()

        x_val_2d = self.val[3].view(-1,self.f)
        x_val_2d = torch.from_numpy(scaler.transform(x_val_2d)).float()
        self.val[3] = x_val_2d.view(self.val[3].shape) 

        x_test_2d = self.test[3].view(-1,self.f)
        x_test_2d = torch.from_numpy(scaler.transform(x_test_2d)) 
        self.test[3] = x_test_2d.view(self.test[3].shape).float()

    def _batchify(self, idx_set):  
        n = len(idx_set)
        C = torch.zeros((n, self.m))
        Y = torch.zeros((n, self.m))
        A = torch.zeros((n, self.P, self.m, self.m))
        X = torch.zeros((n, self.P, self.m, self.f))
 
        for i in range(0,n,self.hw):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            start_y = end  + self.h -1
            end_y = start_y + self.hw  
            curr_event = self.feature[start:end, :].sum(0)
            if start-self.P < 0:
                prev_event = np.zeros(curr_event.shape)
            else:
                prev_event = self.feature[start-self.P:end-self.P, :].sum(0)
            # print('strat',start,'end',end,'prev',start-self.P,end-self.P,curr_event.shape,prev_event.shape)
            treatment = np.where(curr_event-prev_event>0, 1, 0)
            C[i] = torch.from_numpy(treatment[:,self.i_t])
            A[i,:self.P,:]  = torch.from_numpy(self.adj[start:end, :])
            X[i,:self.P,:]  = torch.from_numpy(self.feature[start:end, :])
            tmp_y = self.outcome[start_y:end_y,:].sum(0)
            tmp_y = np.where(tmp_y > 0, 1., 0.)
            Y[i,:]  = torch.from_numpy(tmp_y)
       
        # print('total #',n,C.shape,'Y',Y.shape,'A',A.shape,'X',X.shape)
        return [C, Y, A, X] 

    def get_batches(self, data, batch_size, shuffle=True):
        [C, Y, A, X] = data
        length = len(C)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            cc = C[excerpt,:]
            yy = Y[excerpt,:]
            aa = A[excerpt,:]
            xx = X[excerpt,:]
            if (self.cuda):  
                cc = cc.cuda()
                yy = yy.cuda()
                aa = aa.cuda()
                xx = xx.cuda()
            data = [Variable(cc), Variable(yy), Variable(aa), Variable(xx)]
            yield data
            start_idx += batch_size


class EventDataSITEBasicLoaderStatic(object):
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window # 20
        self.h = args.horizon # 1
        self.hw = args.hw #
        self.rdm = args.rdm
        self.dataset = args.dataset
        self.outcome = np.loadtxt(open("{}{}/outcome.txt".format(args.path,args.dataset)), delimiter=',')
        with open('{}{}/adj.pkl'.format(args.path,args.dataset), 'rb') as f:
            self.adj = pickle.load(f)
         
        geo_adj = np.loadtxt(open("{}{}/geoadj.txt".format(args.path,args.dataset)), delimiter=',')
        if args.model == 'dndc': # dynamic adj
            print('mixed dynamic adj')
            self.adj = np.array([list(normalize_cosadj_w_geoadj(adj, geo_adj).toarray()) for adj in self.adj])
        else:
            print('static adj')
            self.adj = np.array([list(normalize_adj(geo_adj).toarray()) for adj in self.adj])
        
        with open('{}{}/feat-evt.pkl'.format(args.path,args.dataset), 'rb') as f:
            self.feature = pickle.load(f)
            self.feature = np.array([list(v.toarray()) for v in self.feature])
       
        self.n, self.m, self.f = self.feature.shape

        self.i_t = args.i_t
        print('n =',self.n, 'm =', self.m, 'f =',self.f)
        print('outcome',self.outcome.shape)
        print('covariates',self.feature.shape)
        
        self.outcome = np.where(self.outcome > 0., 1., 0.)
        self.get_propensity_score()
        self._split(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)


    def _random_pairs(self,treatment,outcome,feature):
        feature_locs = []
        pairwise_dis_locs = []
        for i in range(self.m):
            pca = PCA(n_components=10)
            feature_new = pca.fit_transform(feature[:,:,i].reshape(feature.shape[0],-1))
            feature_locs.append(feature_new)
            p_dis = metrics.pairwise_distances(feature_new)
            pairwise_dis_locs.append(p_dis)

        sample_indices_locs = [] 
        for i in range(self.m):
            pairwise_dis = pairwise_dis_locs[i]
            indices = (treatment[:,i]==1).nonzero().view(-1).numpy()
            control_indices = (treatment[:,i]==0).nonzero().view(-1).numpy()
            sample_indices = []
            for idx in indices:
                selected_control_indices = []
                sorted_c_indices = pairwise_dis[idx].argsort()
                sorted_c_indices = [v for v in sorted_c_indices if v in control_indices and v not in selected_control_indices]
                if len(sorted_c_indices) > 1:
                    sample_indices += [idx, sorted_c_indices[1]]
                    selected_control_indices.append(sorted_c_indices[1])
                else:
                    break 
            sample_indices_locs += [v+len(treatment[:,i])*i for v in sample_indices]
        return sample_indices_locs
 
    def get_propensity_score(self):
        x_train = self.feature.reshape(-1,self.f)
        y_train = self.outcome.reshape(-1)
        clf = LogisticRegression().fit(x_train, y_train)
        P = clf.predict_proba(x_train)[:,1]
        self.propensity = P.reshape(self.outcome.shape)
 
    def _split(self, train, valid, test):

        self.train_set = range(self.P+self.h-1, train)
        self.valid_set = range(train, valid)
        self.test_set = range(valid, self.n)

        self.train = self._batchify(self.train_set) # torch.Size([179, 20, 47]) torch.Size([179, 47])
        self.val = self._batchify(self.valid_set)
        self.test = self._batchify(self.test_set)

        C = torch.cat((self.train[0], self.val[0], self.test[0]), 0)
        self.p = C.mean()
        print('treatment =',self.p)
    
    def shuffle(self):
        [C_tr, Y_tr, P_tr, X_tr] = self.train
        [C_va, Y_va, P_va, X_va] = self.val
        [C_te, Y_te, P_te, X_te] = self.test
        C = torch.cat((C_tr, C_va, C_te), 0)
        Y = torch.cat((Y_tr, Y_va, Y_te), 0)
        P = torch.cat((P_tr, P_va, P_te), 0)
        X = torch.cat((X_tr, X_va, X_te), 0)
        idx = list(range(C.size(0)))
        random.shuffle(idx)
        idx_tr = idx[:C_tr.size(0)]
        idx_va = idx[C_tr.size(0):C_tr.size(0)+C_va.size(0)]
        idx_te = idx[-C_te.size(0):]
        self.train = [C[idx_tr], Y[idx_tr], P[idx_tr], X[idx_tr]]
        self.val = [C[idx_va], Y[idx_va], P[idx_va], X[idx_va]]
        self.test = [C[idx_te], Y[idx_te], P[idx_te], X[idx_te]]
       
        sel_test_idx = self._random_pairs(self.test[0],self.test[1],self.test[3])
        sel_val_idx = self._random_pairs(self.val[0],self.val[1],self.val[3])
        self.sel_idx = {'test':sel_test_idx,'val':sel_val_idx}
        print("__shuffle__")
        self.standarize()
    
    def standarize(self):
        print('__standarize__')
        x_train_2d = self.train[3].view(-1,self.f)
        scaler = preprocessing.StandardScaler().fit(x_train_2d)
        x_train_2d = torch.from_numpy(scaler.transform(x_train_2d)) 
        self.train[3] = x_train_2d.view(self.train[3].shape).float()

        x_val_2d = self.val[3].view(-1,self.f)
        x_val_2d = torch.from_numpy(scaler.transform(x_val_2d)).float()
        self.val[3] = x_val_2d.view(self.val[3].shape) 

        x_test_2d = self.test[3].view(-1,self.f)
        x_test_2d = torch.from_numpy(scaler.transform(x_test_2d)) 
        self.test[3] = x_test_2d.view(self.test[3].shape).float()


    def _batchify(self, idx_set):  
        n = len(idx_set)
        C = torch.zeros((n, self.m))
        Y = torch.zeros((n, self.m))
        A = torch.zeros((n, self.P, self.m, self.m))
        X = torch.zeros((n, self.P, self.m, self.f))
 
        for i in range(0,n,self.hw):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            start_y = end  + self.h -1
            end_y = start_y + self.hw  
            curr_event = self.feature[start:end, :].sum(0)
            if start-self.P < 0:
                prev_event = np.zeros(curr_event.shape)
            else:
                prev_event = self.feature[start-self.P:end-self.P, :].sum(0)
            treatment = np.where(curr_event-prev_event>0, 1, 0)
            C[i] = torch.from_numpy(treatment[:,self.i_t])
            A[i,:self.P,:]  = torch.from_numpy(self.adj[start:end, :])
            X[i,:self.P,:]  = torch.from_numpy(self.feature[start:end, :])
            tmp_y = self.outcome[start_y:end_y,:].sum(0)
            tmp_y = np.where(tmp_y > 0, 1., 0.)
            Y[i,:]  = torch.from_numpy(tmp_y)
     
        print('total #',n,'C',C.shape,'Y',Y.shape,'A',A.shape,'X',X.shape)
        return [C, Y, A, X] 

    def get_batches(self, data, batch_size, shuffle=True):
        [C, Y, P, X] = data
        length = len(C)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            cc = C[excerpt,:]
            yy = Y[excerpt,:]
            aa = P[excerpt,:]
            xx = X[excerpt,:]
            if (self.cuda):  
                cc = cc.cuda()
                yy = yy.cuda()
                aa = aa.cuda()
                xx = xx.cuda()
            data = [Variable(cc), Variable(yy), Variable(aa), Variable(xx)]
            yield data
            start_idx += batch_size


''' evaluation '''
def eval_causal_effect(treatment, yf, y1_pred, y0_pred, sampled_idx=None):
    r = {}
    treatment = treatment.reshape(-1)
    yf = yf.reshape(-1)
    y1_pred = y1_pred.reshape(-1)
    y0_pred = y0_pred.reshape(-1)

    eff_pred = y1_pred - y0_pred 
    
    att_err = att_eval(treatment, yf, eff_pred, sampled_idx) # error att_err_eval
    r['att_err'] = att_err
    return r

def eval_bi_classifier(y_true, y_score, y_pred_bi=None):
    y_true = y_true.reshape(-1)
    y_score = y_score.reshape(-1)
    r = {}
    try:
        r['auroc'] = metrics.roc_auc_score(y_true, y_score)
    except:
        pass
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)
    r['aupr'] = aupr
    if y_pred_bi is None:# or True:
        y_pred_bi = np.where(y_score>0.5,1,0)
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred_bi, average="binary")  
    r['prec'] = prec
    r['rec'] = rec
    r['f1'] = f1
    r['bacc'] = metrics.balanced_accuracy_score(y_true,y_pred_bi)
    r['acc'] = metrics.accuracy_score(y_true,y_pred_bi)
    return r
 

def att_eval(treatment, yf, eff_pred, sampled_idx=None):
    if sampled_idx is not None and len(sampled_idx) > 0:
        treatment = treatment[sampled_idx]
        # print(treatment.mean(),'treatment mean (check if data are pair-matched)')
        yf = yf[sampled_idx]
        eff_pred = eff_pred[sampled_idx]
    att = np.mean(yf[treatment == 1]) - np.mean(yf[treatment == 0])
    att_pred = np.mean(eff_pred[treatment == 1])
    att_err = np.absolute(att - att_pred) # the smaller the better
    return att_err
  

''' matrix operation '''
def normalize_cosadj_w_geoadj(adj, geo_adj):
    adj.setdiag(0)
    adj = 0.2 * adj
    adj += geo_adj
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

 
def normalize(adj):
    adj = sp.coo_matrix(adj)
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

 
def sparse_mx_to_torch_sparse_tensor(sparse_mx,cuda=False):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    if cuda:
        sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())
    
def move_dgl_to_cuda(g, device):
    g.ndata.update({k: g.ndata[k].to(device) for k in g.ndata})
    g.edata.update({k: g.edata[k].to(device) for k in g.edata})


''' distance measurement '''
def wasserstein(x, y ,p=0.5,lam=10,its=10,sq=False,backpropT=False,cuda=False):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]
    
    x = x.squeeze()
    y = y.squeeze()
    M = pdist(x,y) #distance_matrix(x,y,p=2)
    
    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M,10.0/(nx*ny))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam/M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta*torch.ones(M[0:1,:].shape)
    col = torch.cat([delta*torch.ones(M[:,0:1].shape),torch.zeros((1,1))],0)
    if cuda:
        row = row.cuda()
        col = col.cuda()
    Mt = torch.cat([M,row],0)
    Mt = torch.cat([Mt,col],1)

    '''compute marginal'''
    a = torch.cat([p*torch.ones((nx,1))/nx,(1-p)*torch.ones((1,1))],0)
    b = torch.cat([(1-p)*torch.ones((ny,1))/ny, p*torch.ones((1,1))],0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1)*1e-6
    if cuda:
        temp_term = temp_term.cuda()
        a = a.cuda()
        b = b.cuda()
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K/a

    u = a

    for i in range(its):
        u = 1.0/(ainvK.matmul(b/torch.t(torch.t(u).matmul(K))))
        if cuda:
            u = u.cuda()
    v = b/(torch.t(torch.t(u).matmul(K)))
    if cuda:
        v = v.cuda()

    upper_t = u*(torch.t(v)*K).detach()

    E = upper_t*Mt
    D = 2*torch.sum(E)

    if cuda:
        D = D.cuda()

    return D, Mlam

def wasserstein_ht(X,t,p=0.5,lam=10,its=20,sq=False,backpropT=False,device=torch.device('cpu')):
    """return W dist between x and y"""
    '''distance matrix M'''
    # device = torch.device('cuda' if cuda else 'cpu')
    it = 1*(t>0).nonzero().view(-1)
    ic = 1*(t<1).nonzero().view(-1)
    Xc = X[ic]
    Xt = X[it]
    n = Xt.size(0) 
    m = Xc.size(0) 
    nx = n
    ny = m
    M = pdist(Xt,Xc) #distance_matrix(x,y,p=2)
    '''estimate lambda and delta'''
    M_mean = torch.mean(M)

    M_drop = F.dropout(M,10.0/(nx*ny))
   
    delta = torch.max(M_drop).detach()
    eff_lam = (lam/M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
     
    row = delta*torch.ones(M[0:1,:].shape).to(device)
    col = torch.cat([delta*torch.ones(M[:,0:1].shape).to(device),torch.zeros((1,1)).to(device)],0)
    row = row.to(device)
    col = col.to(device)
    Mt = torch.cat([M,row],0)
    Mt = torch.cat([Mt,col],1)

    '''compute marginal'''
    a = torch.cat([p*torch.ones((nx,1)).to(device)/nx,(1-p)*torch.ones((1,1)).to(device)],0)
    b = torch.cat([(1-p)*torch.ones((ny,1)).to(device)/ny, p*torch.ones((1,1)).to(device)],0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1)*1e-6
    temp_term = temp_term.to(device)
    a = a.to(device)
    b = b.to(device)
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K/a
    u = a
    for i in range(its):
        u = 1.0/(ainvK.matmul(b/torch.t(torch.t(u).matmul(K)))).to(device)
    v = b/(torch.t(torch.t(u).matmul(K)))
    v = v.to(device)

    upper_t = u*(torch.t(v)*K).detach()

    E = upper_t*Mt
    D = 2*torch.sum(E)
    D = D.to(device)
    return D, Mlam



def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

def mmd2_rbf(X,t,p=0.5,sig=0.1):
    """ Computes the l2-RBF MMD for X given t """
    it = 1*(t>0).nonzero().view(-1)
    ic = 1*(t<1).nonzero().view(-1)
    Xc = X[ic]
    Xt = X[it]
    Kcc = torch.exp(-pdist(Xc,Xc)/(sig**2))
    Kct = torch.exp(-pdist(Xc,Xt)/(sig**2))
    Ktt = torch.exp(-pdist(Xt,Xt)/(sig**2))
    m = Xc.size(0) 
    n = Xt.size(0) 
    mmd = (1.0-p)**2/(m*(m-1.0))*(Kcc.sum()-m)
    mmd = mmd + p**2/(n*(n-1.0))*(Ktt.sum()-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*Kct.sum()
    mmd = 4.0*mmd
    return mmd

def mmd2_lin(X,t,p=0.5):
    ''' Linear MMD '''

    it = 1*(t>0).nonzero().view(-1)
    ic = 1*(t<1).nonzero().view(-1)
    Xc = X[ic]
    Xt = X[it]

    mean_control = torch.mean(Xc,dim=0)
    mean_treated = torch.mean(Xt,dim=0)

    mmd = torch.sum(torch.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))
    return mmd

def wasserstein_ht(X,t,p=0.5,lam=10,its=10,sq=False,backpropT=False,device=torch.device('cpu')):
    """return W dist between x and y"""
    '''distance matrix M'''
    # device = torch.device('cuda' if cuda else 'cpu')
    it = 1*(t>0).nonzero().view(-1)
    ic = 1*(t<1).nonzero().view(-1)
    p = t.mean()
    Xc = X[ic]
    Xt = X[it]
    n = Xt.size(0) 
    m = Xc.size(0) 
    nx = n
    ny = m
    M = pdist(Xt,Xc) #distance_matrix(x,y,p=2)
    '''estimate lambda and delta'''
    M_mean = torch.mean(M)

    M_drop = F.dropout(M,10.0/(nx*ny))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam/M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
     
    row = delta*torch.ones(M[0:1,:].shape).to(device)
    col = torch.cat([delta*torch.ones(M[:,0:1].shape).to(device),torch.zeros((1,1)).to(device)],0)
    row = row.to(device)
    col = col.to(device)
    Mt = torch.cat([M,row],0)
    Mt = torch.cat([Mt,col],1)

    '''compute marginal'''
    a = torch.cat([p*torch.ones((nx,1)).to(device)/nx,(1-p)*torch.ones((1,1)).to(device)],0)
    b = torch.cat([(1-p)*torch.ones((ny,1)).to(device)/ny, p*torch.ones((1,1)).to(device)],0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1)*1e-6
    temp_term = temp_term.to(device)
    a = a.to(device)
    b = b.to(device)
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K/a

    u = a

    for i in range(its):
        u = 1.0/(ainvK.matmul(b/torch.t(torch.t(u).matmul(K))))
        u = u.to(device)
    v = b/(torch.t(torch.t(u).matmul(K)))
    v = v.to(device)

    upper_t = u*(torch.t(v)*K).detach()

    E = upper_t*Mt
    D = 2*torch.sum(E)
    D = D.to(device)
    return D, Mlam


def get_cameo_main(code):
    code = int(code)
    if code < 100:
        return code // 10
    if code // 10 > 20:
        return code // 100
    else:
        return code // 10


''' others '''

def find_middle_pair(x, y):
    x = torch.abs(x-0.5)
    y = torch.abs(y-0.5)
    index_1 = torch.argmin(x).item()
    index_2 = torch.argmin(y).item()
    return index_1, index_2

def find_nearest_point(x, p):
    diff = torch.abs(x-p)
    diff_1 = diff[diff>0]
    min_val = torch.min(diff_1)
    I_diff = torch.where(diff == min_val)[0]
    I_diff = I_diff[0]
    if len(I_diff.size())>1:
        I_diff = I_diff[0]
    return I_diff.item()

