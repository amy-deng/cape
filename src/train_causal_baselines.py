# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import time
import os
import numpy as np
import argparse

'''
python train_causal_base.py --loop 1 -m tarnet

'''
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='')
parser.add_argument('--path', type=str, default='../data/')
parser.add_argument('--outdir', type=str, default="search_results")

parser.add_argument('-d','--dataset', type=str, default='NI')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--h_dim', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=1e-4, help='trade-off of representation balancing.')
parser.add_argument('--clip', type=float, default=100., help='gradient clipping')
parser.add_argument("-b",'--batch', type=int, default=64)
parser.add_argument('-w','--window', type=int, default=7)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('-p','--patience', type=int, default=10)
parser.add_argument('--train', type=float, default=0.7)
parser.add_argument('--val', type=float, default=0.15)
parser.add_argument('-m','--model', type=str, default='cevae', help='deconf')
parser.add_argument('--loop', type=int, default=10)
parser.add_argument('--shuffle', action="store_false")
parser.add_argument('--i_t', type=int, default=0, help='index of evaluated treatment')
parser.add_argument('--rdm', action="store_false", help="apply randomization for evaluation") 
parser.add_argument('--treat_type', type=int, default=1, help="treatment type, 0:Inc. 1:50%Inc. 2: 50%Dec")
parser.add_argument('--hw', type=int, default=1, help="predict protest in a future window h=1-3 [horizon=1]")

# cevae
parser.add_argument('--z_dim', type=int, default=32)

parser.add_argument('--rep_layer', type=int, default=2)
parser.add_argument('--hyp_layer', type=int, default=2)

parser.add_argument('--p1', type=float, default=0)
parser.add_argument('--p2', type=float, default=0)

args = parser.parse_args()
print(args)
assert args.val > .0, print('args.val should be greater than 0.')

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal
import pandas as pd
import csv
from cevae_net import p_x_z, p_t_z, p_y_zt, q_t_x, q_y_xt, q_z_tyx, init_qz, CEVAE
from models_causal import *
from utils import *

args.cuda = args.gpu >= 0 and torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor

alpha = Tensor([args.alpha])

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    alpha = alpha.cuda()

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('args.device:',args.device,' args.cuda:',args.cuda)
 
if args.model == 'site':
    data_loader = EventDataSITEBasicLoaderStatic(args)
else:
    data_loader = EventDataBasicLoaderStatic(args)

args.p = data_loader.p
 
os.makedirs('models', exist_ok=True)
os.makedirs('models/' + args.dataset, exist_ok=True)

os.makedirs('results', exist_ok=True)
os.makedirs('results/' + args.dataset, exist_ok=True)

os.makedirs(args.outdir, exist_ok=True)
search_path = "{}/{}/{}-w{}h{}p{}-it{}".format(args.outdir,args.dataset,args.model,args.window,args.horizon,args.hw,args.i_t)
os.makedirs(search_path, exist_ok=True)
 
def prepare(args):
    if args.model == 'tarnet':
        model = TARNet(args, data_loader.f, rep_hid=args.h_dim, hyp_hid=args.h_dim, rep_layer=args.rep_layer, hyp_layer=args.hyp_layer, binary=True, p=args.p, device=args.device)
    elif args.model == 'cfrmmd':
        model = CFR_MMD(args, data_loader.f, rep_hid=args.h_dim, hyp_hid=args.h_dim, rep_layer=args.rep_layer, hyp_layer=args.hyp_layer, binary=True, p=args.p, device=args.device)
    elif args.model == 'cfrwass':
        model = CFR_WASS(args, data_loader.f, rep_hid=args.h_dim, hyp_hid=args.h_dim, rep_layer=args.rep_layer, hyp_layer=args.hyp_layer, binary=True, p=args.p, device=args.device)
    elif args.model == 'deconf':
        model = GCN_DECONF(args, nfeat=data_loader.f, nhid=args.h_dim, dropout=args.dropout,n_in=args.rep_layer, n_out=args.hyp_layer, cuda=args.cuda, binary=True)
    elif args.model == 'cevae':
        model = CEVAE(x_dim=data_loader.f*args.window, h_dim=args.h_dim, z_dim=args.z_dim, binfeats=0, contfeats=data_loader.f*args.window, device=args.device, bi_outcome=True)
    elif args.model == 'site':
        model = SITE(args, data_loader.f, rep_hid=args.h_dim, hyp_hid=args.h_dim, rep_layer=2, hyp_layer=2, binary=True, dropout=args.dropout)
    else: 
        raise LookupError('can not find the model')
    model_name = model.__class__.__name__
    token = args.model + '-lr'+str(args.lr)[1:] + 'wd'+str(args.weight_decay) + 'hd' + str(args.h_dim) \
        + 'dp' + str(args.dropout)[1:] \
        + 'b' + str(args.batch) + 'w' + str(args.window) + 'h'+str(args.horizon) + 'hp'+str(args.hw)+'p' + str(args.patience) \
       + 'p1'+str(args.p1)+'p2'+str(args.p2) + 'rep'+str(args.rep_layer) + 'hyp'+str(args.hyp_layer)
    if args.model == 'cevae':
        token += '-z' + str(args.z_dim) 
         
    if args.rdm:
        token += '-rdm' 
    token += '-it'+str(args.i_t)

    print('Model:', model_name)
    print('Token:', token)
    os.makedirs('models/{}/{}'.format(args.dataset, token), exist_ok=True)
    result_file = 'results/{}/{}.csv'.format(args.dataset,token)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#params:', total_params)
    if args.cuda:
        model.cuda()
    
    if args.shuffle:
        data_loader.shuffle()

    if args.model == 'cevae':
        [C, Y, A, X] = data_loader.train
        batch_size, seq_len, n_loc, n_feat = X.size()
        X = X.permute(0,2,1,3).contiguous()
        feat_tr = X.view(batch_size * n_loc, -1)
        treat_tr = C.view(-1, 1)
        outc_tr = Y.view(-1, 1)
        model.init_qz_func(outc_tr.to(args.device), treat_tr.to(args.device), feat_tr.to(args.device))

    return model, optimizer, result_file, token
    
 
def eval(data_loader, data, tag='val'):
    model.eval()
    n_samples = 0.
    total_loss = 0.  
    treat_eval, yf_eval, y1_pred_eval, y0_pred_eval = [], [], [], []

    for inputs in data_loader.get_batches(data, args.batch, False):
        [C, Y, A, X] = inputs # (b,w,m) (b,w,m) (b,w,m,m)  (b,w,m,f) 
        if args.model in ['tarnet','cfrmmd','cfrwass']:
            loss, y_pred, y0, y1  = model(X, C, Y) 
        elif args.model in ['deconf']:
            loss, y_pred, y0, y1 = model(X.squeeze(1), A.squeeze(1), C.view(-1), Y)
        elif args.model == 'cevae':
            batch_size, seq_len, n_loc, n_feat = X.size()
            X = X.permute(0,2,1,3).contiguous()
            x_train, y_train, t_train = X.view(batch_size * n_loc, -1), Y.view(-1,1), C.view(-1,1)
            loss, y_pred, y0, y1 = model(x_train, y_train, t_train)
        elif args.model in ['site']:
            loss, y_pred, y0, y1 = model(X, C, A, Y)  # A is P

        total_loss += loss.item()
        n_samples += 1

        y0 = y0.view(-1, data_loader.m).contiguous()
        y1 = y1.view(-1, data_loader.m).contiguous()
        yf_eval.append(Y)
        treat_eval.append(C)
        y1_pred_eval.append(y1)
        y0_pred_eval.append(y0) 
    yf_eval = torch.cat(yf_eval,0).permute(1,0).contiguous().cpu().detach().numpy() 
    treat_eval = torch.cat(treat_eval,0).permute(1,0).contiguous().cpu().detach().numpy()
    y1_pred_eval = torch.cat(y1_pred_eval,0).permute(1,0).contiguous().cpu().detach().numpy()
    y0_pred_eval = torch.cat(y0_pred_eval,0).permute(1,0).contiguous().cpu().detach().numpy()
    
    # print(treat_eval.shape, yf_eval.shape, y1_pred_eval.shape, y0_pred_eval.shape,'====')
    if args.rdm:
        sampled_idx = data_loader.sel_idx[tag]
    else:
        sampled_idx = None
    eval_dict = eval_causal_effect(treat_eval, yf_eval, y1_pred_eval, y0_pred_eval, sampled_idx=sampled_idx)

    return float(total_loss / n_samples), eval_dict

def train(data_loader, data, epoch, tag='train'):
    model.train()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    total_loss = 0.
    n_samples = 0.
    for inputs in data_loader.get_batches(data, args.batch, True):
        [C, Y, A, X]  = inputs 
        if args.model in ['tarnet','cfrmmd','cfrwass']:
            loss, y_pred, y0, y1  = model(X, C, Y) 
        elif args.model in ['deconf']:
            loss, y_pred, y0, y1 = model(X.squeeze(1), A.squeeze(1), C.view(-1), Y)
        elif args.model == 'cevae':
            batch_size, seq_len, n_loc, n_feat = X.size()
            X = X.permute(0,2,1,3).contiguous()
            x_train, y_train, t_train = X.view(batch_size * n_loc, -1), Y.view(-1,1), C.view(-1,1)
            loss, y_pred, y0, y1 = model(x_train, y_train, t_train)
        elif args.model in ['site']:
            loss, y_pred, y0, y1  = model(X, C, A, Y)  # A is P
        
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        n_samples += 1
    return float(total_loss / n_samples)
 
for i in range(args.loop):
    print('i =', i, args.dataset)
    model, optimizer, result_file, token = prepare(args)
    model_state_file = 'models/{}/{}/{}.pth'.format(args.dataset, token, i)
    if i == 0 and os.path.exists(result_file):  # if result_file exist
        os.remove(result_file)
    ''' save data to file '''
    bad_counter = 0
    loss_small = float('inf')
    try:
        print('begin training')
        for epoch in range(0, args.epochs):
            train_loss = train(data_loader, data_loader.train, epoch)
            valid_loss, eval_dict = eval(data_loader, data_loader.val, tag='val')
            small_value = valid_loss
            if small_value < loss_small:
                loss_small = small_value
                bad_counter = 0
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                print('Epo {} tr_los:{:.5f} val_los:{:.5f} '.format(epoch, train_loss, valid_loss),'|'.join(['{}:{:.5f}'.format(k, eval_dict[k]) for k in eval_dict]))
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break
        print("training done")
            
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early, epoch',epoch)

    checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    f = open(result_file,'a')
    wrt = csv.writer(f)
    
    print("Test using best epoch: {}".format(checkpoint['epoch']))
    val_loss, eval_dict = eval(data_loader, data_loader.val, 'val')
    print('Val','|'.join(['{}:{:.5f}'.format(k, eval_dict[k]) for k in eval_dict]))
    val_res = [eval_dict[k] for k in eval_dict]
 
    _, eval_dict = eval(data_loader, data_loader.test, 'test')
    print('Test','|'.join(['{}:{:.5f}'.format(k, eval_dict[k]) for k in eval_dict]))
    test_res = [eval_dict[k] for k in eval_dict]
    wrt.writerow([val_loss] + [0] + val_res + [0] + test_res)
    f.close()
 
# cauculate mean and std, and save them to res_stat
with open(result_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    arr = []
    for row in csv_reader:
        arr.append(list(map(float, row))) 
arr = np.array(arr)
arr = np.nan_to_num(arr)
line_count = arr.shape[0]
mean = [round(float(v),3) for v in arr.mean(0)]
std = [round(float(v),3) for v in arr.std(0)]
res = [str(mean[i]) +' ' + str(std[i]) for i in range(len(mean))]
print(res)

all_res_file = '{}/{}-{}.csv'.format(search_path,args.dataset,args.model)
f = open(all_res_file,'a')
wrt = csv.writer(f)
wrt.writerow([token] + [line_count] + res)
f.close()