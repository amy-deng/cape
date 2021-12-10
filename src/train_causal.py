# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import time
import os
import numpy as np
import argparse
import pickle
'''

'''
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='')
parser.add_argument('--path', type=str, default='../data/')
parser.add_argument('--outdir', type=str, default="search_results")
parser.add_argument('-d','--dataset', type=str, default='NI')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=1e-4, help='trade-off of representation balancing.')
parser.add_argument('--clip', type=float, default=100., help='gradient clipping')
parser.add_argument("-b",'--batch', type=int, default=64)
parser.add_argument('-w','--window', type=int, default=7)
parser.add_argument('-ho','--horizon', type=int, default=1)
parser.add_argument('--hw', type=int, default=1, help="predict protest in a future window h=1-3 [horizon=1]")

parser.add_argument('-p','--patience', type=int, default=20)
parser.add_argument('--train', type=float, default=0.7)
parser.add_argument('--val', type=float, default=0.15)
parser.add_argument('-m','--model', type=str, default='cape_cau', help='')
parser.add_argument('--loop', type=int, default=10)
parser.add_argument('--shuffle', action="store_false")
parser.add_argument('--rdm', action="store_false", help="apply randomization for evaluation")
parser.add_argument('--res', type=str, default='event_res_stat', help='event_res_stat or other filename')
parser.add_argument('--u', type=float, default=0.05,help="hyper-param for loss constraints")

 
parser.add_argument('--h_dim', type=int, default=32, help='')
parser.add_argument('--p1', type=float, default=1e-2, help='')
parser.add_argument('--balance', type=str, default='wass', help='wass or mmd')

parser.add_argument('--i_t', type=int, default=0, help='index of evaluated treatment')
parser.add_argument('--fixadj', action="store_false") 
parser.add_argument('--wl', action="store_true", help="weight loss")
parser.add_argument('--treat_type', type=int, default=1, help="treatment type, 0:Inc. 1:50%Inc. 2: 50%Dec")
parser.add_argument('--w_t', type=float, default=0.1, help="weight treat loss")
parser.add_argument('--cf', action="store_true", help="cf")
parser.add_argument('--treat_balance', action="store_false", help="weight loss") # for treatment

 
parser.add_argument('--train_noise', type=float, default=0.0, help="add training possion noise ")
parser.add_argument('--test_noise', type=float, default=0.0, help="add test possion noise ")
parser.add_argument('--train_miss', type=float, default=0.0, help="Probability of masked training feature; randomly set some training features to 0")
parser.add_argument('--test_miss', type=float, default=0.0, help="Probability of masked test feature; randomly set some test features to 0") 


# sensitivity
parser.add_argument('--c',type=int, default=20) # 2, 4, 6, 8, 10, 12, 14, 15, 16, 18
parser.add_argument('--z_dim', type=int, default=32)

args = parser.parse_args()
print(args)

assert args.val > .0, print('args.val should be greater than 0.')
 

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import csv
from models_event import *
from utils import *
from cevae_net import *

args.cuda = args.gpu >= 0 and torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor

alpha = Tensor([args.alpha])

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    alpha = alpha.cuda()

args.device = torch.device('cuda' if args.cuda else 'cpu')
print('args.device:',args.device,' args.cuda:',args.cuda)
 
print('args.i_t =',args.i_t)
# data_loader = EvtGraphLoader(args)
data_loader = EventDataLoaderStatic(args)
args.n_c = data_loader.n_c 
args.p_all = data_loader.p_all.to(args.device)
args.p_vec = data_loader.p_vec.to(args.device)
args.p = args.p_vec[args.i_t]
 
os.makedirs('models', exist_ok=True)
os.makedirs('models/' + args.dataset, exist_ok=True)

os.makedirs('results', exist_ok=True)
os.makedirs('results/' + args.dataset, exist_ok=True)

os.makedirs(args.outdir, exist_ok=True)
search_path = "{}/{}/{}-w{}h{}p{}-it{}".format(args.outdir,args.dataset,args.model,args.window,args.horizon,args.hw,args.i_t)
os.makedirs(search_path, exist_ok=True)


def prepare(args):
    if args.model == 'cape_cau':
        model = cape_cau(args, data_loader)  
    model_name = model.__class__.__name__
    token = args.model + '-lr'+str(args.lr)[1:] + 'wd'+str(args.weight_decay) \
        + 'dp' + str(args.dropout)[1:] \
        + 'b' + str(args.batch) + 'w' + str(args.window) + 'h'+str(args.horizon) + 'hw' + str(args.hw) + 'p' + str(args.patience) + '-hid'+str(args.h_dim) +'-'+str(args.balance)+str(args.p1) \
 
    if args.fixadj:
        token += '-fa' 
    if args.rdm:
        token += '-rdm' 
    token += '-it'+str(args.i_t)

    print('Model:', model_name)
    print('Token:', token)
    os.makedirs('models/{}/{}'.format(args.dataset, token), exist_ok=True)
    result_file = 'results/{}/{}.csv'.format(args.dataset,token)
    # print(model) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#params:', total_params)
    if args.cuda:
        model.cuda()
    
    if args.shuffle:
        data_loader.shuffle()
        print('<<< shuffled >>>')
 
    return model, optimizer, result_file, token
    
 
def eval(data_loader, data, tag='val'):
    model.eval()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    n_samples = 0.
    total_loss = 0.  
    treat_eval, yf_eval, y1_pred_eval, y0_pred_eval, yp_evt_eval = [], [], [], [], []
    for inputs in data_loader.get_batches(data, args.batch, False):
        # [C, Y, A, X] = inputs # (b,w,m) (b,w,m) (b,w,m,m)  (b,w,m,f) 
        [C, Y, A, X, CC]  = inputs
        loss, y_pred, y0, y1  = model(X, A, C, Y, CC)
        total_loss += loss.item()
        n_samples += 1 
        y0 = y0.view(-1, data_loader.m).contiguous()
        y1 = y1.view(-1, data_loader.m).contiguous()
        y_pred = y_pred.view(-1, data_loader.m).contiguous() 
        yf_eval.append(Y)
        yp_evt_eval.append(y_pred)
        treat_eval.append(C)
        y1_pred_eval.append(y1)
        y0_pred_eval.append(y0) 
    yf_eval = torch.cat(yf_eval,0).permute(1,0).contiguous().cpu().detach().numpy() 
    yp_evt_eval = torch.cat(yp_evt_eval,0).permute(1,0).contiguous().cpu().detach().numpy()
    treat_eval = torch.cat(treat_eval,0).permute(1,0).contiguous().cpu().detach().numpy()
    y1_pred_eval = torch.cat(y1_pred_eval,0).permute(1,0).contiguous().cpu().detach().numpy()
    y0_pred_eval = torch.cat(y0_pred_eval,0).permute(1,0).contiguous().cpu().detach().numpy()

    if args.rdm:
        sampled_idx = data_loader.sel_idx[tag]
    else:
        sampled_idx = None
    eval_dict = eval_causal_effect(treat_eval, yf_eval, y1_pred_eval, y0_pred_eval, sampled_idx=sampled_idx)
    return float(total_loss / n_samples), eval_dict

def train(data_loader, data, epoch, tag='train'):
    model.train()
    total_loss = 0.
    n_samples = 0.
    for inputs in data_loader.get_batches(data, args.batch, True):
        [C, Y, A, X, CC]  = inputs
        loss, y_pred, y0, y1  = model(X, A, C, Y, CC)
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
 
    bad_counter = 0
    loss_small = float('inf')
    value_large = float('-inf')
    try:
        print('begin training');
        for epoch in range(0, args.epochs):
            epoch_start_time = time.time()
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
mean = [round(float(v),4) for v in arr.mean(0)]
std = [round(float(v),4) for v in arr.std(0)]
res = [str(mean[i]) +' ' + str(std[i]) for i in range(len(mean))]
print(res)

all_res_file = '{}/{}-{}.csv'.format(search_path,args.dataset,args.model)
f = open(all_res_file,'a')
wrt = csv.writer(f)
wrt.writerow([token] + [line_count] + res)
f.close()
