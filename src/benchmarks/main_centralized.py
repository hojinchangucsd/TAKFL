import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import torchvision
import timm
import random 

from src.data import *
from src.models import *
from src.client import *
from src.clustering import *
from src.utils import *

import concurrent.futures
import torch.cuda

#torch.backends.cudnn.benchmark = True

def main_centralized(args):
    print(' ')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    #print(str(args))
    ##################################### Data partitioning section
    print('-'*40)
    print('Getting Data')
    
    arch = args.models[0]
    data_ratio = args.data_ratios[0]
    print(f'arch: {arch}')
    print(f'Data ratios: {args.data_ratios}')
    
    train_ds_global, test_ds_global, train_dl_global, test_dl_global \
    = get_dataset_global(args.dataset, args.datadir, batch_size=128,
                        p_train=data_ratio, p_test=1.0, seed=args.seed)
    
    _, net_glob, initial_state_dict = get_models_fedmh(num_users=0, model=arch, 
                                                    dataset=args.dataset, args=args)
    
    total = 0
    for name, param in net_glob.named_parameters():
        print(name, param.size())
        total += np.prod(param.size())   
    print(f'total params {total}')
    print('-'*40)
    
    #### Training Loop
    mixed_precision_training = False
    #net_glob = torch.nn.DataParallel(net_glob)
    net_glob.to(args.device)
    net_glob.train()
        
    lr = args.lr[0]
    wd = args.local_wd[0]
    optim_scheduler = args.lr_scheduler[0]
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(net_glob.parameters(), lr=lr, weight_decay=wd)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(net_glob.parameters(), lr=lr, weight_decay=wd)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(net_glob.parameters(), lr=lr, momentum=args.momentum, weight_decay=0)
    
    if optim_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
    elif optim_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.local_ep, eta_min=1e-6)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.local_ep//2, T_mult=1, eta_min=1e-5, last_epoch=-1)
    
    if mixed_precision_training:
        scaler = GradScaler()
    
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    epoch_loss = []
    epoch_acc = []
    for iteration in range(args.local_ep):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(train_dl_global):
            images, labels = images.to(args.device), labels.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            
            net_glob.zero_grad()
            optimizer.zero_grad()
            if mixed_precision_training:
                with autocast():
                    log_probs = net_glob(images)
                    loss = criterion(log_probs, labels)
            
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs = net_glob(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

            batch_loss.append(loss.item())
        
        if optim_scheduler != "none":
            scheduler.step()
        
        _, acc = eval_test(net_glob, args, test_dl_global)
        epoch_acc.append(acc)
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

        template = "--[Epoch {}] Training Loss: {:.2f}, Accuracy: {:.2f}"
        print(template.format(iteration+1, epoch_loss[-1], epoch_acc[-1]))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        sys.stdout.flush()
        
    net_glob.to('cpu')
    
    end = time.time()
    duration = end-start
    
    return epoch_acc[-1], duration

def run_centralized(args, fname):
    alg_name = 'Centralized'
    
    seeds = [2023, 2024, 2022, 2021]
    
    exp_acc=[]
    exp_time=[]
    
    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))
        
        set_seed(seeds[trial])
        acc, duration = main_centralized(args)
        
        exp_acc.append(acc)
        exp_time.append(duration)
        
        print('*'*40)
        print(' '*20, 'End of Trial %d'%(trial+1))
        print(' '*20, 'Final Results')
        
        template = "-- Final Acc: {:.2f}, Time: {:.2f}"
        print(template.format(exp_acc[-1], exp_time[-1]))
        
    avg_acc = np.mean(exp_acc)
    std_acc = np.std(exp_acc)
    avg_time = np.mean(exp_time)
    std_time = np.std(exp_time)
    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, 'Avg %d Trial Results'%args.ntrials)
    
    template = "-- Final Acc: {:.2f} +- {:.2f}, Time: {:.2f} +- {:.2f}"
    print(template.format(avg_acc, std_acc, avg_time, std_time))
    
    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40, file=text_file)
        print(' '*20, alg_name, file=text_file)
        print(' '*20, 'Avg %d Trial Results'%args.ntrials, file=text_file)
        
        template = "-- Final Acc: {:.2f} +- {:.2f}, Time: {:.2f} +- {:.2f}"
        print(template.format(avg_acc, std_acc, avg_time, std_time), file=text_file)