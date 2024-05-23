import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time 

from src.data import *
from src.models import *
from src.client import *
from src.utils import *
from src.nlp import model as nlp_model, dataset as nlp_dataset
from src.utils.utils import eval_test, AvgWeights

import concurrent.futures
import torch.cuda
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext

def main_fedmh(args):

    path = args.path

    print(' ')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    ##################################### Data partitioning section
    print('-'*40)
    print('Getting Clients Data')

    num_users = args.num_users
    archs = args.models
    fracs = args.fracs
    num_clusters = len(num_users)
    
    print(f'Num Users: {num_users}')
    print(f'archs: {archs}')
    print(f'fracs: {fracs}')
    print(f'Data ratios: {args.data_ratios}')

    if args.nlp: 
        public_ds = nlp_dataset.get_public_ds(args)
        train_ds_global, val_ds_global, test_ds_global = nlp_dataset.get_train_val_test_ds(args)
        
        test_dl_global = DataLoader(dataset=test_ds_global, batch_size=128, shuffle=False, drop_last=False)

        num_labels = nlp_dataset.hf_ds_config[args.dataset]['num_labels']

        partitions_train, partitions_test, partitions_train_stat, partitions_test_stat = \
            get_partitions_customD(train_ds_global, test_ds_global, args, num_labels)

    else: 
        public_ds = get_distill_data(args.distill_dataset, args.datadir, args.alg, args.dataset, num_clusters)

        train_ds_global, test_ds_global, train_dl_global, test_dl_global \
        = get_dataset_global(args.dataset, args.datadir, batch_size=128,
                                            p_train=1.0, p_test=1.0, seed=args.seed)
        
        partitions_train, partitions_test, partitions_train_stat, partitions_test_stat = \
            get_partitions_customD(train_ds_global, test_ds_global, args)
    

    Y_train = np.array(train_ds_global.target) if not args.nlp else np.array(train_ds_global['labels'])

    print('-'*40)
    ################################### build model
    print('-'*40)
    print('Building models for clients')
    
    users_model = []
    net_glob = []
    initial_state_dict = []
    for num, arch in zip(num_users, archs):
        users_model_tmp, net_glob_tmp, initial_state_dict_tmp = \
            get_models_fedmh(num_users=num, model=arch, dataset=args.dataset, args=args) \
            if not args.nlp else \
            nlp_model.get_models(num, arch, num_labels, args.datadir)
        users_model.append(users_model_tmp)
        net_glob.append(net_glob_tmp)
        initial_state_dict.append(initial_state_dict_tmp)
    
    if not args.nlp: 
        for cn, models in enumerate(users_model): 
            for mod in models: 
                mod.load_state_dict(initial_state_dict[cn])
        
        for cn, mod in enumerate(net_glob): 
            mod.load_state_dict(initial_state_dict[cn])

    num_params_list = []
    for cn, mod in enumerate(net_glob): 
        print(f'Model No {cn+1}')
        total = 0
        for name, param in mod.named_parameters():
            total += np.prod(param.size())
        num_params_list.append(total)
        print(f'total params {total}')
        print('-'*40)

    ################################# Initializing Clients
    print('-'*40)
    print('Initializing Clients')
    clients = []
    for cn in range(len(num_users)):
        print(f'---- Clients Group # {cn+1}')
        clients_tmp = []
        for idx in range(num_users[cn]):
            sys.stdout.flush()
            print(f'-- Client {idx}, Train Stat {partitions_train_stat[cn][idx]} Test Stat {partitions_test_stat[cn][idx]}')

            noise_level=0
            dataidxs = partitions_train[cn][idx]
            dataidxs_test = partitions_test[cn][idx]
            
            unq, unq_cnt = np.unique(Y_train[dataidxs], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            print(f'Actual Partition Stat: {tmp}')
            
            if args.nlp: 
                train_ds_local = train_ds_global.select(dataidxs)
                test_ds_local  = test_ds_global.select(dataidxs_test)
                
                bs = min(args.local_bs, len(train_ds_local))
                
                train_dl_local = DataLoader(dataset=train_ds_local, batch_size=bs, shuffle=True, drop_last=False,
                                            num_workers=4, pin_memory=True)
                test_dl_local = DataLoader(dataset=test_ds_local, batch_size=64, shuffle=False, drop_last=False, 
                                           num_workers=4, pin_memory=True)
                
            else: 
                train_ds_local = torch.utils.data.Subset(train_ds_global, dataidxs)
                test_ds_local  = torch.utils.data.Subset(test_ds_global, dataidxs_test)

                bs = min(args.local_bs, len(train_ds_local))
                def custom_collate(batch):
                    batch = list(filter(lambda x: x is not None, batch))  # Remove None elements
                    if len(batch) > 1:
                        return torch.utils.data.dataloader.default_collate(batch)
                    else:
                        return None
        
                train_dl_local = DataLoader(dataset=train_ds_local, batch_size=bs, shuffle=True, drop_last=False,
                                            collate_fn=custom_collate, num_workers=4, pin_memory=True)
                test_dl_local = DataLoader(dataset=test_ds_local, batch_size=64, shuffle=False, drop_last=False, 
                                            collate_fn=custom_collate, num_workers=4, pin_memory=True)

            optim = args.optim
            
            clients_tmp.append(Client_FedDFMH(idx, users_model[cn][idx], args.local_bs, args.local_ep[cn], optim,
                    args.lr[cn], args.momentum, args.local_wd[cn], args.lr_scheduler[cn], args.device, 
                    train_dl_local, test_dl_local, args.nlp))
        
        clients.append(clients_tmp)
    
    MIXED_PRECISION = clients[0][0].mixed_precision_training

    print('-'*40)
    ###################################### Federation
    print('Starting FL')
    print('-'*40)
    start = time.time()
    
    loss_train = []
    w_locals, loss_locals = [], []
    glob_acc_wavg = [[] for _ in range(len(num_users))]
    glob_acc_kd = [[] for _ in range(len(num_users))]

    w_glob = copy.deepcopy(initial_state_dict)

    for iteration in range(args.rounds):
        iter_start_time = time.time()
        idxs_users=[]
        for cn in range(len(num_users)):
            m = max(int(fracs[cn] * num_users[cn]), 1)
            idxs_users_tmp = np.random.choice(range(num_users[cn]), m, replace=False)
            idxs_users.append(idxs_users_tmp)
        
        print(f'----- ROUND {iteration+1} -----')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        sys.stdout.flush()
        s_time = time.time()
        for cn in range(len(idxs_users)):
            for idx in idxs_users[cn]:
                clients[cn][idx].set_state_dict(copy.deepcopy(w_glob[cn]))

                loss = clients[cn][idx].train()
                loss_locals.append(copy.deepcopy(loss))     
        e_time = time.time()

        loss_avg = sum(loss_locals) / len(loss_locals)
        template = '-- Average Train loss {:.3f}, Time: {:.3f}'
        print(template.format(loss_avg, e_time-s_time))
        ####### FedAvg ####### START
        for cn in range(len(num_users)):
            total_data_points = sum([len(partitions_train[cn][r]) for r in idxs_users[cn]])
            fed_avg_freqs = [len(partitions_train[cn][r]) / total_data_points for r in idxs_users[cn]]
            w_locals = []
            for idx in idxs_users[cn]:
                w_locals.append(copy.deepcopy(clients[cn][idx].get_state_dict()))

            ww = AvgWeights(w_locals, weight_avg=fed_avg_freqs)
            w_glob[cn] = copy.deepcopy(ww)
            net_glob[cn].load_state_dict(copy.deepcopy(ww))
            
            acc = eval_test(net_glob[cn], args, test_dl_global)

            glob_acc_wavg[cn].append(acc)
        ####### FedAvg ####### END
        
        ###### Logits Avg #######
        s_time = time.time()
        logits_locals = []
        for cn in range(len(num_users)):
            for idx in idxs_users[cn]:
                logits_locals.append(clients[cn][idx].inference(public_ds))
            torch.cuda.empty_cache()
        
        teacher_logits = torch.mean(torch.stack(logits_locals), dim=0)
        public_ds.set_logits(teacher_logits)
        e_time = time.time()
        print(f'Inference & Logit Avg done: {e_time-s_time:.2f}')
        ###### Logits Avg #######
        
        ##### Global Model KD #####
        for cn in range(len(num_users)):
            if MIXED_PRECISION: 
                scaler = GradScaler()

            net_glob[cn].load_state_dict(w_glob[cn])
            net_glob[cn].to(args.device)
            net_glob[cn].train()
            
            public_dl = torch.utils.data.DataLoader(public_ds, batch_size=32, shuffle=True, 
                                                    drop_last=False)
            optimizer = torch.optim.Adam(net_glob[cn].parameters(), lr=args.distill_lr, weight_decay=args.distill_wd[cn])

            kl_criterion = nn.KLDivLoss(reduction="batchmean")
            mse_criterion = nn.MSELoss()
            T = args.distill_T
            for _ in range(args.distill_E):
                batch_loss = []
                for batch_idx, d2 in enumerate(public_dl):
                    net_glob[cn].zero_grad()
                    
                    teacher_x, teacher_y, teacher_logits = d2
                    
                    with autocast() if MIXED_PRECISION else nullcontext(): 
                        if not args.nlp: 
                            teacher_x, teacher_logits = teacher_x.to(args.device), teacher_logits.to(args.device)
                            logits_student = net_glob[cn](teacher_x)
                        else: 
                            teacher_x = {k:v.to(args.device) for k,v in teacher_x.items() if k != 'labels'}
                            teacher_logits = teacher_logits.to(args.device)
                            logits_student = net_glob[cn](**teacher_x).logits

                        kd_loss = kl_criterion(F.log_softmax(logits_student/T, dim=1), F.softmax(teacher_logits/T, dim=1))
                    
                        loss = (T**2) * kd_loss

                    optimizer.zero_grad()
                    if MIXED_PRECISION: 
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:     
                        loss.backward()
                        optimizer.step()
                
            w_glob[cn] = copy.deepcopy(net_glob[cn].state_dict())
            acc_kd = eval_test(net_glob[cn], args, test_dl_global)
            net_glob[cn] = net_glob[cn].cpu()
            glob_acc_kd[cn].append(acc_kd)
        ##### Global Model KD #####
        print("-- Results:")
        for cn in range(len(num_users)):
            template = "--[Cluster {:.1f}]: {}, Global Acc Wavg: {:.2f}, After KD: {:.2f}, Best: {:.2f}"
            print(template.format(cn, archs[cn], glob_acc_wavg[cn][-1], glob_acc_kd[cn][-1], 
                                np.max(glob_acc_kd[cn])))

        loss_train.append(loss_avg)

        ## clear the placeholders for the next round
        loss_locals.clear()

        ## calling garbage collector
        #gc.collect()
        iter_end_time = time.time()

        print(f'Round {iteration+1} Time: {(iter_end_time - iter_start_time)/60:.1f} mins')

    end = time.time()
    duration = end-start
    print('-'*40)
    ############################### FedAvg Final Results
    print('-'*40)
    print('FINAL RESULTS')
    for cn in range(len(num_users)):
        template = "-- Global Acc Final Wavg: {:.2f}, KD: {:.2f}"
        print(template.format(glob_acc_wavg[cn][-1], glob_acc_kd[cn][-1]))

        template = "-- Global Acc Avg Final 10 Rounds: {:.2f}, After: {:.2f}"
        print(template.format(np.mean(glob_acc_wavg[cn][-10:]), np.mean(glob_acc_kd[cn][-10:])))

        template = "-- Global Best Acc: {:.2f}, After: {:.2f}"
        print(template.format(np.max(glob_acc_wavg[cn]), np.max(glob_acc_kd[cn])))
        print(f'-- FL Time: {duration/60:.2f} minutes')
    print('-'*40)
    
    final_glob = []
    avg_final_glob = []
    best_glob = []
    for cn in range(len(num_users)):
        final_glob.append(glob_acc_kd[cn][-1])
        kk = int(num_users[cn]*fracs[cn])
        avg_final_glob.append(np.mean(glob_acc_kd[cn][-kk:]))
        best_glob.append(np.max(glob_acc_kd[cn]))

    return (glob_acc_wavg, glob_acc_kd, final_glob, avg_final_glob, best_glob, duration)

def run_fedmh(args, fname):
    alg_name = 'FedMH'

    seeds = [2023, 2024, 2022, 2021]
    
    exp_glob_acc_wavg = []
    exp_glob_acc_kd = []
    exp_final_glob=[]
    exp_avg_final_glob=[]
    exp_best_glob=[]
    exp_fl_time=[]

    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))

        set_seed(seeds[trial])
        glob_acc_wavg, glob_acc_kd, final_glob, avg_final_glob, best_glob, duration \
            = main_fedmh(args)

        exp_glob_acc_wavg.append(glob_acc_wavg)
        exp_glob_acc_kd.append(glob_acc_kd)
        exp_final_glob.append(final_glob)
        exp_avg_final_glob.append(avg_final_glob)
        exp_best_glob.append(best_glob)
        exp_fl_time.append(duration/60)

        print('*'*40)
        print(' '*20, 'End of Trial %d'%(trial+1))
        print(' '*20, 'Final Results')

        template = "-- Global Final Acc: {}"
        r = [float(f'{item:.2f}') for item in exp_final_glob[-1]]
        print(template.format(r))

        template = "-- Global Avg Final 10 Rounds Acc : {}"
        r = [float(f'{item:.2f}') for item in exp_avg_final_glob[-1]]
        print(template.format(r))

        template = "-- Global Best Acc: {}"
        r = [float(f'{item:.2f}') for item in exp_best_glob[-1]]
        print(template.format(r))

        print(f'-- FL Time: {exp_fl_time[-1]:.2f} minutes')

    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, 'Avg %d Trial Results'%args.ntrials)

    template = "-- Global Final Acc: {} +- {}"
    r1 = [float(f'{item:.2f}') for item in np.mean(exp_final_glob, axis=0)]
    r2 = [float(f'{item:.2f}') for item in np.std(exp_final_glob, axis=0)]
    print(template.format(r1, r2))

    template = "-- Global Avg Final 10 Rounds Acc: {} +- {}"
    r1 = [float(f'{item:.2f}') for item in np.mean(exp_avg_final_glob, axis=0)]
    r2 = [float(f'{item:.2f}') for item in np.std(exp_avg_final_glob, axis=0)]
    print(template.format(r1, r2))

    template = "-- Global Best Acc: {} +- {}"
    r1 = [float(f'{item:.2f}') for item in np.mean(exp_best_glob, axis=0)]
    r2 = [float(f'{item:.2f}') for item in np.std(exp_best_glob, axis=0)]
    print(template.format(r1, r2))

    print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes')

    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40, file=text_file)
        print(' '*20, alg_name, file=text_file)
        print(' '*20, 'Avg %d Trial Results'%args.ntrials, file=text_file)

        template = "-- Global Final Acc: {} +- {}"
        r1 = [float(f'{item:.2f}') for item in np.mean(exp_final_glob, axis=0)]
        r2 = [float(f'{item:.2f}') for item in np.std(exp_final_glob, axis=0)]
        print(template.format(r1, r2), file=text_file)

        template = "-- Global Avg Final 10 Rounds Acc: {} +- {}"
        r1 = [float(f'{item:.2f}') for item in np.mean(exp_avg_final_glob, axis=0)]
        r2 = [float(f'{item:.2f}') for item in np.std(exp_avg_final_glob, axis=0)]
        print(template.format(r1, r2), file=text_file)

        template = "-- Global Best Acc: {} +- {}"
        r1 = [float(f'{item:.2f}') for item in np.mean(exp_best_glob, axis=0)]
        r2 = [float(f'{item:.2f}') for item in np.std(exp_best_glob, axis=0)]
        print(template.format(r1, r2), file=text_file)

        print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes', file=text_file)

        print('*'*40)
    
    print('Saving Global Accuracy')
    np.save(fname+'_glob_acc_wavg.npy', np.array(exp_glob_acc_wavg))
    np.save(fname+'_glob_acc_kd.npy', np.array(exp_glob_acc_kd))
    return