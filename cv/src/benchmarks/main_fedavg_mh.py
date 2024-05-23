import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import time

from src.data import *
from src.models import *
from src.client import *
from src.utils import *

import torch.cuda

def train_client(client, w_glob):
    client.set_state_dict(copy.deepcopy(w_glob))
    loss = client.train(is_print=False)
    return loss

def main_fedavg_mh(args, fname=None):

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
    
    public_ds = get_distill_data(args.distill_dataset, args.datadir, args.alg, args.dataset, num_clusters)
    
    train_ds_global, test_ds_global, train_dl_global, test_dl_global \
    = get_dataset_global(args.dataset, args.datadir, batch_size=128,
                                        p_train=1.0, p_test=1.0, seed=args.seed)
    
    if args.dataset in ["cifar10", "cifar100"]:
        Y_train = np.array(train_ds_global.target)
    elif args.dataset in ["cinic10"]:
        Y_train = np.array([el[1] for el in train_ds_global])
    elif args.dataset in ["tinyimagenet"]:
        Y_train = np.array([el[1] for el in train_ds_global])
    elif args.dataset in ["stl10"]:
        Y_train = np.array([el[1] for el in train_ds_global])

    partitions_train, partitions_test, partitions_train_stat, partitions_test_stat = \
        get_partitions_customD(train_ds_global, test_ds_global, args)

    print('-'*40)
    ################################### build model
    print('-'*40)
    print('Building models for clients')
    
    users_model = []
    net_glob = []
    initial_state_dict = []
    for num, arch in zip(num_users, archs):
        print(arch)
        users_model_tmp, net_glob_tmp, initial_state_dict_tmp = get_models_fedmh(num_users=num, model=arch, 
                                                                                dataset=args.dataset, args=args)
        users_model.append(users_model_tmp)
        net_glob.append(net_glob_tmp)
        initial_state_dict.append(initial_state_dict_tmp)
    
    for cn, models in enumerate(users_model): 
        for mod in models: 
            mod.load_state_dict(initial_state_dict[cn])
    
    for cn, mod in enumerate(net_glob): 
        mod.load_state_dict(initial_state_dict[cn])
        
    print('-'*40)
    print(net_glob)
    print('')
    
    num_params_list = []
    for cn, mod in enumerate(net_glob): 
        print(f'Model No {cn+1}')
        total = 0
        for name, param in mod.named_parameters():
            print(name, param.size())
            total += np.prod(param.size())
        num_params_list.append(total)
        print(f'total params {total}')
        print('-'*40)
        
    print(num_params_list)
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
        
            dataidxs = partitions_train[cn][idx]
            dataidxs_test = partitions_test[cn][idx]
            
            unq, unq_cnt = np.unique(Y_train[dataidxs], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            print(f'Actual Partition Stat: {tmp}')

            train_ds_local = torch.utils.data.Subset(train_ds_global, dataidxs)
            test_ds_local  = torch.utils.data.Subset(test_ds_global, dataidxs_test)

            bs = min(args.local_bs, len(train_ds_local))
    
            train_dl_local = DataLoader(dataset=train_ds_local, batch_size=bs, shuffle=True, drop_last=False,
                                        collate_fn=None, num_workers=4, pin_memory=True)
            test_dl_local = DataLoader(dataset=test_ds_local, batch_size=64, shuffle=False, drop_last=False, 
                                        collate_fn=None, num_workers=4, pin_memory=True)
            optim = args.optim
            clients_tmp.append(Client_FedMH(idx, copy.deepcopy(users_model[cn][idx]), args.local_bs, args.local_ep[cn], args.optim,
                    args.lr[cn], args.momentum, args.local_wd[cn], args.lr_scheduler[cn], args.device, train_dl_local, test_dl_local))
        
        clients.append(clients_tmp)
    print('-'*40)
    ###################################### Federation
    print('Starting FL')
    print('-'*40)
    start = time.time()
    
    loss_train = []
    clients_local_acc = {i:{j:[] for j in range(num_users[i])} for i in range(len(num_users))}
    w_locals, loss_locals = [], []
    glob_acc_wavg = [[] for _ in range(len(num_users))]

    w_glob = copy.deepcopy(initial_state_dict)

    for iteration in range(args.rounds):
        
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

                loss = clients[cn][idx].train(is_print=False)
                loss_locals.append(copy.deepcopy(loss))

        loss_avg = sum(loss_locals) / len(loss_locals)
        
        e_time = time.time()
        template = '-- Average Train loss {:.3f}, Time: {:.3f}'
        print(template.format(loss_avg, e_time-s_time))
        
        for cn in range(len(num_users)):
            total_data_points = sum([len(partitions_train[cn][r]) for r in idxs_users[cn]])
            fed_avg_freqs = [len(partitions_train[cn][r]) / total_data_points for r in idxs_users[cn]]
            w_locals = []
            for idx in idxs_users[cn]:
                w_locals.append(copy.deepcopy(clients[cn][idx].get_state_dict()))

            ww = AvgWeights(w_locals, weight_avg=fed_avg_freqs)
            w_glob[cn] = copy.deepcopy(ww)
            net_glob[cn].load_state_dict(copy.deepcopy(ww))
            
            _, acc = eval_test(net_glob[cn], args, test_dl_global)

            glob_acc_wavg[cn].append(acc)
        
        print("-- Results:")
        for cn in range(len(num_users)):
            template = "--[Cluster {:.1f}]: {}, Global Acc Wavg: {:.2f}, Best: {:.2f}"
            print(template.format(cn, archs[cn], glob_acc_wavg[cn][-1], 
                                np.max(glob_acc_wavg[cn])))

        loss_train.append(loss_avg)
        loss_locals.clear()
        gc.collect()

    end = time.time()
    duration = end-start
    print('-'*40)
    ############################### Testing Local Results
    print('*'*25)
    print('---- Testing Final Local Results ----')
    temp_acc = [[] for _ in range(len(num_users))]
    temp_best_acc = [[] for _ in range(len(num_users))]
    for cn in range(len(num_users)):
        for k in range(num_users[cn]):
            sys.stdout.flush()
            loss, acc = clients[cn][k].eval_test()
            clients_local_acc[cn][k].append(acc)
            temp_acc[cn].append(clients_local_acc[cn][k][-1].numpy())
            temp_best_acc[cn].append(np.max(clients_local_acc[cn][k]))

            template = ("Client {:3d}, Final_acc {:3.2f}, best_acc {:3.2f} \n")
            print(template.format(k, clients_local_acc[cn][k][-1], np.max(clients_local_acc[cn][k])))
    print('*'*25)
    ############################### FedAvg Final Results
    print('-'*40)
    print('FINAL RESULTS')
    for cn in range(len(num_users)):
        template = "-- Global Acc Final Wavg: {:.2f}"
        print(template.format(glob_acc_wavg[cn][-1]))

        template = "-- Global Acc Avg Final 10 Rounds: {:.2f}"
        print(template.format(np.mean(glob_acc_wavg[cn][-10:])))

        template = "-- Global Best Acc: {:.2f}"
        print(template.format(np.max(glob_acc_wavg[cn])))

        template = ("-- Avg Local Acc: {:3.2f}")
        print(template.format(np.mean(temp_acc[cn])))

        template = ("-- Avg Best Local Acc: {:3.2f}")
        print(template.format(np.mean(temp_best_acc[cn])))

        print(f'-- FL Time: {duration/60:.2f} minutes')
    print('-'*40)
    
    final_glob = []
    avg_final_glob = []
    best_glob = []
    for cn in range(len(num_users)):
        final_glob.append(glob_acc_wavg[cn][-1])
        kk = int(num_users[cn]*fracs[cn])
        avg_final_glob.append(np.mean(glob_acc_wavg[cn][-kk:]))
        best_glob.append(np.max(glob_acc_wavg[cn]))
    
    temp_acc = [item for sublist in temp_acc for item in sublist]
    temp_best_acc = [item for sublist in temp_best_acc for item in sublist]
    print(temp_acc)
    print(temp_best_acc)
    
    avg_final_local = np.mean(temp_acc)
    avg_best_local = np.mean(temp_best_acc)

    return (glob_acc_wavg, final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, duration)

def run_fedavg_mh(args, fname):
    alg_name = 'FedAvg-MH'

    seeds = [2023, 2024, 2022, 2021]
    
    exp_glob_acc_wavg = []
    exp_final_glob=[]
    exp_avg_final_glob=[]
    exp_best_glob=[]
    exp_avg_final_local=[]
    exp_avg_best_local=[]
    exp_fl_time=[]

    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))
        
        set_seed(seeds[trial])
        glob_acc_wavg, final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, \
        duration = main_fedavg_mh(args, fname)

        exp_glob_acc_wavg.append(glob_acc_wavg)
        exp_final_glob.append(final_glob)
        exp_avg_final_glob.append(avg_final_glob)
        exp_best_glob.append(best_glob)
        exp_avg_final_local.append(avg_final_local)
        exp_avg_best_local.append(avg_best_local)
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

        template = ("-- Avg Final Local Acc: {:3.2f}")
        print(template.format(exp_avg_final_local[-1]))

        template = ("-- Avg Best Local Acc: {:3.2f}")
        print(template.format(exp_avg_best_local[-1]))

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

    template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)))

    template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)))

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

        template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)), file=text_file)

        template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)), file=text_file)

        print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes', file=text_file)

        print('*'*40)

    print('Saving Global Accuracy')
    np.save(fname+'_glob_acc_wavg.npy', np.array(exp_glob_acc_wavg))
    return