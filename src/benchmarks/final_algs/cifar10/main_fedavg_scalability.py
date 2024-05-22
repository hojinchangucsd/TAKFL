import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import random
import timm

from src.data import *
from src.models import *
from src.client import *
from src.clustering import *
from src.utils import *

torch.backends.cudnn.benchmark = True

def get_models_scalability(num_users, model, dataset, args, config=None):

    users_model = []

    for i in range(-1, num_users):
        if model == "resnet":
            if config is None:
                if dataset in ("cifar10"):
                    net = ResNet8(BasicBlock, [1,1,1], scaling=1.0, num_classes=10)
            else:
                if dataset in ("cifar10"):
                    net = ResNet(BasicBlock, config, num_classes=10)
        elif model == "timm":
            if dataset in ("cifar10"):
                    net = timm.create_model(config, pretrained=False, num_classes=10)
        else:
            print("not supported yet")
            sys.exit()
        
        if i == -1:
            net_glob = copy.deepcopy(net)
            net_glob.apply(weight_init)
            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)
        else:
            users_model.append(copy.deepcopy(net))
            users_model[i].load_state_dict(initial_state_dict)

    return users_model, net_glob, initial_state_dict

def main_fedavg_scalability(n_clusters, args):

    path = args.path

    print(' ')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print('')
    print('')
    print(f'---------------------- {n_clusters} Clusters --------------------------')
    #print(str(args))
    ##################################### Data partitioning section
    print('-'*40)
    print('Getting Clients Data')
    
    #num_users = [100, 20, 4]
    #archs = ['resnet8', 'resnet14', 'resnet18']
    #archs = ['vgg7', 'vgg11', 'vgg16']
    #fracs = [0.1, 0.2, 0.5]
    
    a = np.arange(10)+1
    NN=50000
    data_per_cluster = a*(NN/sum(a)).astype('int')
    total_num_users = 110
    num_users_per_cluster = a*(total_num_users/sum(a))
    num_users_per_cluster  = num_users_per_cluster[::-1].astype('int')
    #num_users_per_cluster = np.array([21., 19., 16., 14., 10., 8.,  7.,  6.,  4.,  2.]).astype('int')
    #num_users_per_cluster = np.array([21., 19., 17., 14., 10., 8.,  7.,  6.,  4.,  2.]).astype('int')
    num_users_per_cluster = np.array([21., 19., 17., 15., 10., 8.,  7.,  6.,  4.,  2.]).astype('int')
    
    ratio = 2.128
    params_per_cluster = a*(ratio)
    
    #fracs_per_cluster = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.5, 1.0])
    #fracs_per_cluster = np.array([0.2, 0.25, 0.25, 0.3, 0.35, 0.4, 0.5, 0.5, 0.5, 1.0])
    fracs_per_cluster = np.array([0.2, 0.25, 0.20, 0.3, 0.35, 0.4, 0.5, 0.5, 0.75, 1.0])
    c_per_round = fracs_per_cluster*num_users_per_cluster
    c_per_round = c_per_round.astype('int')
    
    config10 = [3,4,6,3]
    config9 = [3,4,4,3]
    config8 = [3,4,3,3]
    config7 = [3,4,3,2]
    config6 = [3,4,2,2]
    config5 = [3,4,1,2]
    config4 = [2,3,3,1]
    config3 = [2,2,2,1]
    config2 = [2,1,1,1]
    
    #configs = [None, config2, config3, config4, config5, config6, config7, config8, config9, config10]
   # net = ResNet(BasicBlock, config2, num_classes=10)
    configs = ['regnetx_002', 'resnet10t', 'resnet14t', 'resnet18', 'resnet26', 'resnet32ts', 
            'resnet33ts', 'resnet34', 'resnet50', 'resnest50d']
    
    num_users = num_users_per_cluster[0:n_clusters]
    fracs = fracs_per_cluster[0:n_clusters]
    archs = configs[0:n_clusters]
    
    print(num_users)
    print(archs)
    print(fracs)
    
    public_train_ds, public_test_ds, _, \
    _ = get_dataset_global(args.distill_dataset, args.datadir, batch_size=128,
                                        p_train=1.0, p_test=1.0, seed=args.seed)
    
    transform_train = transforms.Compose([
            transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    train_ds_global = datasets.CIFAR10(root=args.datadir, train=True, transform=transform_train, download=False)
    test_ds_global = datasets.CIFAR10(root=args.datadir, train=False, transform=transform_test, download=False)
    
    train_dl_global = DataLoader(dataset=train_ds_global, batch_size=128, shuffle=True, drop_last=False)
    test_dl_global = DataLoader(dataset=test_ds_global, batch_size=128, shuffle=False, drop_last=False)
    
    X_train = train_ds_global.data
    Y_train = np.array(train_ds_global.targets)

    X_test = test_ds_global.data
    Y_test = np.array(test_ds_global.targets)
    
    indices = list(range(len(train_ds_global)))
    random.shuffle(indices)

    partitions_train = []
    partitions_test = []
    partitions_train_stat = []
    partitions_test_stat = []
    
    cntt = 0
    for k in range(len(num_users)):
        inds_subset = indices[cntt:cntt+data_per_cluster[k]]
        if k == 9:
            inds_subset = indices[cntt:]

        Y_train_temp = Y_train[inds_subset]

        partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
        partitions_test_stat_tmp= dir_partition(num_users[k], niid_beta=args.niid_beta, nclasses=10, 
                                                y_train=Y_train_temp, y_test=Y_test, train_inds=inds_subset)

        partitions_train.append(partitions_train_tmp)
        partitions_test.append(partitions_test_tmp)
        partitions_train_stat.append(partitions_train_stat_tmp)
        partitions_test_stat.append(partitions_test_stat_tmp)
        
        cntt+= data_per_cluster[k]


    print('-'*40)
    ################################### build model
    print('-'*40)
    print('Building models for clients')
    print(f'MODEL: {args.model}, Dataset: {args.dataset}')
    
    users_model = []
    net_glob = []
    initial_state_dict = []
    for num, arch in zip(num_users, archs):
        users_model_tmp, net_glob_tmp, initial_state_dict_tmp = get_models_scalability(num_users=num, model='timm', 
                                                                                 dataset=args.dataset, args=args, config=arch)
        users_model.append(users_model_tmp)
        net_glob.append(net_glob_tmp)
        initial_state_dict.append(initial_state_dict_tmp)
    
    for cn, models in enumerate(users_model): 
        for mod in models: 
            mod.load_state_dict(initial_state_dict[cn])
    
    for cn, mod in enumerate(net_glob): 
        mod.load_state_dict(initial_state_dict[cn])
        
    #initial_state_dict = nn.DataParallel(initial_state_dict)
    #net_glob = nn.DataParallel(net_glob)
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
            #print(np.array(param.data.cpu().numpy().reshape([-1])))
            #print(isinstance(param.data.cpu().numpy(), np.array))
        num_params_list.append(total)
        print(f'total params {total}')
        print('-'*40)
        
    print(num_params_list)
    for cn in range(len(num_users)):
        scale = [num_params_list[i]/num_params_list[cn] for i in range(len(num_users))]
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

            train_ds_local = get_subset(train_ds_global, dataidxs)
            test_ds_local  = get_subset(test_ds_global, dataidxs_test)

            transform_train, transform_test = get_transforms(args.dataset, noise_level=0, net_id=None, total=0)

            bs = min(args.local_bs, len(train_ds_local))
            train_dl_local = DataLoader(dataset=train_ds_local, batch_size=bs, shuffle=True, drop_last=False,
                                       num_workers=4, pin_memory=True)
            test_dl_local = DataLoader(dataset=test_ds_local, batch_size=64, shuffle=False, drop_last=False, num_workers=4,
                                      pin_memory=True)

            clients_tmp.append(Client_FedMH(idx, copy.deepcopy(users_model[cn][idx]), args.local_bs, args.local_ep,
                       args.lr, args.momentum, args.local_wd, args.device, train_dl_local, test_dl_local))
        
        clients.append(clients_tmp)
    print('-'*40)
    ###################################### Federation
    print('Starting FL')
    print('-'*40)
    start = time.time()
    
    num_users_FL = args.num_users

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
        sys.stdout.flush()
        for cn in range(len(idxs_users)):
            for idx in idxs_users[cn]:
                #print(f'cn {cn} \n idx {idx}')
                clients[cn][idx].set_state_dict(copy.deepcopy(w_glob[cn]))

                loss = clients[cn][idx].train(is_print=False)
                loss_locals.append(copy.deepcopy(loss))

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        template = '-- Average Train loss {:.3f}'
        print(template.format(loss_avg))
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
            
            _, acc = eval_test(net_glob[cn], args, test_dl_global)

            glob_acc_wavg[cn].append(acc)
        ####### FedAvg ####### END
        
        print("-- Results:")
        for cn in range(len(num_users)):
            template = "--[Cluster {:.1f}]: {}, Global Acc Wavg: {:.2f}, Best: {:.2f}"
            print(template.format(cn, archs[cn], glob_acc_wavg[cn][-1], 
                                  np.max(glob_acc_wavg[cn])))

        print_flag = False
        if iteration+1 in [int(0.5*args.rounds), int(0.8*args.rounds)]:
            print_flag = True

        if print_flag:
            print('*'*25)
            print(f'Check Point @ Round {iteration+1} --------- {int((iteration+1)/args.rounds*100)}% Completed')
            temp_acc = []
            temp_best_acc = []
            for cn in range(len(num_users)):
                for k in range(num_users[cn]):
                    sys.stdout.flush()
                    loss, acc = clients[cn][k].eval_test()
                    clients_local_acc[cn][k].append(acc)
                    temp_acc.append(clients_local_acc[cn][k][-1])
                    temp_best_acc.append(np.max(clients_local_acc[cn][k]))

                    template = ("Client {:3d}, current_acc {:3.2f}, best_acc {:3.2f}")
                    print(template.format(k, clients_local_acc[cn][k][-1], np.max(clients_local_acc[cn][k])))

            #print('*'*25)
            template = ("-- Avg Local Acc: {:3.2f}")
            print(template.format(np.mean(temp_acc)))
            template = ("-- Avg Best Local Acc: {:3.2f}")
            print(template.format(np.mean(temp_best_acc)))
            print('*'*25)

        loss_train.append(loss_avg)

        ## clear the placeholders for the next round
        loss_locals.clear()

        ## calling garbage collector
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

#     template = ("-- Avg Local Acc: {:3.2f}")
#     temp_acc = np.array(temp_acc).flatten()
#     print(template.format(temp_acc))
#     template = ("-- Avg Best Local Acc: {:3.2f}")
#     temp_best_acc = np.array(temp_best_acc).flatten()
#     print(template.format(np.mean(temp_best_acc)))
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

    return (final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, duration)

def run_fedavg_scalability(n_clusters, args, fname):
    alg_name = 'FedAvg_scalability'

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

        final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, \
        duration = main_fedavg_scalability(n_clusters, args)

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

    return