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
from src.nlp import model as nlp_model, dataset as nlp_dataset

import concurrent.futures
import torch.cuda

#torch.backends.cudnn.benchmark = False

#torch.set_num_threads(1)
#torch.set_num_interop_threads(1)

def train_client(client, w_glob):
    client.set_state_dict(copy.deepcopy(w_glob))
    loss = client.train(is_print=False)
    return loss

def main_fedavg_mh(args):

    path = args.path

    print(' ')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    #print(str(args))
    ##################################### Data partitioning section
    print('-'*40)
    print('Getting Clients Data')
    
    #num_users = [100, 20, 4]
    #fracs = [0.1, 0.2, 0.5]
    
    # if args.arch_family == 'resnet':
    #     archs = ['resnet8', 'resnet14', 'resnet18']
    # elif args.arch_family == 'vgg':
    #     archs = ['vgg7', 'vgg11', 'vgg16']
    # elif args.arch_family == 'hetero':
    #     #archs = ['regnetx_002', 'vgg12', 'resnet18']
    #     #archs = ['edgenext_x_small', 'vgg12', 'resnet18']
    #     archs = ['vgg7', 'vgg11', 'resnet18']
        
    #num_users = args.num_users_per_cluster
    #archs = args.archs
    #p_trains = args.data_per_cluster
    #fracs = args.frac_per_cluster
    
    num_users = args.num_users
    archs = args.models
    fracs = args.fracs
    num_clusters = len(num_users)
    
    print(f'Num Users: {num_users}')
    print(f'archs: {archs}')
    print(f'fracs: {fracs}')
    print(f'Data ratios: {args.data_ratios}')
    
    #public_ds = get_distill_data(args.distill_dataset, args.datadir, args.alg, args.dataset, num_clusters)
    
    if args.nlp: 
        train_ds_global, val_ds_global, test_ds_global = nlp_dataset.get_train_val_test_ds(args)

        test_dl_global = DataLoader(dataset=test_ds_global, batch_size=128, shuffle=False, drop_last=False)
        
        num_labels = nlp_dataset.hf_ds_config[args.dataset]['num_labels']

        partitions_train, partitions_test, partitions_train_stat, partitions_test_stat = \
            get_partitions_customD(train_ds_global, test_ds_global, args, num_labels)

    else: 
        train_ds_global, test_ds_global, train_dl_global, test_dl_global \
        = get_dataset_global(args.dataset, args.datadir, batch_size=128,
                                            p_train=1.0, p_test=1.0, seed=args.seed)
    
        #partitions_train, partitions_test, partitions_train_stat, partitions_test_stat = get_partitions(num_users, train_ds_global, test_ds_global, args)

        partitions_train, partitions_test, partitions_train_stat, partitions_test_stat = \
            get_partitions_customD(train_ds_global, test_ds_global, args)
        
    if args.dataset in ["cifar10", "cifar100"]:
        Y_train = np.array(train_ds_global.target)
    elif args.dataset in ["cinic10", "tinyimagenet", "food101"]:
        Y_train = np.array([el[1] for el in train_ds_global])
    elif args.nlp: 
        Y_train = np.array(train_ds_global['labels'])
    
    print('-'*40)
    ################################### build model
    print('-'*40)
    print('Building models for clients')
    #print(f'MODEL: {args.model}, Dataset: {args.dataset}')
    
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
        
    #initial_state_dict = nn.DataParallel(initial_state_dict)
    #net_glob = nn.DataParallel(net_glob)
    #print('-'*40)
    #print(net_glob)
    #print('')
    
    #num_params_list = []
    #for cn, mod in enumerate(net_glob): 
    #    print(f'Model No {cn+1}')
    #    total = 0
    #    for name, param in mod.named_parameters():
    #        print(name, param.size())
    #        total += np.prod(param.size())
    #        #print(np.array(param.data.cpu().numpy().reshape([-1])))
    #        #print(isinstance(param.data.cpu().numpy(), np.array))
    #    num_params_list.append(total)
    #    print(f'total params {total}')
    #    print('-'*40)
        
    #print(num_params_list)
    #for cn in range(len(num_users)):
    #    scale = [num_params_list[i]/num_params_list[cn] for i in range(len(num_users))]
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
    print('-'*40)
    ###################################### Federation
    print('Starting FL')
    print('-'*40)
    start = time.time()
    
    loss_train = []
    #clients_local_acc = {i:{j:[] for j in range(num_users[i])} for i in range(len(num_users))}
    w_locals, loss_locals = [], []
    glob_acc_wavg = [[] for _ in range(len(num_users))]

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

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        #template = '-- Average Train loss {:.3f}'
        #print(template.format(loss_avg))
        
        # jobs, results = [], []
        # #max_len = sum(len(user) for user in idxs_users) - 4
        # max_len = 1
        # streams = [torch.cuda.Stream() for _ in range(max_len)]

        # # Execute tasks with limited concurrency
        # with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_len, os.cpu_count() - 1)) as executor:
        #     for i, (cn, idx) in enumerate([(cn, idx) for cn in range(len(idxs_users)) for idx in idxs_users[cn]]):
        #         stream = streams[i % max_len]  # Select stream
        #         #torch.cuda.synchronize()
        #         with torch.cuda.stream(stream):
        #             jobs.append(executor.submit(train_client, clients[cn][idx], copy.deepcopy(w_glob[cn])))

        # concurrent.futures.wait(jobs)
        # for stream in streams:
        #     with torch.cuda.stream(stream):
        #         torch.cuda.synchronize()

        # loss_locals = [job.result() for job in jobs]
        # e_time = time.time()
        # # Gather results
        # #loss_locals = results

        # # print loss
        # loss_avg = sum(loss_locals) / len(loss_locals)
        e_time = time.time()
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
        ###### Saving Weights #######
        #save_path = '/home/srip23/results_scalability/fedavg_mh/cinic10/niid-labeldir/0.5/Model-4-L'
        #model_name = 'logs_7Dev_Archs_resnet10_xxs_resnet10_s_resnet10_m_resnet10_l_resnet10_resnet18_resnet50_{:s}.pt'
        #def get_name(round, name, who): 
        #    return model_name.format(f'rounds{round}_global-type-{name}_{who}')
        # 
        #if iteration+1 in [1,5,10,20]:
        #    for cn in range(len(num_users)):
        #        if cn == 4: 
        #            name = get_name(iteration+1, cn, f'global')
        #            torch.save(net_glob[cn].state_dict(), os.path.join(save_path,name))
        #        for cc in range(num_users[cn]): 
        #            if cn == 4: 
        #                name = get_name(iteration+1, cn, f'client-{cc}')
        #                torch.save(clients[cn][cc].net.state_dict(), os.path.join(save_path,name))

        ##### Saving Weights #######
        
        print("-- Results:")
        for cn in range(len(num_users)):
            template = "--[Cluster {:.1f}]: {}, Global Acc Wavg: {:.2f}, Best: {:.2f}"
            print(template.format(cn, archs[cn], glob_acc_wavg[cn][-1], 
                                np.max(glob_acc_wavg[cn])))

        #print_flag = False
        ## if iteration+1 in [int(0.5*args.rounds), int(0.8*args.rounds)]:
        ##     print_flag = True
#
        #if print_flag:
        #    print('*'*25)
        #    print(f'Check Point @ Round {iteration+1} --------- {int((iteration+1)/args.rounds*100)}% Completed')
        #    temp_acc = []
        #    temp_best_acc = []
        #    for cn in range(len(num_users)):
        #        for k in range(num_users[cn]):
        #            sys.stdout.flush()
        #            loss, acc = clients[cn][k].eval_test()
        #            clients_local_acc[cn][k].append(acc)
        #            temp_acc.append(clients_local_acc[cn][k][-1])
        #            temp_best_acc.append(np.max(clients_local_acc[cn][k]))
#
        #            template = ("Client {:3d}, current_acc {:3.2f}, best_acc {:3.2f}")
        #            print(template.format(k, clients_local_acc[cn][k][-1], np.max(clients_local_acc[cn][k])))
#
        #    #print('*'*25)
        #    template = ("-- Avg Local Acc: {:3.2f}")
        #    print(template.format(np.mean(temp_acc)))
        #    template = ("-- Avg Best Local Acc: {:3.2f}")
        #    print(template.format(np.mean(temp_best_acc)))
        #    print('*'*25)

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
    ############################### Testing Local Results
    #print('*'*25)
    #print('---- Testing Final Local Results ----')
    #temp_acc = [[] for _ in range(len(num_users))]
    #temp_best_acc = [[] for _ in range(len(num_users))]
    #for cn in range(len(num_users)):
    #    for k in range(num_users[cn]):
    #        sys.stdout.flush()
    #        loss, acc = clients[cn][k].eval_test()
    #        clients_local_acc[cn][k].append(acc)
    #        temp_acc[cn].append(clients_local_acc[cn][k][-1].numpy())
    #        temp_best_acc[cn].append(np.max(clients_local_acc[cn][k]))
#
    #        template = ("Client {:3d}, Final_acc {:3.2f}, best_acc {:3.2f} \n")
    #        print(template.format(k, clients_local_acc[cn][k][-1], np.max(clients_local_acc[cn][k])))
    #print('*'*25)
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

        #template = ("-- Avg Local Acc: {:3.2f}")
        #print(template.format(np.mean(temp_acc[cn])))
#
        #template = ("-- Avg Best Local Acc: {:3.2f}")
        #print(template.format(np.mean(temp_best_acc[cn])))

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
    
    #temp_acc = [item for sublist in temp_acc for item in sublist]
    #temp_best_acc = [item for sublist in temp_best_acc for item in sublist]
    #print(temp_acc)
    #print(temp_best_acc)
    #
    #avg_final_local = np.mean(temp_acc)
    #avg_best_local = np.mean(temp_best_acc)

    return (glob_acc_wavg, final_glob, avg_final_glob, best_glob, duration)

def run_fedavg_mh(args, fname):
    alg_name = 'FedAvg-MH'

    seeds = [2023, 2024, 2022, 2021]
    #seeds = [102023, 552024, 912022, 212021]
    
    exp_glob_acc_wavg = []
    exp_final_glob=[]
    exp_avg_final_glob=[]
    exp_best_glob=[]
    #exp_avg_final_local=[]
    #exp_avg_best_local=[]
    exp_fl_time=[]

    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))
        
        set_seed(seeds[trial])
        glob_acc_wavg, final_glob, avg_final_glob, best_glob, duration = main_fedavg_mh(args)

        exp_glob_acc_wavg.append(glob_acc_wavg)
        exp_final_glob.append(final_glob)
        exp_avg_final_glob.append(avg_final_glob)
        exp_best_glob.append(best_glob)
        #exp_avg_final_local.append(avg_final_local)
        #exp_avg_best_local.append(avg_best_local)
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

        #template = ("-- Avg Final Local Acc: {:3.2f}")
        #print(template.format(exp_avg_final_local[-1]))
#
        #template = ("-- Avg Best Local Acc: {:3.2f}")
        #print(template.format(exp_avg_best_local[-1]))

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

    #template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
    #print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)))
#
    #template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
    #print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)))

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

        #template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
        #print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)), file=text_file)
#
        #template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
        #print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)), file=text_file)

        print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes', file=text_file)

        print('*'*40)

    print('Saving Global Accuracy')
    np.save(fname+'_glob_acc_wavg.npy', np.array(exp_glob_acc_wavg))
    #np.load(fname+'_glob_acc_wavg.npy')
    return