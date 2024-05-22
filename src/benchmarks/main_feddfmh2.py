import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data import *
from src.models import *
from src.client import *
from src.clustering import *
from src.utils import *

torch.backends.cudnn.benchmark = True

def get_models_feddfmh(num_users, model, dataset, args):

    users_model = []

    for i in range(-1, num_users):
        if model == "resnet9":
            if dataset in ("cifar10"):
                net = ResNet9(in_channels=3, num_classes=10)
        elif model == "resnet18":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [2,2,2,2], num_classes=10)
        elif model == "resnet34":
            if dataset in ("cifar10"):
                net = ResNet(BasicBlock, [3,4,6,3], num_classes=10)
        elif model == "lenet5": 
            if dataset in ("cifar10"):
                net = LeNet5(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
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

def main_feddfmh2(args):

    path = args.path

    print(' ')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    #print(str(args))
    ##################################### Data partitioning section
    print('-'*40)
    print('Getting Clients Data')
    
    num_users = [20, 20, 4]
    archs = ["arch_s", "arch_m", "arch_l"]
    p_trains = [0.25, 0.25, 0.5]
    fracs = [0.05, 0.35, 0.25]
    
    num_users = args.num_users_per_cluster
    archs = args.archs
    p_trains = args.data_per_cluster
    fracs = args.frac_per_cluster
    
    print(num_users)
    print(archs)
    print(p_trains)
    print(fracs)
    
    public_train_ds, public_test_ds, _, \
    _ = get_dataset_global(args.distill_dataset, args.datadir, batch_size=128,
                                        p_train=1.0, p_test=1.0)
    
    p_data = torch.utils.data.ConcatDataset([public_train_ds, public_test_ds])
    soft_t = np.random.randn(len(p_data), 10)
    public_ds = DatasetKD(p_data, soft_t)
    
    train_ds_global, test_ds_global, train_dl_global, \
    test_dl_global = get_dataset_global(args.dataset, args.datadir, batch_size=128,
                                        p_train=1.0, p_test=1.0)
    
    x_train, y_train = train_ds_global.data, train_ds_global.target
    x_test, y_test = test_ds_global.data, test_ds_global.target

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test  = np.array(x_test)
    y_test  = np.array(y_test)
    
    partitions_train = []
    partitions_test = []
    partitions_train_stat = []
    partitions_test_stat = []
    
    p_start=0
    p_end=0
    for num, p in zip(num_users, p_trains):
        p_end+=int(p*len(train_ds_global))
        x_train_tmp, y_train_tmp = x_train[p_start:p_end], y_train[p_start:p_end]
        x_test_tmp, y_test_tmp = x_test, y_test
        p_start=p_end
        
        partitions_train_tmp, partitions_test_tmp, partitions_train_stat_tmp, \
        partitions_test_stat_tmp = partition_data(args.dataset, args.datadir, args.partition,
                                                num, niid_beta=args.niid_beta, iid_beta=args.iid_beta,
                                                p_train=1.0, p_test=1.0, 
                                                x_train=x_train_tmp, y_train=y_train_tmp, 
                                                x_test=x_test_tmp, y_test=y_test_tmp)
        
        partitions_train.append(partitions_train_tmp)
        partitions_test.append(partitions_test_tmp)
        partitions_train_stat.append(partitions_train_stat_tmp)
        partitions_test_stat.append(partitions_test_stat_tmp)


    print('-'*40)
    ################################### build model
    print('-'*40)
    print('Building models for clients')
    print(f'MODEL: {args.model}, Dataset: {args.dataset}')
    
    users_model = []
    net_glob = []
    initial_state_dict = []
    for num, arch in zip(num_users, archs):
        users_model_tmp, net_glob_tmp, initial_state_dict_tmp = get_models_feddfmh(num_users=num, model=arch, 
                                                                                 dataset=args.dataset, args=args)
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
    
    for cn, mod in enumerate(net_glob): 
        print(f'Model No {cn+1}')
        total = 0
        for name, param in mod.named_parameters():
            print(name, param.size())
            total += np.prod(param.size())
            #print(np.array(param.data.cpu().numpy().reshape([-1])))
            #print(isinstance(param.data.cpu().numpy(), np.array))
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

            train_ds_local = get_subset(train_ds_global, dataidxs)
            test_ds_local  = get_subset(test_ds_global, dataidxs_test)

            transform_train, transform_test = get_transforms(args.dataset, noise_level=0, net_id=None, total=0)

            train_dl_local = DataLoader(dataset=train_ds_local, batch_size=args.local_bs, shuffle=True, drop_last=False,
                                       num_workers=4, pin_memory=False)
            test_dl_local = DataLoader(dataset=test_ds_local, batch_size=64, shuffle=False, drop_last=False, num_workers=4,
                                      pin_memory=False)

            clients_tmp.append(Client_FedDFMH(idx, copy.deepcopy(users_model[cn][idx]), args.local_bs, args.local_ep,
                       args.lr, args.momentum, args.device, train_dl_local, test_dl_local))
        
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
    glob_acc_kd = [[] for _ in range(len(num_users))]

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
        
        if iteration > 85:
            ###### Logits Avg #######
            avg_logits = []
            for cn in range(len(num_users)):
                logits_locals = []
                for idx in idxs_users[cn]:
                    logits_locals.append(clients[cn][idx].inference(public_ds))

                np.mean(logits_locals, axis=0)
                avg_logits.append(np.mean(logits_locals, axis=0))
            ###### Logits Avg #######

            ##### Global Model KD #####
            for cn in range(len(num_users)):
                net_glob[cn].load_state_dict(copy.deepcopy(w_glob[cn]))
                net_glob[cn].to(args.device)
                
                if cn==0:
                    public_ds.set_logits(avg_logits[1])
                else:
                    public_ds.set_logits(avg_logits[0])
                    
                public_dl = torch.utils.data.DataLoader(public_ds, batch_size=128, shuffle=True, drop_last=False)
                steps = int(len(public_ds)/128)
        #         optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0)
                optimizer = torch.optim.Adam(net_glob[cn].parameters(), lr=args.distill_lr)
                #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

                kl_criterion = nn.KLDivLoss(reduction="batchmean")
                mse_criterion = nn.MSELoss()
                T = args.distill_T
                for _ in range(args.distill_E):
                    batch_loss = []
                    for batch_idx, d2 in enumerate(public_dl):
                        net_glob[cn].zero_grad()
                        #optimizer.zero_grad()

                        teacher_x, teacher_y, teacher_logits = d2
                        teacher_x, teacher_logits = teacher_x.to(args.device), teacher_logits.to(args.device)

                        logits_student = net_glob[cn](teacher_x)

                        kd_loss = kl_criterion(F.log_softmax(logits_student/T, dim=1), F.softmax(teacher_logits/T, dim=1))
                        #kd_loss = mse_criterion(F.softmax(logits_student/T, dim=1), F.softmax(teacher_logits/T, dim=1))/2
                        kd_loss.backward()
                        optimizer.step()
                    #scheduler.step()

                w_glob[cn] = copy.deepcopy(net_glob[cn].state_dict())
                _, acc_kd = eval_test(net_glob[cn], args, test_dl_global)
                glob_acc_kd[cn].append(acc_kd)
            ##### Global Model KD #####

            print("-- Results:")
            for cn in range(len(num_users)):
                template = "--[Cluster {:.1f}]: {}, Global Acc Wavg: {:.2f}, After KD: {:.2f}, Best: {:.2f}"
                print(template.format(cn, archs[cn], glob_acc_wavg[cn][-1], glob_acc_kd[cn][-1], 
                                      np.max(glob_acc_kd[cn])))
        else: 
            print("-- Results:")
            for cn in range(len(num_users)):
                template = "--[Cluster {:.1f}]: {}, Global Acc Wavg: {:.2f}"
                print(template.format(cn, archs[cn], glob_acc_wavg[cn][-1]))

        
        print_flag = False
        if iteration+1 in [int(0.10*args.rounds), int(0.25*args.rounds), int(0.5*args.rounds), int(0.8*args.rounds)]:
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
            temp_acc[cn].append(clients_local_acc[cn][k][-1])
            temp_best_acc[cn].append(np.max(clients_local_acc[cn][k]))

            template = ("Client {:3d}, Final_acc {:3.2f}, best_acc {:3.2f} \n")
            print(template.format(k, clients_local_acc[cn][k][-1], np.max(clients_local_acc[cn][k])))

    template = ("-- Avg Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_acc)))
    template = ("-- Avg Best Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_best_acc)))
    print('*'*25)
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
        final_glob.append(glob_acc_kd[cn][-1])
        avg_final_glob.append(np.mean(glob_acc_kd[cn][-10:]))
        best_glob.append(np.max(glob_acc_kd[cn]))
        
    avg_final_local = np.mean(temp_acc)
    avg_best_local = np.mean(temp_best_acc)

    return (final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, duration)

def run_feddfmh2(args, fname):
    alg_name = 'FedDFMH2'

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
        duration = main_feddfmh2(args)

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
