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
from src.utils import tune_lambda, eval_test, split_val, AvgWeights

torch.backends.cudnn.benchmark = True

def inference(public_ds, net, args):
    public_dl = torch.utils.data.DataLoader(public_ds, batch_size=256, shuffle=False, drop_last=False)
    net.eval()
    net.to(args.device)
    
    outs = []
    for data, _,_,_ in public_dl:
        data = data.to(args.device)
        out = net(data)
        outs.append(out.detach().cpu())

    outputs = torch.cat(outs)#.numpy()
    return outputs

def heuristic(num_clusters=3, n_candidates=25, position=None):
    candidates = [[1/num_clusters for _ in range(num_clusters)]]
    
    if position:
        specific_values = []
        value = 0.99
        while value > 1/num_clusters:
            specific_values.append(value)
            value -= 0.1
        
        for value in specific_values:
            candidate = np.zeros(num_clusters)
            candidate[position] = value
            remaining_sum = 1 - value
            remaining_values = remaining_sum / (num_clusters - 1)
            for j in range(num_clusters):
                if j != position:
                    candidate[j] = remaining_values
            candidates.append(candidate)
    
    for exponent in [1, 5, 10]:
        for i in range(n_candidates):
            candidate = np.random.beta(a=1, b=100, size=num_clusters)
            candidate = candidate ** exponent
            candidate = np.sort(candidate)
            candidate = candidate / np.sum(candidate)
            candidates.append(candidate)   
    return candidates

def main_fedhd(args):
    path = args.path

    print(' ')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    #print(str(args))
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
    print(f'Task Weights: {args.task_weights}')
    
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

    train_ds_global, val_ds_global = split_val(train_ds_global)
    
    partitions_train, partitions_test, partitions_train_stat, partitions_test_stat = \
        get_partitions_customD(train_ds_global, test_ds_global, args)
    
    print('-'*40)
    ################################### build model
    users_model = []
    net_glob = []
    initial_state_dict = []
    for num, arch in zip(num_users, archs):
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

            unq, unq_cnt = np.unique(Y_train[dataidxs], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            print(f'Actual Partition Stat: {tmp}')
            
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
                                        collate_fn=None, num_workers=4, pin_memory=True)
            test_dl_local = DataLoader(dataset=test_ds_local, batch_size=64, shuffle=False, drop_last=False, 
                                        collate_fn=None, num_workers=4, pin_memory=True)
            
            optim = args.optim
            
            clients_tmp.append(Client_FedMH(idx, copy.deepcopy(users_model[cn][idx]), args.local_bs, args.local_ep[cn], optim,
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
        torch.cuda.empty_cache()
        sys.stdout.flush()
        s_time = time.time()
        for cn in range(len(idxs_users)):
            for idx in idxs_users[cn]:
                clients[cn][idx].set_state_dict(copy.deepcopy(w_glob[cn]))

                loss = clients[cn][idx].train(is_print=False)
                loss_locals.append(copy.deepcopy(loss))
        
        e_time = time.time()
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        template = '-- Average Train loss {:.2f}, Time: {:.2f}'
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
            
            _, acc = eval_test(net_glob[cn], args, test_dl_global)

            glob_acc_wavg[cn].append(acc)
        ####### FedAvg ####### END
        ###### Inference ######
        s_time = time.time()
        avg_logits_list = []
        for cn in range(len(num_users)):
            logits_locals = []
            for idx in idxs_users[cn]:
                logits_locals.append(clients[cn][idx].inference(public_ds))
            avg_logits_list.append(torch.mean(torch.stack(logits_locals), dim=0))
            
        public_ds.set_logits(avg_logits_list)
        
        self_logits_list = []
        for cn in range(len(num_users)):
            self_logits_list.append(inference(public_ds, net_glob[cn], args))
            
        public_ds.set_self_logits(self_logits_list)
        
        e_time = time.time()
        print(f'Inference & Logit Avg done: {e_time-s_time:.2f}')
        ###### Inference ######
        ##### Global Model KD #####
        public_dl = torch.utils.data.DataLoader(public_ds, batch_size=128, shuffle=True, drop_last=False)
        #public_dl = torch.utils.data.DataLoader(public_ds, batch_size=64, shuffle=True, drop_last=False)
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        new_w_glob = []
        for cn in range(num_clusters):
            T = args.distill_T
            T_self=20
            A_self=args.self_reg
            ft_weights = []
            s_time = time.time()
            for idxx in range(num_clusters):
                net_glob[cn].load_state_dict(copy.deepcopy(w_glob[cn]))
                net_glob[cn].to(args.device)
                net_glob[cn].train()
                optimizer = torch.optim.Adam(net_glob[cn].parameters(), lr=args.distill_lr, weight_decay=args.distill_wd[cn])

                for _ in range(args.distill_E):
                    batch_loss = []
                    for batch_idx, d2 in enumerate(public_dl):
                        net_glob[cn].zero_grad()
                        optimizer.zero_grad()
                        
                        teacher_x, teacher_y, teachers_logits, self_logits = d2
                        teacher_x = teacher_x.to(args.device)
                        teacher_logits = teachers_logits[idxx].to(args.device)
                        self_logit = self_logits[cn].to(args.device)

                        logits_student = net_glob[cn](teacher_x)
                        loss_self_kd = kl_criterion(F.log_softmax(logits_student/T_self, dim=1), 
                                                    F.softmax(self_logit/T_self, dim=1))
                        
                        loss_kd = kl_criterion(F.log_softmax(logits_student/T, dim=1), 
                                            F.softmax(teacher_logits/T, dim=1))
                        
                        loss = (T**2) * loss_kd + A_self[cn]*(T_self**2)*loss_self_kd
                        
                        optimizer.zero_grad()
                        loss.backward(retain_graph=False)
                        optimizer.step()

                ft_weights.append(copy.deepcopy(net_glob[cn].state_dict()))
            
            task_vectors = []
            tmp_w_glob = copy.deepcopy(w_glob[cn])
            for www in ft_weights:
                task_vectors.append(copy.deepcopy(www))
                for key in tmp_w_glob.keys():
                    www[key] = www[key].type_as(tmp_w_glob[key])
                    task_vectors[-1][key] = www[key] - tmp_w_glob[key]

            if args.lambda_tuning == "tune":
                tune_time = time.time()
                
                keys = [k for k,_ in net_glob[cn].named_parameters()]
                w_avg = tuple(v.to(torch.float32).detach().clone().requires_grad_().cpu() \
                            for _,v in net_glob[cn].named_parameters())
                task_vector_tuples = [tuple(v.detach().clone().requires_grad_().cpu() \
                                            for k,v in task_vector.items() if k in keys) \
                                                for task_vector in task_vectors]
                empty_model = copy.deepcopy(net_glob[cn]).to(args.device)
                _, names = tune_lambda.make_functional(empty_model)

                init_alphas = args.task_weights[cn] if iteration > 0 else None
                lambda_wrapper = tune_lambda.LambdaWrapper(
                                    num_clusters=num_clusters,
                                    device=args.device,
                                    alpha_init=init_alphas,
                                    w_avg=w_avg,
                                    task_vectors=task_vector_tuples,
                                    names=names,
                                    arch=empty_model
                                )
                lambda_wrapper.to(args.device)
                alphas = tune_lambda.train(
                            wrapper=lambda_wrapper, 
                            optimizer=torch.optim.AdamW(lambda_wrapper.parameters(), lr=0.05, weight_decay=0.),
                            epochs=5,
                            batch_size=64,
                            dataset=val_ds_global,
                            verbose=True,
                            verbose_freq=200
                        )
                args.task_weights[cn] = alphas
                str_alpha_list = [f'{alpha:.4f}' for alpha in alphas]
                print(f'Cluster {cn} task weights tuned: {str_alpha_list}')
                print(f'Tune_lambda time: {time.time() - tune_time:.2f}')
            elif args.lambda_tuning == "heuristic":
                val_dl_global = DataLoader(val_ds_global, batch_size=128, shuffle=False, num_workers=4)
                candidates = heuristic(num_clusters, args.n_candidates)
                acc_candidates = []
                for i, lamb in enumerate(candidates):
                    tmp_w_glob = copy.deepcopy(w_glob[cn])
                    for key in tmp_w_glob.keys():
                        example_tensor = next(iter(task_vectors[0].values()))
                        wl = torch.tensor(lamb, dtype=example_tensor.dtype, device=example_tensor.device)
                        
                        weighted_sum_tensors = torch.zeros_like(tmp_w_glob[key])
                        for jj in range(num_clusters):
                            weighted_sum_tensors += wl[jj] * task_vectors[jj][key]
                            
                        weighted_sum_tensors = weighted_sum_tensors.type_as(tmp_w_glob[key])
                        tmp_w_glob[key] = tmp_w_glob[key] + weighted_sum_tensors
                    
                    net_glob[cn].load_state_dict(tmp_w_glob)
                    _, acc_kd = eval_test(net_glob[cn], args, val_dl_global)
                    acc_candidates.append(acc_kd)
                
                max_ind = acc_candidates.index(max(acc_candidates))
                print(f'Best candidate: {np.round(candidates[max_ind], 4)}, accuracy: {acc_candidates[max_ind]:.2f}')
                args.task_weights[cn] = candidates[max_ind]
            elif args.lambda_tuning == "fixed":
                pass
            else:
                print("Lambda Tuning option not supported yet")
                sys.exit()
            
            tmp_w_glob = copy.deepcopy(w_glob[cn])
            for key in tmp_w_glob.keys():
                wl = args.task_weights[cn]            

                example_tensor = next(iter(task_vectors[0].values()))
                wl = torch.tensor(args.task_weights[cn], dtype=example_tensor.dtype, device=example_tensor.device)
                
                weighted_sum_tensors = torch.zeros_like(tmp_w_glob[key])
                for jj in range(num_clusters):
                    weighted_sum_tensors += wl[jj] * task_vectors[jj][key]
                    
                weighted_sum_tensors = weighted_sum_tensors.type_as(tmp_w_glob[key])
                tmp_w_glob[key] = tmp_w_glob[key] + weighted_sum_tensors

            new_w_glob.append(copy.deepcopy(tmp_w_glob))
            net_glob[cn].load_state_dict(copy.deepcopy(new_w_glob[cn]))
            _, acc_kd = eval_test(net_glob[cn], args, test_dl_global)
            glob_acc_kd[cn].append(acc_kd)
            net_glob[cn].load_state_dict(copy.deepcopy(w_glob[cn]))
            
            e_time = time.time()
            print(f'Training cluster {cn} finished: {e_time-s_time:.2f}, A_self: {A_self[cn]}')
        ### Finished Now Update the Global weights
        for cn in range(num_clusters):
            w_glob[cn] = copy.deepcopy(new_w_glob[cn])
        ##### Global Model KD #####
        print("-- Results:")
        for cn in range(len(num_users)):
            template = "--[Cluster {:.1f}]: {}, Global Acc Wavg: {:.2f}, After KD: {:.2f}, Best: {:.2f}"
            print(template.format(cn, archs[cn], glob_acc_wavg[cn][-1], glob_acc_kd[cn][-1], 
                                np.max(glob_acc_kd[cn])))

        loss_train.append(loss_avg)
        loss_locals.clear()

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
        kk = int(num_users[cn]*fracs[cn])
        avg_final_glob.append(np.mean(glob_acc_kd[cn][-kk:]))
        best_glob.append(np.max(glob_acc_kd[cn]))
    
    temp_acc = [item for sublist in temp_acc for item in sublist]
    temp_best_acc = [item for sublist in temp_best_acc for item in sublist]
    print(temp_acc)
    print(temp_best_acc)
    
    avg_final_local = np.mean(temp_acc)
    avg_best_local = np.mean(temp_best_acc)

    return (glob_acc_wavg, glob_acc_kd, final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, duration)

def run_fedhd(args, fname):
    alg_name = 'FedHD'

    seeds = [2023, 2024, 2022, 2021]
    
    exp_glob_acc_wavg = []
    exp_glob_acc_kd = []
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

        #set_seed(seeds[trial])
        glob_acc_wavg, glob_acc_kd, final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, \
        duration = main_fedhd(args)

        exp_glob_acc_wavg.append(glob_acc_wavg)
        exp_glob_acc_kd.append(glob_acc_kd)
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
    np.save(fname+'_glob_acc_kd.npy', np.array(exp_glob_acc_kd))
    return