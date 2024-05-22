import argparse
import json

## CIFAR-10 has 50000 training images (5000 per class), 10 classes, 10000 test images (1000 per class)
## CIFAR-100 has 50000 training images (500 per class), 100 classes, 10000 test images (100 per class)
## MNIST has 60000 training images (min: 5421, max: 6742 per class), 10000 test images (min: 892, max: 1135
## per class) --> in the code we fixed 5000 training image per class, and 900 test image per class to be
## consistent with CIFAR-10

## CIFAR-10 Non-IID 250 samples per label for 2 class non-iid is the benchmark (500 samples for each client)

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_threads', type=int, default=0)
    # federated arguments
    parser.add_argument('--rounds', type=int, default=500, help="rounds of training")
    #parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    #parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', nargs="+", type=int, default=[20, 20, 20], help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    #parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    #parser.add_argument('--local_wd', type=float, default=0.0, help="local weight decay")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--warmup_epoch', type=int, default=0, help="the number of pretrain local ep")
    parser.add_argument('--ntrials', type=int, default=1, help="the number of trials")
    parser.add_argument('--log_filename', type=str, default=None, help='The log file name')
    parser.add_argument('--p_train', type=float, default=1.0, help="Percentage of Train Data")
    parser.add_argument('--p_test', type=float, default=1.0, help="Percentage of Test Data")
    #parser.add_argument('--model', type=str, default='lenet5', help='model name')
    parser.add_argument('--optim', type=str, default='adam', help='Local Optimizer')
    parser.add_argument('--train_size', type=int, default=100000, help='Traning set size limit')
    parser.add_argument('--public_size', type=int, default=30000, help='Public set size limit')
    parser.add_argument('--n_candidates', type=int, default=10)

    parser.add_argument('--tune_lambda', type=str, default='no_tuning', choices=['no_tuning', 'backprop', 'heuristic'])

    # FedDFMH
    parser.add_argument('--distill_lr', type=float, default=0.01, help="Distillation learning rate")
    parser.add_argument('--distill_T', type=float, default=1.0, help="Distillation Temprature")
    parser.add_argument('--distill_E', type=int, default=10, help="Distillation Epoch")
    parser.add_argument('--distill_dataset', type=str, default='cifar100', help="Distillation Dataset cifar100")
    parser.add_argument('--distill_wd', nargs='*', default=[5e-5, 5e-5, 5e-5], type=float,
                        help='Distillation Wd. Default is [5e-5, 5e-5, 5e-5].')
    parser.add_argument('--frac_per_cluster', nargs='+', type=float)
    parser.add_argument('--num_users_per_cluster', nargs='+', type=int)
    parser.add_argument('--data_per_cluster', nargs='+', type=float)
    parser.add_argument('--archs', nargs='+')
    parser.add_argument('--self_reg', nargs="+", type=float, default=[0.5, 0.5, 0.5], help="Self-Guarded Regularizer")
    # parser.add_argument(
    #     "--adaptive_weight_T",
    #     nargs="+",
    #     type=float,
    #     default=[3.0, 3.0, 3.0],
    #     help="List of float values for adaptive_weight_T for ensembles of teachers",
    # )
    parser.add_argument('--adaptive_weight_T', nargs='*', default=[2, 2, 2], type=float,
                        help="the fraction of clients for each cluster")
    
    # dataset partitioning arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help="name of dataset: mnist, cifar10, cifar100")

    # NIID Benchmark dataset partitioning
    parser.add_argument('--savedir', type=str, default='../save/', help='save directory')
    parser.add_argument('--datadir', type=str, default='../data/', help='data directory')
    parser.add_argument('--logdir', type=str, default='../logs/', help='logs directory')
    parser.add_argument('--partition', type=str, default='noniid-#label2', help='method of partitioning')
    parser.add_argument('--alg', type=str, default='cluster_fl', help='Algorithm')
    parser.add_argument('--batch_size', type=int, default=64, help="test batch size")
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--clustering_setting', type=str, default='3_clusters', help='clustering setting')
    parser.add_argument('--arch_family', type=str, default='resnet', help='Architecture Family')
    parser.add_argument('--old_type', action='store_true', help='Old Type Partitioning')

    ## Clustering
    parser.add_argument('--nclusters', type=int, default=3, help='Number of Clusters')
    parser.add_argument('--data_ratios', nargs='*', default=[0.8, 0.1, 0.1], type=float,
                        help='List of dataset ratios. Default is [0.8, 0.1, 0.1].')
    parser.add_argument('--models', nargs='*', default=['resnet8', 'resnet18', 'resnet50'], type=str,
                        help="List of model names. Default is ['resnet8', 'resnet18', 'resnet50'].")
    parser.add_argument('--num_users', nargs='*', default=[100, 20, 4], type=int, 
                        help="number of users for each cluster")
    parser.add_argument('--fracs', nargs='*', default=[0.4, 0.4, 0.4], type=float,
                        help="the fraction of clients for each cluster")
    parser.add_argument('--lr', nargs='*', default=[1e-3, 1e-3, 1e-3], type=float,
                        help='Learning rates. Default is [1e-3, 1e-3, 1e-3].')
    # parser.add_argument('--task_weights', nargs='*', default=[[0.1, 0.1, 0.8], 
    #                                                         [0.1, 0.1, 0.8], 
    #                                                         [0.1, 0.1, 0.8]], type=float,
    #                     help='Default is [0.1 0.1 0.8].')
    parser.add_argument('--task_weights', type=str, default="[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]",
        help='Default is [0.1 0.1 0.8].')
    
    # LR
    parser.add_argument('--lr_scheduler', nargs='*', default=['none', 'none', 'none'], type=str,
                        help="type of lr schedulers, choices: none, step, cosine")
    parser.add_argument('--local_wd', nargs='*', default=[0.0, 0.0, 0.0], type=float, help="local weight decay")

    # FedZoo
    parser.add_argument('--niid_beta', type=float, default=0.5, help='The parameter for non-iid data partitioning')
    parser.add_argument('--iid_beta', type=float, default=0.5, help='The parameter for iid data partitioning')

    # other arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--is_print', action='store_true', help='verbose print')
    parser.add_argument('--print_freq', type=int, default=100, help="printing frequency during training rounds")
    parser.add_argument('--seed', type=int, default=2023, help='random seed (default: 2023)')
    parser.add_argument('--load_initial', type=str, default='', help='define initial model path')
    parser.add_argument('--nlp', action='store_true', help='NLP task')

    args = parser.parse_args()
    
    try:
        args.task_weights = json.loads(args.task_weights)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse --task_weights as JSON.")

    return args
