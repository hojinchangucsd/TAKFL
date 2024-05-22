if __name__ == '__main__':
    from src.utils.options_cluster import args_parser
    args = args_parser()
    import os
    import torch

    if args.num_threads > 0: 
        os.environ["OMP_NUM_THREADS"] = str(args.num_threads) 
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.num_threads) 
        os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
        torch.set_num_threads(args.num_threads)

    import sys
    import datetime
    import numpy as np
    from src.utils import Logger

    from src.data import *
    from src.models import *
    from src.client import *
    from src.utils import *
    from src.benchmarks.main_fedavg_mh import *
    from src.benchmarks.main_fedmh import *
    from src.benchmarks.main_fedet import *
    from src.benchmarks.main_fedhd import *


    def set_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    print('-'*40)

    args = args_parser()
    if args.gpu == -1:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(args.gpu) ## Setting cuda on GPU

    args.path = args.logdir + args.alg +'/' + args.dataset + '/' + args.partition + '/'
    if args.partition != 'iid':
        if args.partition == 'iid_qskew':
            args.path = args.path + str(args.iid_beta) + '/'
        else:
            if args.niid_beta.is_integer():
                args.path = args.path + str(int(args.niid_beta)) + '/'
            else:
                args.path = args.path + str(args.niid_beta) + '/'

    mkdirs(args.path)

    args.datadir = os.path.abspath(args.datadir)

    if args.log_filename is None:
        filename='logs_%s.txt' % datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    else:
        filename='logs_'+args.log_filename+'.txt'

    sys.stdout = Logger(fname=args.path+filename)

    fname=args.path+filename
    fname=fname[0:-4]
    
    if args.alg == 'fedavg_mh':
        alg_name = 'FedAvg-MH'
        run_fedavg_mh(args, fname=fname)
    elif args.alg == 'fedet':
        alg_name = 'FedET'
        run_fedet(args, fname=fname)
    elif args.alg in ['fedmh', 'feddf']:
        alg_name = 'FedMH'
        run_fedmh(args, fname=fname)
    elif args.alg == 'fedhd':
        alg_name = 'FedHD'
        run_fedhd(args, fname=fname)
    else:
        print('Algorithm Does Not Exist')
        sys.exit()
