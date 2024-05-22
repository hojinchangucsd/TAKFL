#!/bin/bash

alpha="0.5"

self_reg="0 0 1.0"
task_weights="[[0.3,0.3,0.4],[0.3,0.3,0.4],[0.3,0.3,0.4]]"
tune="heuristic" # options [no_tuning, backprop, heuristic]
n_candidates="10"

private_ds="mnli"
public_ds="snli"

alg="fedavg_mh"
gpu="1"

logdir="./results_nlp_pri-"$private_ds"_pub-"$public_ds"_"
log_filename="$alg"

python ./main.py \
--nlp \
--tune_lambda "$tune" \
--n_candidates "$n_candidates" \
--gpu "$gpu" \
--num_threads -1 \
--train_size 100000 \
--public_size 30000 \
--task_weights "$task_weights" \
--self_reg ${self_reg} \
--ntrials 2 \
--rounds 20 \
--nclusters 3 \
--num_users 8 4 2 \
--fracs 0.4 0.5 1.0 \
--data_ratios 0.1 0.3 0.6 \
--models bert-tiny bert-mini bert-small \
--local_ep 1 1 1 \
--local_bs 32 \
--optim adam \
--lr 3e-5 3e-5 3e-5 \
--lr_scheduler none none none \
--local_wd 0 0 0 \
--dataset "$private_ds" \
--distill_dataset "$public_ds" \
--distill_lr 3e-5 \
--distill_wd 0 0 0 \
--distill_E 1 \
--distill_T 3 \
--partition niid-labeldir \
--datadir ./data \
--logdir "$logdir" \
--log_filename "$log_filename" \
--alg "$alg" \
--niid_beta "$alpha" \
--seed 2023
