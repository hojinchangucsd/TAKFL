#!/bin/bash

NCLUSTERS="3"
NUM_USERS="10 5 2"
DATA_RATIOS="0.1 0.3 0.6"
MODELS="vit-s resnet14 vgg16"
FRACS="0.1 0.2 0.5"
LOCAL_EPOCHS="20 20 20"

DATASET="cifar10"
DISTILL_DATA="cifar100"
DISTILL_WD="5e-5 5e-5 5e-5"
DISTILL_E="5"

TASK_WEIGHTS="[[0.6, 0.3, 0.1], [0.3, 0.5, 0.2], [0.1, 0.1, 0.8]]"
SELF_REG="0.01 0.01 0.01"
NIID="0.1"
GPU="0"

LR="1e-3 1e-3 1e-3"
SCHEDULER="none none step"
WD="5e-5 5e-5 5e-5"

LAMBDA="heuristic"

FILENAME="Distill_E${DISTILL_E}_${MODELS// /_}_${DISTILL_DATA}_T_heuristic_Reg${SELF_REG// /_}"

python ./main.py \
--ntrials=2 \
--rounds=60 \
--nclusters=$NCLUSTERS \
--num_users ${NUM_USERS} \
--fracs ${FRACS} \
--data_ratios ${DATA_RATIOS} \
--models ${MODELS} \
--local_ep ${LOCAL_EPOCHS} \
--local_bs=64 \
--optim='adam' \
--lr ${LR} \
--lr_scheduler ${SCHEDULER} \
--local_wd ${WD} \
--dataset=$DATASET \
--distill_lr=0.00001 \
--distill_wd ${DISTILL_WD} \
--distill_E=$DISTILL_E \
--distill_T=3 \
--distill_data=$DISTILL_DATA \
--self_reg ${SELF_REG} \
--task_weights="$TASK_WEIGHTS" \
--p_train=1.0 \
--partition='niid-labeldir' \
--datadir='./data/' \
--logdir='./results_nlp_pri-'"$DATASET_pub"'-'"$DISTILL_DATA"'_'\
--log_filename="$FILENAME" \
--alg='fedhd' \
--iid_beta=0.5 \
--niid_beta=$NIID \
--seed=2023 \
--gpu=$GPU \
--print_freq=10 \
--lambda_tuning $LAMBDA \
--n_candidates=15