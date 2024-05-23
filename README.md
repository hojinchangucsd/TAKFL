# TAKFL

<a name="readme-top"></a>

### Prerequisites

requirements.txt lists the packages used. 

## Usage

For computer vision experiments, enter the cv folder and run main.py. Similarly for NLP experiments, enter the nlp folder and run main.py. 
An example script called run.sh is provided in each folder. 

CV Arguments: 

| Parameter                 | Description   |	
| :------------------------ |:-------------|
| --tune_lambda      | Tuning option for task weights. Choices: 'no_tuning', 'heuristic'
| --n_candidates	      | Number of candidates to use for the heuristic method
| --gpu  		      |GPU device ID
| --train_size	      | Training set size
| --public_size   | Distillation set size
| --task_weights | Task weights for TAKFL. Is a string representing a 2D square matrix. Each row is a vector of merging coefficients that is to be multiplied with the task vector in the task arithmetic. 
| --self_reg | List of floats for the coefficient to be multiplied with the self-regularization term. 
| --n_trials | Number of trials to run. 
| --rounds | Number of communication rounds per trial.
| --nclusters  | Number of device prototypes. 
| --num_users | List of number of users per device prototype. 
| --fracs  | List of floats for the sampling rate every round of each prototype. 
| --data_ratios  | Dataset ratios per prototype. 
| --models | Prototype model names. 
| --local_ep | List of integers for the client local training epochs per prototype. 
| --local_bs | Integer for the client local training batch size. 
| --optim | Optimizer for local training. 
| --lr | Learning rate for local training. 
| --lr_scheduler | Learning rate scheduler for local training. 
| --local_wd | List of floats for the local training optimizer's weight decay per prototype. 
| --dataset | Private dataset for local training to be distributed across clients. 
| --distill_dataset | Public dataset for distillation. 
| --distill_E | Distillation epoch count. 
| --distill_T | Distillation softmax temperature. 
| --partition | Specifies I.I.D or N.I.I.D partitioning. 
| --datadir | Dataset directory. Huggingface datasets should already be present in the directory. The code does not automatically download Huggingface datasets. 
| --logdir | Logging file directory. 
| --log_filename | Log file name. 
| --alg | Algorithm to run. TAKFL is 'FedHD' and FedDF is 'FedMH'. 
| --niid_beta | N.I.I.D dirichlet distribution heterogeneity parameter. 
| --seed | Random seed. 


NLP Arguments: 
Same as CV arguments but added --nlp option. Should be on for NLP experiments. 
| Parameter                 | Description   |	
| :------------------------ |:-------------|
| --nlp          	      | Have this option on when running NLP experiments. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

