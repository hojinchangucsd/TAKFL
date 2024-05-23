# TAKFL

<a name="readme-top"></a>


Code for [https://openreview.net/forum?id=y6JotynERr](https://openreview.net/forum?id=y6JotynERr)



### Prerequisites

requirements.txt lists the packages used. 

## Usage

For computer vision experiments, enter the cv folder and run main.py. Similarly for NLP experiments, enter the nlp folder and run main.py. 
An example script called run.sh is provided in each folder. 

NLP Arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --nlp          	       |	False           | Have this option on when running NLP experiments. 
| --tune_lambda          | 'no_tuning'          | Tuning option for task weights. Choices: 'no_tuning', 'heuristic'
| --n_candidates	       |	10            | Number of candidates to use for the heuristic method
| --gpu  		       | 0           |GPU device ID
| --train_size	        | 100000   | Training set size
| --public_size    | 30000 | Distillation set size
| --task_weights  | '[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]'  | Task weights for TAKFL. Is a string representing a 2D square matrix. Each row is a vector of merging coefficients that is to be multiplied with the task vector in the task arithmetic. 
| --self_reg  | \[0.5, 0.5, 0.5\] | List of floats for the coefficient to be multiplied with the self-regularization term. 
| --n_trials    | 1 | Number of trials to run. 
| --rounds  |500| Number of communication rounds per trial.
| --nclusters    | 3 | Number of device prototypes. 
| --num_users| \[100, 20, 4\] | List of number of users per device prototype. 
| --fracs | \[0.4, 0.4, 0.4\]  | List of floats for the sampling rate every round of each prototype. 
| --data_ratios | \[0.8, 0.1, 0.1\] | Dataset ratios 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

