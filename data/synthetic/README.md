# Synthetic Mixture Dataset

## Introduction
We propose a process to generate synthetic federated datasets,
with the particularity that the data distribution of each client/task is the mixture
of \(M\) underlying distributions.


## Instructions

Run generate_data.py with a choice of the following arguments:

- ```--n_tasks```: number of tasks/clients, written as integer;
- ```--n_classes``` : number of classes, written as integer; default=``2``
- ```--n_components```: number of mixture components, written as integer; default=``3``
- ```--dimension```: dimension of the data, written as integer; default=``150``
- ```--noise_level``` : proportion of noise, default=``0.1`` 
- ```--n_test```: size of test set; default=``5_000``
- ```--alpha```: parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar; 
  default=``0.4``
- ```--uniform_marginal```: flag indicating if the all tasks should have the same marginal; default=``True``
- ```--train_tasks_frac```: fraction of test tasks; default=``1.0``  
- ```--seed``` : seed to be used before random sampling of data; default=``12345``


## Paper Experiments

In order to generate the data split for Table 2 (Full client participation), run

```
python generate_data.py \
    --n_tasks 300 \
    --n_classes 2 \
    --n_components 3 \
    --dimension 150 \
    --noise_level 0.1
    --n_test 5000 \
    --alpha 0.4 \
    --seed 12345    
```

In order to generate the data split for Table 3 (Unseen clients), run

```
python generate_data.py \
    --n_tasks 300 \
    --n_classes 2 \
    --n_components 3 \
    --dimension 150 \
    --noise_level 0.1
    --n_test 5000 
    --alpha 0.4 \
    --tr_tasks_frac 0.8 \
    --seed 12345    
```
