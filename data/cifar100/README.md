 # CIFAR100 Dataset

## Introduction
Split CIFAR100 dataset among `n_clients` using a hierarchical Latent Dirichlet
Allocation (LDA) process, referred to as the
[Pachinko Allocation Method](https://people.cs.umass.edu/~mccallum/papers/pam-icml06.pdf) (PAM).
This method uses a two-stage LDA process, where each client has an associated
multinomial distribution over the coarse labels of CIFAR-100, and a
coarse-to-fine label multinomial distribution for that coarse label over the
labels under that coarse label. The coarse label multinomial is drawn from a
symmetric Dirichlet with the parameter $\alpha$, and each coarse-to-fine multinomial
distribution is drawn from a symmetric Dirichlet with the parameter $\beta$. 
To generate a sample for the client, we first select
a coarse label by drawing from the coarse label multinomial distribution, and
then draw a fine label using the coarse-to-fine multinomial distribution. We
then randomly draw a sample from CIFAR-100 with that label (without
replacement). If this exhausts the set of samples with this label, we
remove the label from the coarse-to-fine multinomial and re-normalize the
multinomial distribution.

Inspired by the split in
[Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295)

## Instructions

### Base instructions

For basic usage, run generate_data.py with a choice of the following arguments:

- ```--n_tasks```: number of tasks/clients, written as integer
- ```--pachinko_allocation_split```:  if selected, the dataset will be split
  using Pachinko allocation,
- ```--alpha```: parameter controlling tasks dissimilarity, the smaller alpha
  is the more tasks are dissimilar; 
  default=``0.4``
-  ```--beta```: parameter controlling tasks dissimilarity, the smaller alpha
   is the more tasks are dissimilar; 
  default=``10`` 
- ```--s_frac```: fraction of the dataset to be used; default=``1.0``
- ```--tr_frac```: train set proportion for each task; default=``0.8``
- ```--val_frac```: fraction of validation set (from train set); default: ``0.0``
- ```--test_tasks_frac```: fraction of test tasks; default=``0.0``
- ```--seed``` := seed to be used before random sampling of data; default=``12345``


### Additional options

We als o provide some additional options to split the dataset

- ```--pathological_split```: if selected, the dataset will be split as in
  [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629);
  i.e., each client will receive `n_shards` of dataset, where each shard 
  contains at most two classes.
- ```--n_shards```: number of shards given to each client/task;
  ignored if `--pathological_split` is not used;
  default=`2`
- ```n_components```: if neither  ``--pachinko_allocation_split`` nor 
  `--pathological_split` are selected,
  the dataset is split as follows; 1) classes are grouped into `n_clusters`.2) for
  each cluster `c`,  samples are partitioned across clients using
  dirichlet distribution.
- ```--val_frac```: fraction of validation set (from train set); default=`0.0`
  
## Paper Experiments

### Full client participation (Table 2)

In order to generate the data split for Table 2 (Full client participation) without
validation set, run

```
python generate_data.py \
    --n_tasks 100 \
    --pachinko_allocation_split \
    --alpha 0.4 \
    --beta 10 \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --seed 12345    
```

In order to include the validation set, run

```
python generate_data.py \
    --n_tasks 100 \
    --pachinko_allocation_split \
    --alpha 0.4 \
    --beta 10 \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --val_frac 0.25 \
    --seed 12345    
```

### Unseen clients (Table 3)

In order to generate the data split for Table 3 (Unseen clients), run

```
python generate_data.py \
    --n_tasks 100 \
    --pachinko_allocation_split \
    --alpha 0.4 \
    --beta 10 \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --test_tasks_frac 0.2 \
    --seed 12345   
```

In order to include the validation set, run

```
python generate_data.py \
    --n_tasks 100 \
    --pachinko_allocation_split \
    --alpha 0.4 \
    --beta 10 \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --val_frac 0.25 \
    --test_tasks_frac 0.2 \
    --seed 12345   
```