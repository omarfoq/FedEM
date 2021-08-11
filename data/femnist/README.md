# FEMNIST Dataset

## Introduction
This dataset is derived from the Leaf repository
([LEAF](https://github.com/TalwalkarLab/leaf)) pre-processing of the
Extended MNIST dataset, grouping examples by writer.

Details about LEAF were published in
"[LEAF: A Benchmark for Federated Settings"](https://arxiv.org/abs/1812.01097)

## Setup Instructions

First, run `./preprocess.sh`, then run `generate_data.py` with a choice of the following arguments:

- ```--s_frac```: fraction of the dataset to be used; default=``0.3``  
- ```--tr_frac```: train set proportion for each task; default=``0.8``
- ```--val_frac```: fraction of validation set (from train set); default=`0.0`
- ```--train_tasks_frac```: fraction of test tasks; default=``1.0``
- ```--seed``` : seed to be used before random sampling of data; default=``12345``

## Paper Experiments

In order to generate the data split for Table 2 (Full client participation), run

```
python generate_data.py \
    --s_frac 0.2 \
    --tr_frac 0.8 \
    --seed 12345    
```

In order to generate the data split for Table 3 (Unseen clients), run

```
python generate_data.py \
    --s_frac 0.2 \
    --tr_frac 0.8 \
    --train_tasks_frac 0.8 \
    --seed 12345
```
