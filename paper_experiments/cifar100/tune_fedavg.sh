cd ../../


echo "Run FedAvg lr=0.003"
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.003 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --validation \
 --verbose 1 --logs_root logs/cifar100_validation/FedAvg_lr_0.003

echo "Run FedAvg lr=0.01"
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --validation \
 --verbose 1 --logs_root logs/cifar100_validation/FedAvg_lr_0.01

echo "Run FedAvg lr=0.03"
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --validation \
 --verbose 1 --logs_root logs/cifar100_validation/FedAvg_lr_0.03

echo "Run FedAvg lr=0.1"
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --validation \
 --verbose 1 --logs_root logs/cifar100_validation/FedAvg_lr_0.1
