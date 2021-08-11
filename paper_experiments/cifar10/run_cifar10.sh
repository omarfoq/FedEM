cd ../../

# run FedAvg
echo "Run FedAvg"
python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedAvg + local adaption
echo "run FedAvg + local adaption"
python run_experiment.py cifar10 FedAvg --n_learners 1 --locally_tune_clients --n_rounds 201 --bz 128 \
 --lr 0.001 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run training using local data only
echo "Run Local"
python run_experiment.py cifar10 local --n_learners 1 --n_rounds 201 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run Clustered FL
echo "Run Clustered FL"
python run_experiment.py cifar10 clustered --n_learners 1 --n_rounds 201 --bz 128 --lr 0.003 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedProx
echo "Run FedProx"
python run_experiment.py cifar10 FedProx --n_learners 1 --n_rounds 201 --bz 128 --lr 0.01 --mu 1.0\
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# Run pFedME
echo "Run pFedME"
python run_experiment.py cifar10 personalized --n_learners 1 --n_rounds 201 --bz 128 --lr 0.001 --mu 1.0 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# run FedEM
echo "Run FedEM"
python run_experiment.py cifar10 FedEM --n_learners 3 --n_rounds 201 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1