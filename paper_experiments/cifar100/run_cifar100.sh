cd ../../

# run FedAvg
echo "Run FedAvg"
python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedAvg + local adaption
echo "run FedAvg + local adaption"
python run_experiment.py cifar100 FedAvg --n_learners 1 --log_before_aggregate --n_rounds 200 --bz 128 \
 --lr 0.001 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run training using local data only
echo "Run Local"
python run_experiment.py cifar100 local --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run Clustered FL
echo "Run Clustered FL"
python run_experiment.py cifar100 clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedProx
echo "Run FedProx"
python run_experiment.py cifar100 FedProx --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 0.1 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# Run Richtarek's Formulation
echo "Run Personalized (Richtarek's Formulation)"
python run_experiment.py cifar100 personalized --n_learners 1 --n_rounds 200 --bz 128 --lr 0.001 --mu 1.0 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# run FedEM
echo "Run FedEM"
python run_experiment.py cifar100 FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1