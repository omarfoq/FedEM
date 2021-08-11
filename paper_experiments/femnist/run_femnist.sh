cd ../../

# run FedAvg
echo "Run FedAvg"
python run_experiment.py femnist FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedAvg + local adaption
echo "Run FedAvg + local adaption"
python run_experiment.py femnist FedAvg --n_learners 1 --locally_tune_clients --n_rounds 200 --bz 128 \
 --lr 0.03 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run training using local data only
echo "Run Local"
python run_experiment.py femnist local --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run Clustered FL
echo "Run Clustered FL"
python run_experiment.py femnist clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedProx
echo "Run FedProx"
python run_experiment.py femnist FedProx --n_learners 1 --n_rounds 200 --bz 128 --lr 0.05 --mu 0.05 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# Run Richtarek's Formulation
echo "Run Personalized (Richtarek's Formulation)"
python run_experiment.py femnist personalized --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 --mu 10.0 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# run FedEM
echo "Run FedEM"
python run_experiment.py femnist FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1