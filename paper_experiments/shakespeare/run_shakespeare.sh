cd ../../

# run FedAvg
echo "Run FedAvg"
python run_experiment.py shakespeare FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedAvg + local adaption
echo "Run FedAvg + local adaption"
python run_experiment.py shakespeare FedAvg --n_learners 1 --locally_tune_clients  --n_rounds 200 --bz 128 \
 --lr 0.01 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run training using local data only
echo "Run Local"
python run_experiment.py shakespeare local --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run Clustered FL
echo "Run Clustered FL"
python run_experiment.py shakespeare clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedProx
echo "Run FedProx"
python run_experiment.py shakespeare FedProx --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 0.1 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# Run Richtarek's Formulation
echo "Run Personalized (Richtarek's Formulation)"
python run_experiment.py shakespeare personalized --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 1.0 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# run FedEM
echo "Run FedEM"
python run_experiment.py shakespeare FedEM --n_learners 3 --n_rounds 200 --bz 128 --lr 0.05 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1