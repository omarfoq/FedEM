cd ../../


echo "Run Personalized (Richtarek's Formulation), mu=0.01"
python run_experiment.py femnist personalized --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 0.01 \
 --lr_scheduler constant --log_freq 20 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 \
 --logs_root logs/femnist/personalized/mu_0.01

echo "Run Personalized (Richtarek's Formulation), mu=0.1"
python run_experiment.py femnist personalized --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 0.1 \
 --lr_scheduler constant --log_freq 20 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 \
 --logs_root logs/femnist/personalized/mu_0.1

echo "Run Personalized (Richtarek's Formulation), mu=1"
python run_experiment.py femnist personalized --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 1.0 \
 --lr_scheduler constant --log_freq 20 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 \
 --logs_root logs/femnist/personalized/mu_1

echo "Run Personalized (Richtarek's Formulation), mu=10"
python run_experiment.py femnist personalized --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 10.0 \
 --lr_scheduler constant --log_freq 20 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 \
 --logs_root logs/femnist/personalized/mu_10

echo "Run Personalized (Richtarek's Formulation), mu=100"
python run_experiment.py femnist personalized --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 --mu 100.0 \
 --lr_scheduler constant --log_freq 20 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 \
 --logs_root logs/femnist/personalized/mu_100
