#!/usr/bin/env bash

set -e

dataset=NAS-Bench-101
subgraph_size=5
subgraph_sampling_method=RESU
sampling_density=0.1
train_batch_size=128
estimate_batch_size=256
epochs=50
device=0
algo=GNNRanker
rounds=1
output_dirpath=outputs/${dataset}_${subgraph_sampling_method}_${subgraph_size}_${sampling_density}_${algo}

python main.py diffusion \
    --mode train \
    --dataset ${dataset} \
    -ssm ${subgraph_sampling_method} -ss ${subgraph_size} -sd ${sampling_density} \
    --output_dirpath ${output_dirpath} \
    --batch_size ${train_batch_size} --epochs ${epochs} --device ${device} \
    ${algo} --hidden_dim 128 --n_layers 4 --lr 0.001 --wd 0.00001 --rank_loss_weight 1.0 --enable_scheduler

python main.py diffusion \
    --mode estimate \
    --dataset ${dataset} \
    -ssm ${subgraph_sampling_method} -ss ${subgraph_size} -sd ${sampling_density} \
    --output_dirpath ${output_dirpath} \
    --model_load_path ${output_dirpath}/saved_models/best_validation.pt \
    --batch_size ${estimate_batch_size} --device ${device} \
    ${algo} --rounds ${rounds} --hidden_dim 128 --n_layers 4 --lr 0.001 --wd 0.00001 --rank_loss_weight 1.0 --enable_scheduler

python main.py evaluation model \
    --dataset ${dataset} \
    -ssm ${subgraph_sampling_method} -ss ${subgraph_size} -sd ${sampling_density} \
    --estimate_logps_path ${output_dirpath}/estimate_logps.json \
    --metrics spearman kendall --results_dirpath ${output_dirpath}
