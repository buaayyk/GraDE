#!/usr/bin/env bash

set -e

dataset=Younger
subgraph_sampling_method=RESU
sampling_density=40000
train_batch_size=256
score_batch_size=768
epochs=200
device=0
algo=DisCo
rounds=10
topK=1000
max_instance=20

for subgraph_size in 4 5 6 7 8 9 10
do
    output_dirpath=outputs/${dataset}_${subgraph_sampling_method}_${subgraph_size}_${sampling_density}_${algo}
    python main.py diffusion \
        --mode train \
        --dataset ${dataset} \
        -ssm ${subgraph_sampling_method} -ss ${subgraph_size} -sd ${sampling_density} \
        --output_dirpath ${output_dirpath} \
        --batch_size ${train_batch_size} --epochs ${epochs} --device ${device} \
        ${algo} --level_fea 1 --extra_time 1 --n_dim -1
done

python main.py beam_search \
    --dataset ${dataset} --k 10 --selection topk \
    -ssm ${subgraph_sampling_method} -sd ${sampling_density} \
    --topK ${topK} --max_instance ${max_instance} \
    --batch_size ${score_batch_size} --device ${device} \
    ${algo} --rounds ${rounds} --sampling_steps 100 --level_fea 1 --extra_time 1 --n_dim -1

for subgraph_size in 4 5 6 7 8 9 10
do
    python main.py evaluation discovery \
        --dataset ${dataset} \
        -ssm ${subgraph_sampling_method} -ss ${subgraph_size} -sd ${sampling_density} \
        --algo ${algo} --selection topk \
        --topK ${topK} --max_instance ${max_instance} --eval_topK 50 --resume
done
