#!/usr/bin/env bash

set -e

dataset=NAS-Bench-101
subgraph_size=5
sampling_density=0.1

for subgraph_sampling_method in ARS NRS RESU RFASE
do
    output_dirpath=outputs/${dataset}_${subgraph_sampling_method}_${subgraph_size}_${sampling_density}_Sampling
    python main.py evaluation model \
        --sampling_only 1 \
        --dataset ${dataset} \
        -ssm ${subgraph_sampling_method} -ss ${subgraph_size} -sd ${sampling_density} \
        --metrics spearman kendall --results_dirpath ${output_dirpath}
done
