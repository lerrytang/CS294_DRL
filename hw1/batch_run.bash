#!/bin/bash
set -eux
for e in Hopper-v1 Ant-v1 HalfCheetah-v1 Humanoid-v1 Reacher-v1 Walker2d-v1
do
    for n in 5 10 15 20 25 30
    do
        python run_expert.py experts/$e.pkl $e --num_rollouts=$n --clone_expert 2>&1 | tee log/log_${e}_${n}.txt
    done
done
