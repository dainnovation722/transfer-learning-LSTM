#!/bin/sh
#PBS -l select=1:ngpus=1
#PBS -q gpuq_cuda10
#PBS -N relu-TL

cd $PBS_O_WORKDIR
. ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu

condition="test"
n_iteration=3


for n_iter in `seq 1 $n_iteration`
do
    python main.py -m pre-train -n "${condition}_${n_iter}"
    python main.py -m transfer-learning -n "${condition}_${n_iter}"
    python main.py -m without-transfer-learning -n "${condition}_${n_iter}"   
done
