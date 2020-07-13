#!/bin/sh
#PBS -l select=1:ngpus=1:ncpus=8:vnode=BDGPU11
#PBS -q gpuq_all
#PBS -N bagging
#PBS -o job_archive
#PBS -e job_archive

cd $PBS_O_WORKDIR
. ~/anaconda3/etc/profile.d/conda.sh
conda activate keras

output="bagging_100"
epochs="50"
nb_subset="100"
python main.py  --train-mode bagging \
                --out-dir ${output} \
                --nb-epochs ${epochs} \
                --nb-subset ${nb_subset} \
                --gpu