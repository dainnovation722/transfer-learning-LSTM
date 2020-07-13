#!/bin/sh
#PBS -l select=1:ngpus=1:ncpus=12:vnode=BDGPU01
#PBS -q gpuq_all
#PBS -N bagging
#PBS -o job_archive
#PBS -e job_archive

cd $PBS_O_WORKDIR
. ~/anaconda3/etc/profile.d/conda.sh
conda activate keras

python bagging.py
