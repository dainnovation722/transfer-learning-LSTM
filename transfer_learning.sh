#!/bin/bash
#PBS -l select=1:ngpus=1:ncpus=8:vnode=BDGPU09
#PBS -q gpuq_all
#PBS -N tl
#PBS -o job_archive
#PBS -e job_archive

cd $PBS_O_WORKDIR
. ~/anaconda3/etc/profile.d/conda.sh
conda activate keras

output="transfer_learning"
epochs="50"

python main.py -m pre-train -o ${output} --gpu --nb-epochs $epochs && \
python main.py -m transfer-learning -o ${output} --gpu --nb-epochs $epochs && \
python main.py -m without-transfer-learning -o ${output} --gpu --nb-epochs $epochs 