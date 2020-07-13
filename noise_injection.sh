#!/bin/sh
#PBS -l select=1:ngpus=1:ncpus=8:vnode=BDGPU10
#PBS -q gpuq_all
#PBS -N noise_injection
#PBS -o job_archive
#PBS -e job_archive

cd $PBS_O_WORKDIR
. ~/anaconda3/etc/profile.d/conda.sh
conda activate keras

epochs="50"

for var in 0.1 0.01 0.001 0.0001 0.00001
do
    python main.py  --train-mode noise-injection \
                    --out-dir "noise_injection/var_${var}" \
                    --nb-epochs ${epochs} \
                    --noise-var ${var} \
                    --gpu
done 