#!/bin/sh
#PBS -l select=1:ngpus=1:vnode=BDGPU11
#PBS -q gpuq_V100_32
#PBS -N without_tl
#PBS -o job_archive
#PBS -e job_archive

cd $PBS_O_WORKDIR
. ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu

output="test1"
python main.py -m pre-train -o "${output}" --gpu && \
python main.py -m transfer-learning -o "${output}" --gpu && \
python main.py -m without-transfer-learning -o "${output}" --gpu && \
python main.py -m score -o "${output}" --gpu  