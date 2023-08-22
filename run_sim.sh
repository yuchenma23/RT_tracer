#!/bin/bash
#SBATCH -N 1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --mem=500GB

source setup_satori.sh

cat > launch.sh << EoF_s
#! /bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec \$*
EoF_s
chmod +x launch.sh

srun --mpi=pmi2 ./launch.sh $JULIA --check-bounds=no --project simulation.jl
