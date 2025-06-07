#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=explain
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G

source "/etc/slurm/local_job_dir.sh"

start=`date +%s`

singularity \
run \
      --nv \
      --bind ${HOME}/DualView/config_files:/mnt/config_files \
      --bind ${HOME}/DualView/src:/mnt/src \
      --bind ${HOME}/DualView/checkpoints:/mnt/checkpoints \
      --bind ${DATAPOOL3}/datasets:/mnt/dataset \
      --bind ${HOME}/DualView/cache:/mnt/cache \
      ../singularity/featime.sif
