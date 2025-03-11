#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=dualview_analysrs
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G

source "/etc/slurm/local_job_dir.sh"

start=`date +%s`

mkdir -p ${LOCAL_JOB_DIR}/outputs

singularity run \
  --nv \
  --bind ${HOME}/DualView/config_files:/mnt/config_files \
  --bind ${HOME}/DualView/src:/mnt/src \
  --bind ${HOME}/DualView/results:/mnt/results \
  --bind ${HOME}/DualView/checkpoints:/mnt/checkpoints \
  --bind ${DATAPOOL3}/datasets:/mnt/dataset \
  --bind ${LOCAL_JOB_DIR}/outputs:/mnt/outputs \
  --bind ${HOME}/DualView/cache/MNIST:/mnt/cache/MNIST \
  --bind ${HOME}/DualView/cache/CIFAR:/mnt/cache/CIFAR \
  --bind ${HOME}/DualView/cache/AWA:/mnt/cache/AWA \
  ../singularity/dualview_analysis.sif --device cuda --dataset $1

cd ${LOCAL_JOB_DIR}
tar -czf $1_surrogate_analyses_${SLURM_JOB_ID}.tgz outputs
cp $1_surrogate_analyses_${SLURM_JOB_ID}.tgz ${SLURM_SUBMIT_DIR}/surr_analyses

end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime"
echo "In minutes: $(($runtime / 60))"
