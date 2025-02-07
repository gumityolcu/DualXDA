#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=evaluate
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

source "/etc/slurm/local_job_dir.sh"

start=`date +%s`

mkdir -p ${LOCAL_JOB_DIR}/outputs

fname_config=$(basename "$1")
config_name=${fname_config::-5}

if [[ "$fname_config" == *"trak"* ]]; then
  singularity \
  run \
        --nv \
        --bind ${HOME}/DualView/config_files:/mnt/config_files \
        --bind ${HOME}/DualView/src:/mnt/src \
        --bind ${HOME}/DualView/explanations:/mnt/explanations \
        --bind ${HOME}/DualView/checkpoints:/mnt/checkpoints \
        --bind ${DATAPOOL3}/datasets:/mnt/dataset \
        --bind ${LOCAL_JOB_DIR}/outputs:/mnt/outputs \
        --bind ${HOME}/DualView/cache:/mnt/cache \
        ../singularity/evaluate.sif --trak --config_file /mnt/config_files/cluster/evaluate/$2/${fname_config}
else
  singularity \
    run \
          --nv \
          --bind ${HOME}/DualView/config_files:/mnt/config_files \
          --bind ${HOME}/DualView/src:/mnt/src \
          --bind ${HOME}/DualView/explanations:/mnt/explanations \
          --bind ${HOME}/DualView/checkpoints:/mnt/checkpoints \
          --bind ${DATAPOOL3}/datasets:/mnt/dataset \
          --bind ${LOCAL_JOB_DIR}/outputs:/mnt/outputs \
          --bind ${HOME}/DualView/cache:/mnt/cache \
          ../singularity/evaluate.sif --config_file /mnt/config_files/cluster/evaluate/$2/${fname_config}
fi
cd ${LOCAL_JOB_DIR}
tar -czf $2_evaluate_${fname_config}-output_data_${SLURM_JOB_ID}.tgz outputs
cp $2_evaluate_${fname_config}-output_data_${SLURM_JOB_ID}.tgz ${SLURM_SUBMIT_DIR}

end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime"
echo "In minutes: $(($runtime / 60))"
