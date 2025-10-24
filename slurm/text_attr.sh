#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=text_attr
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G

source "/etc/slurm/local_job_dir.sh"

start=`date +%s`

mkdir -p ${LOCAL_JOB_DIR}/outputs

singularity \
run \
      --nv \
      --bind ${HOME}/DualView/src:/mnt/src \
      --bind ${HOME}/DualView/explanations:/mnt/explanations \
      --bind ${DATAPOOL3}/datasets:/mnt/dataset \
      --bind ${LOCAL_JOB_DIR}/outputs:/mnt/outputs \
      --bind ${HOME}/DualView/cache:/mnt/cache \
      ../singularity/text_attributions.sif --start 0 --length 2 --save_dir /mnt/outputs --dataset_name ag_news --xai_method $1
cd ${LOCAL_JOB_DIR}
tar -czf $1-output_data_${SLURM_JOB_ID}.tgz outputs
cp $1-output_data_${SLURM_JOB_ID}.tgz ${SLURM_SUBMIT_DIR}

end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime"
echo "In minutes: $(($runtime / 60))"
