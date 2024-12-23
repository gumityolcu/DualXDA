#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=explain
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

source "/etc/slurm/local_job_dir.sh"

start=`date +%s`

mkdir -p ${LOCAL_JOB_DIR}/outputs

tar -czf ${LOCAL_JOB_DIR}/config_files.tgz ${HOME}/DualView/config_files
tar -czf ${LOCAL_JOB_DIR}/checkpoints.tgz ${HOME}/DualView/checkpoints
tar -czf ${LOCAL_JOB_DIR}/cache.tgz ${HOME}/DualView/cache
echo "TAR DONE"
tar -C ${LOCAL_JOB_DIR} -zxf ${LOCAL_JOB_DIR}/config_files.tgz home/fe/yolcu/DualView/config_files --strip-components=4
tar -C ${LOCAL_JOB_DIR} -zxf ${LOCAL_JOB_DIR}/checkpoints.tgz home/fe/yolcu/DualView/checkpoints --strip-components=4
tar -C ${LOCAL_JOB_DIR} -zxf ${LOCAL_JOB_DIR}/cache.tgz home/fe/yolcu/DualView/cache --strip-components=4

fname_config=$(basename "$1")
config_name=${fname_config::-5}

if [[ "$fname_config" == *"trak"* ]]; then
  singularity \
  run \
        --nv \
        --bind ${LOCAL_JOB_DIR}/config_files:/mnt/config_files \
        --bind ${LOCAL_JOB_DIR}/checkpoints:/mnt/checkpoints \
        --bind ${DATAPOOL3}/datasets:/mnt/dataset \
        --bind ${LOCAL_JOB_DIR}/outputs:/mnt/outputs \
        --bind ${LOCAL_JOB_DIR}/cache:/mnt/cache \
        ../singularity/explain.sif --trak --config_file /mnt/config_files/cluster/$2/$3/${fname_config}
else
  singularity \
    run \
          --nv \
          --bind ${LOCAL_JOB_DIR}/config_files:/mnt/config_files \
          --bind ${LOCAL_JOB_DIR}/checkpoints:/mnt/checkpoints \
          --bind ${DATAPOOL3}/datasets:/mnt/dataset \
          --bind ${LOCAL_JOB_DIR}/outputs:/mnt/outputs \
          --bind ${LOCAL_JOB_DIR}/cache:/mnt/cache \
          ../singularity/explain.sif --config_file /mnt/config_files/cluster/$2/$3/${fname_config}
fi
cd ${LOCAL_JOB_DIR}
tar -czf $3_$2_${fname_config}-output_data_${SLURM_JOB_ID}.tgz outputs
cp $3_$2_${fname_config}-output_data_${SLURM_JOB_ID}.tgz ${SLURM_SUBMIT_DIR}

rm -rf ${LOCAL_JOB_DIR}/*

end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime"
echo "In minutes: $(($runtime / 60))"