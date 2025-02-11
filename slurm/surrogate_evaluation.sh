for file in ../config_files/cluster/surrogate_evaluation/$1/*$2*.yaml; do
    echo $file
    sbatch surrogate_evaluation_job.sh $file $1
done;
