for file in ../config_files/cluster/lds/$1/*$2*.yaml; do
    echo $file
    sbatch evaluate_job.sh $file lds $1
done;
