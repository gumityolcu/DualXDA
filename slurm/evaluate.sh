for file in ../config_files/cluster/evaluate/$1/*$2*.yaml; do
    echo $file
    sbatch evaluate_job.sh $file $1 evaluate
done;
