for file in ../config_files/cluster/cache/$1/*$2*.yaml; do
    echo $file
    sbatch explain_job.sh $file cache $1
done;
