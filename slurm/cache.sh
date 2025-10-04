for file in ../config_files/cluster/cache/$1/*$2*.yaml; do
    echo $file
    sbatch -p testing explain_job.sh $file cache $1
done;
