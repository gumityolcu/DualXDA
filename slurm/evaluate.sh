for file in ../config_files/cluster/explain/$1/*$2*.yaml; do
    echo $file
    sbatch explain_job.sh $file $1
done;
