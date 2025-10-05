for file in ../config_files/cluster/explain/$1/*$2*yaml; do
    echo $file
    sbatch -p gpu4 explain_job.sh $file explain $1
done;

