work_path=$(pwd)
partition=${1}
config=${2}

job_name='fsl_save_test'

echo 'saving feature...'
GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p ${partition} --gres=gpu:1 \
    --job-name=${job_name} \
    python -u tools/save.py --config=${work_path}/${config}

echo 'testing...'
GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p ${partition} --gres=gpu:1 \
    --job-name=${job_name} \
    python -u tools/test.py --config=${work_path}/${config}
