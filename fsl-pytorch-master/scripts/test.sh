work_path=$(pwd)
GPU_number=${1}
config=${2}

job_name='fsl_test'
echo 'testing...'

CUDA_VISIBLE_DEVICES=${GPU_number} python -u tools/test.py --config=${work_path}/${config}
