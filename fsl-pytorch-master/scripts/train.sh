work_path=$(pwd)
GPU_number=${1}
config=${2}

job_name='fsl_train'

CUDA_VISIBLE_DEVICES=${GPU_number} python -u tools/train.py --config=${work_path}/${config}
