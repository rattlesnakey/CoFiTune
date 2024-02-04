
#!/bin/bash
export GCC_ROOT=/path/to/gcc-7.5.0 or gcc-5.4.0
export CUDA_HOME=/path/to/cuda-11.7
export LD_LIBRARY_PATH=/path/to/gcc/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/cuda-11.7/lib64/:/path/to/cuda-11.7/:$LD_LIBRARY_PATH

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
partition="your partition name"

BAD_NODE="your bad node name"


compute_impt_vector(){
    srun -p ${partition} --ntasks-per-node=${NUM_GPUS} --gres=gpu:${NUM_GPUS} -n${NUM_TASKS} -x ${BAD_NODE} \
    python -u src/training.py \
    --masterport ${MASTER_PORT} \
    --data_config_path ${DATA_PATH} \
    --model_name_or_path ${LOAD_PATH} \
    --args_output_path ${ARGS_OUTPUT_PATH} \
    --tokenizer_name_or_path ${TOKEN_PATH} \
    --per_device_train_batch_size ${BATCH} \
    --per_device_eval_batch_size ${BATCH} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --weight_decay 0.1 \
    --gradient_accumulation_steps 1 \
    --seed 1234 \
    --gradient_checkpointing True \
    --zero_stage ${ZERO_STAGE} \
    --deepspeed \
    --output_dir ${OUTPUT_PATH} \
    --offload ${OFFLOAD} \
    --compute_impt_vector ${COMPUTE_IMPT_VECTOR} \
    2>&1 | tee ${ARGS_OUTPUT_PATH}/computing_impt.log
}


main(){

    DATA_PATH=${data_dict["${DATA_NAME}"]}

    ZERO_STAGE=3 

    OFFLOAD=True 
    MAX_SEQ_LEN=512 

    COMPUTE_IMPT_VECTOR=True
    OUTPUT_PATH=outputs/${MODEL_NAME}/impt_vectors/${DATA_NAME}

    ARGS_OUTPUT_PATH=outputs/${MODEL_NAME}/impt_vectors/${DATA_NAME}
    mkdir -p ${ARGS_OUTPUT_PATH}

    NUM_GPUS=8
    NUM_TASKS=8
    BATCH=${BATCH_SIZE}
    compute_impt_vector
}


#! model path
declare -A model_dict
model_dict["7b"]="path/to/7b-model/"
model_dict["13b"]="path/to/13b-model/"
model_dict["33b"]="path/to/33b-model/"

#! load pretrained model
MODEL_NAME="model_name" 
LOAD_PATH=${model_dict["${MODEL_NAME}"]}
TOKEN_PATH=${LOAD_PATH}

#! train dataset config path
declare -A data_dict
data_dict["train-dataset-name"]="xxx/train_data_config.json"

TRAIN_DATA_NAME=xxx
DATA_PATH=${data_dict[${TRAIN_DATA_NAME}]}
BATCH_SIZE=16

main
