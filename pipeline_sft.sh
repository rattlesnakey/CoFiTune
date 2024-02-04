
#!/bin/bash
export GCC_ROOT=/path/to/gcc-7.5.0 or gcc-5.4.0
export CUDA_HOME=/path/to/cuda-11.7
export LD_LIBRARY_PATH=/path/to/gcc/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/cuda-11.7/lib64/:/path/to/cuda-11.7/:$LD_LIBRARY_PATH

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
partition="your partition name"

BAD_NODE="your bad node name"


train(){
    srun -p ${partition} --ntasks-per-node=${NUM_GPUS} --gres=gpu:${NUM_GPUS} -n${NUM_TASKS} -x ${BAD_NODE} \
    python -u src/training.py \
    --masterport ${MASTER_PORT} \
    --data_config_path ${DATA_PATH} \
    --model_name_or_path ${LOAD_PATH} \
    --args_output_path ${ARGS_OUTPUT_PATH} \
    --tokenizer_name_or_path $TOKEN_PATH \
    --per_device_train_batch_size ${BATCH} \
    --per_device_eval_batch_size ${BATCH} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --learning_rate ${LR} \
    --weight_decay 0.1 \
    --num_train_epochs ${EPOCHS}  \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --lr_scheduler_type cosine \
    --seed 1234 \
    --gradient_checkpointing True \
    --zero_stage ${ZERO_STAGE} \
    --deepspeed \
    --masterport ${MASTER_PORT} \
    --output_dir ${OUTPUT_PATH} \
    --offload ${OFFLOAD} \
    --only_optimize_some_blocks ${OPTIMIZE_LAYERS} \
    --only_optimize_some_params ${OPTIMIZE_PARAMS} \
    --train_with_softmask ${TRAIN_WITH_SOFTMASK} \
    --softmask_path ${SOFTMASK_PATH} \
    --apply_softmask ${APPLY_SOFTMASK} \
    2>&1 | tee ${ARGS_OUTPUT_PATH}/training.log

}


ddp_infer(){
    srun -p ${partition} --ntasks-per-node=${NUM_GPUS} --gres=gpu:${NUM_GPUS} -N 1 -x ${BAD_NODE} \
    python -u src/ddp_batch_inference.py \
    --masterport ${MASTER_PORT} \
    --ckpt_path ${CKPT_PATH} \
    --save_path ${SAVE_PATH} \
    --tk_path ${TK_PATH} \
    --test_path ${TEST_PATH} \
    --args_output_path ${ARGS_OUTPUT_PATH} \
    --batch_size ${TEST_BATCH_SIZE} \
    2>&1 | tee ${SAVE_DIR}/infering_${TEST_DATA_NAME}.log
}



#! train dataset config path
declare -A data_dict
data_dict["train-dataset-name"]="xxx/train_data_config.json"


#! model path
declare -A model_dict
model_dict["7b"]="absolute_7b_model_path" 
model_dict["13b"]="absolute_13b_model_path" 
model_dict["33b"]="absolute_33b_model_path" 

#! test data path
declare -A test_data_dict
test_data_dict["test-dataset-name"]="xxx/test_data.json"



#！ softmask path
declare -A softmask_dict
softmask_dict["7b-impt"]="${ROOT_DIR}/outputs/7b/impt_vectors/data-name"
softmask_dict["13b-impt"]="${ROOT_DIR}/outputs/13b/impt_vectors/data-name"
softmask_dict["33b-impt"]="${ROOT_DIR}/outputs/33b/impt_vectors/data-name"


################################## train ######################################
#! train blocks
declare -A train_layers_7B
### 7B
train_layers_7B['bottom1']='0,1,2,3,4,5,6,7'
train_layers_7B['bottom2']='8,9,10,11,12,13,14,15'
train_layers_7B['bottom-plus']='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
train_layers_7B['middle']='16,17,18,19,20,21,22,23'
train_layers_7B['middle-plus']='8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23'
train_layers_7B['top']='24,25,26,27,28,29,30,31'
train_layers_7B['top-plus']='16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31'
train_layers_7B['None']='None'

### 13B
declare -A train_layers_13B
train_layers_13B['bottom1']='0,1,2,3,4,5,6,7,8,9'
train_layers_13B['bottom2']='10,11,12,13,14,15,16,17,18,19'
train_layers_13B['bottom-plus']='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19'
train_layers_13B['middle']='20,21,22,23,24,25,26,27,28,29'
train_layers_13B['middle-plus']='10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29'
train_layers_13B['top']='30,31,32,33,34,35,36,37,38,39'
train_layers_13B['top-plus']='20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39'
train_layers_13B['None']='None'


### 33B
declare -A train_layers_33B
train_layers_33B['bottom2']='15,16,17,18,19,20,21,22,23,24,25,26,27,28,29'
train_layers_33B['None']='None'






CKPT_OUTPUT_DIR=/path/to/save/ckpt
ROOT_DIR=/path/to/code_dir

#! load pretrained model
MODEL_NAME=model_name # 
LOAD_PATH=${model_dict["${MODEL_NAME}"]}
TOKEN_PATH=${LOAD_PATH}


#! train data config
DATA_NAME=train_data_name #
DATA_PATH=${data_dict["${DATA_NAME}"]}

#! training param
EPOCHS=3 
GRAD_ACC=2
ZERO_STAGE=3 

OFFLOAD=True 
MAX_SEQ_LEN=512



#! efficient train param
TRAIN_LAYER=bottom2 #! bottom, middle, top, None ..
OPTIMIZE_LAYERS=${train_layers["${TRAIN_LAYER}"]} 
OPTIMIZE_PARAMS=up_proj,down_proj #q_proj,k_proj,v_proj,up_proj,down_proj


#! fine-grained softmask
TRAIN_WITH_SOFTMASK=True #! True or False

SOFTMASK_PATH=${ROOT_DIR}/outputs/xb/impt_vectors/${DATA_NAME}
APPLY_SOFTMASK=input_projection,output_projection



LR=4e-5


#! output param
IMPORT_PARAM=${MODEL_NAME}-${DATA_NAME}/optimize_param-${OPTIMIZE_PARAMS}_optimize_blocks-${OPTIMIZE_LAYERS}-epoch-${EPOCHS}
OUTPUT_PATH=${CKPT_OUTPUT_DIR}/output_models/${IMPORT_PARAM}

ARGS_OUTPUT_PATH=outputs/${IMPORT_PARAM}
mkdir -p ${ARGS_OUTPUT_PATH}

NUM_GPUS=8
NUM_TASKS=8 
BATCH=16
train

################################## train ######################################





################################## infer ######################################
export NCCL_DEBUG=INFO


CKPT_PATH=None
TK_PATH=${LOAD_PATH}


TEST_DATA_NAME=test_data_name
TEST_PATH=${test_data_dict["${TEST_DATA_NAME}"]}
NUM_GPUS=8
TEST_BATCH_SIZE=2 

if [ "${ARGS_OUTPUT_PATH}" = 'None' ]; then 
    SAVE_DIR=outputs/${MODEL_NAME}
    SAVE_PATH=${SAVE_DIR}/${TEST_DATA_NAME}-test.json
else
    SAVE_DIR=${ARGS_OUTPUT_PATH}
    SAVE_PATH=${ARGS_OUTPUT_PATH}/${TEST_DATA_NAME}-test.json
fi

mkdir -p ${SAVE_DIR}

ddp_infer
################################## infer ######################################
