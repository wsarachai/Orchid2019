#!/bin/bash

TRAIN_DIR=$PWD
trained_dir="${WORKSPACE}/_trained_models/checkpoints/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt"
script="${TRAIN_DIR}/train.py"
learning_rate_decay="fixed"

#Hyperparameters
readonly connections=(
    #train_step|batch_size|total_epochs|learning_rate|dropout|fine_tune|fine_tune_at|save_best_only
    '2|32|10|0.001|0.6|False|0|True'
    '3|32|10|0.001|0.6|False|0|True'
    '4|8|100|0.001|0.6|True|108|True'
    '4|8|100|0.001|0.6|True|99|True'
    '4|8|100|0.001|0.6|True|90|True'
    '4|8|100|0.001|0.6|True|81|True'
    '4|8|100|0.001|0.6|True|72|True'
    '4|8|100|0.001|0.6|True|64|True'
    '4|8|100|0.001|0.6|True|55|True'
    '4|8|100|0.001|0.6|True|46|True'
    '4|8|100|0.001|0.6|True|37|True'
    '4|8|100|0.001|0.6|True|28|True'
    '4|8|100|0.001|0.6|True|19|True'
    '4|8|100|0.001|0.6|True|10|True'
    '5|32|50|0.001|0.6|False|10|True'
)

function training_model(){
    # shellcheck disable=SC2034
    local range proto port
    for fields in "${connections[@]}"
    do
        IFS=$'|' read -r train_step batch_size total_epochs learning_rate dropout fine_tune fine_tune_at save_best_only <<< "$fields"
        python train.py "--train_step=${train_step}" "--batch_size=${batch_size}" "--dataset_format=tf-records" "--dataset=orchids52_data" "--dataset_version=v1" "--model=mobilenet_v2_140_stn_v15" "--total_epochs=${total_epochs}" "--learning_rate_decay=${learning_rate_decay}" "--learning_rate=${learning_rate}" "--dropout=${dropout}" "--fine_tune=${fine_tune}" "--fine_tune_at=${fine_tune_at}" "--save_best_only=${save_best_only}" "--save_model=True" "--bash=False" "--trained_dir=${trained_dir}"
    done
}

if [[ -f "${script}" ]]; then
	echo "running for script: ${script}"
	training_model
else
	echo "Invalid script filename."
fi
