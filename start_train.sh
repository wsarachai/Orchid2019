#!/bin/bash

TRAIN_DIR=$PWD
trained_dir="${WORKSPACE}/_trained_models/model-v1/mobilenet_v2_140_orchids52_0001/pretrain2/model.ckpt-12000"
script="${TRAIN_DIR}/train.py"

readonly connections=(       
    '1|32|100|0.001'
    '2|4|15|0.001'
    '3|4|15|0.001'
    '4|4|12|0.001'
)

function training_model(){
    # shellcheck disable=SC2034
    local range proto port
    for fields in "${connections[@]}"
    do
        IFS=$'|' read -r train_step batch_size total_epochs learning_rate <<< "$fields"
        python train.py "--train_step=${train_step}" "--batch_size=${batch_size}" "--dataset_format=files" "--dataset=orchids52_data" "--dataset_version=v1" "--model=mobilenet_v2_140_stn_v15" "--learning_rate=${learning_rate}" "--total_epochs=${total_epochs}" "--save_model=True" "--bash=False" "--learning_rate_decay=exponential" "--trained_dir=${trained_dir}"
        #echo "--train_step=${train_step}" "--batch_size=${batch_size}" "--dataset_format=files" "--dataset=orchids52_data" "--dataset_version=v1" "--model=mobilenet_v2_140_stn_v15" "--learning_rate=${learning_rate}" "--total_epochs=${total_epochs}" "--save_model=True" "--bash=False" "--learning_rate_decay=exponential" "--trained_dir=${trained_dir}"
    done
}

if [[ -f "${script}" ]]; then
	echo "running for script: ${script}"
	training_model
else
	echo "Invalid script filename."
fi
