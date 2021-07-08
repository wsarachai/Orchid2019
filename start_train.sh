#!/bin/bash

TRAIN_DIR=$PWD
trained_path="/home/keng/Documents/_trained_models/model-v1/mobilenet_v2_140_orchids52_0001/pretrain2/model.ckpt-12000"
script="${TRAIN_DIR}/train.py"

readonly connections=(       
    '2|10|0.01'
    '3|20|0.01'
    '4|30|0.01'
    '5|40|0.001'
)

function training_model(){
    local range proto port
    for fields in ${connections[@]}
    do
        IFS=$'|' read -r train_step total_epochs learning_rate <<< "$fields"
        python train.py "--train_step=${train_step}" "--batch_size=4" "--dataset_format=files" "--dataset=orchids52_data" "--dataset_version=v1" "--model=mobilenet_v2_140_stn_v15" "--learning_rate=${learning_rate}" "--total_epochs=${total_epochs}" "--save_model=True" "--bash=False" "--trained_path=${trained_path}"
    done
}

if [[ -f "${script}" ]]; then
	echo "running for script: ${script}"
	training_model
else
	echo "Invalid script filename."
fi
