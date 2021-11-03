#!/bin/bash

TRAIN_DIR=$PWD
trained_dir="${WORKSPACE}/_trained_models/checkpoints/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt"
script="${TRAIN_DIR}/train.py"

#Hyperparameters
readonly connections=(
    '1|32|100|0.001|0.8|False|True'
    #'1|64|100|0.0085|0.8|False|True'
    #'1|64|200|0.0005|0.8|True|True'
    #'2|4|5|0.001|0.8|False|False'
    #'3|4|5|0.001|0.8|False|False'
    #'4|4|100|0.0001|0.5|False|True'
    #'5|4|200|0.001|0.5|True|True'
)

function training_model(){
    # shellcheck disable=SC2034
    local range proto port
    for fields in "${connections[@]}"
    do
        IFS=$'|' read -r train_step batch_size total_epochs learning_rate dropout fine_tune save_best_only <<< "$fields"
        python train.py "--train_step=${train_step}" "--fine_tune=${fine_tune}" "--batch_size=${batch_size}" "--save_best_only=${save_best_only}" "--dataset_format=tf-records" "--dataset=orchids52_data" "--dataset_version=v1" "--model=resnet_v2_50_stn_v15" "--learning_rate=${learning_rate}" "--dropout=${dropout}" "--total_epochs=${total_epochs}" "--save_model=True" "--bash=False" "--learning_rate_decay=exponential" "--trained_dir=${trained_dir}"
        #python train.py "--train_step=${train_step}" "--fine_tune=${fine_tune}" "--batch_size=${batch_size}" "--save_best_only=${save_best_only}" "--dataset_format=tf-records" "--dataset=orchids52_data" "--dataset_version=v1" "--model=mobilenet_v2_140_stn_v15" "--learning_rate=${learning_rate}" "--dropout=${dropout}" "--total_epochs=${total_epochs}" "--save_model=True" "--bash=False" "--learning_rate_decay=exponential" "--trained_dir=${trained_dir}"
        #echo "--train_step=${train_step}" "--batch_size=${batch_size}" "--dataset_format=files" "--dataset=orchids52_data" "--dataset_version=v1" "--model=mobilenet_v2_140_stn_v15" "--learning_rate=${learning_rate}" "--total_epochs=${total_epochs}" "--save_model=True" "--bash=False" "--learning_rate_decay=exponential" "--trained_dir=${trained_dir}"
    done
}

if [[ -f "${script}" ]]; then
	echo "running for script: ${script}"
	training_model
else
	echo "Invalid script filename."
fi
