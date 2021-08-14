#!/bin/bash

TRAIN_DIR=$PWD
trained_dir="${WORKSPACE}/_trained_models/checkpoints/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224.ckpt"
script="${TRAIN_DIR}/train.py"

#Hyperparameters
readonly connections=(
    #train_step|batch_size|total_epochs|learning_rate_boundaries|learning_rate|dropout|fine_tune|fine_tune_at|save_best_only
    #'1|64|100|0.0085|0.8|False|True'
    #'1|64|200|0.0005|0.8|True|True'
    #'2|4|5|0.001|0.8|False|False'
    #'3|4|5|0.001|0.8|False|False'
    #'4|4|100|0.0001|0.5|False|True'
    #'5|4|200|0.001|0.5|True|True'
    '1|64|150|60,100|0.001,0.0005,0.0001|0.6|False|0|True'
    '1|32|400|75|0.00001,0.000001|0.6|True|99|True'
    '1|32|400|75|0.00001,0.000001|0.6|True|90|True'
    '1|32|400|75|0.00001,0.000001|0.6|True|81|True'
    '1|32|400|75|0.00001,0.000001|0.6|True|72|True'
    '1|32|400|75|0.00001,0.000001|0.6|True|64|True'
    '1|32|400|75|0.00001,0.000001|0.6|True|55|True'
    '1|32|400|75|0.00001,0.000001|0.6|True|46|True'
    '1|32|400|75|0.00001,0.000001|0.6|True|37|True'
    '1|32|400|75|0.00001,0.000001|0.6|True|28|True'
    '1|32|400|75|0.00001,0.000001|0.6|True|19|True'
)

function training_model(){
    # shellcheck disable=SC2034
    local range proto port
    for fields in "${connections[@]}"
    do
        IFS=$'|' read -r train_step batch_size total_epochs learning_rate_boundaries learning_rate dropout fine_tune fine_tune_at save_best_only <<< "$fields"
        python train.py "--train_step=${train_step}" "--batch_size=${batch_size}" "--dataset_format=tf-records" "--dataset=orchids52_data" "--dataset_version=v1" "--model=mobilenet_v2_140_stn_v15" "--total_epochs=${total_epochs}" "--learning_rate_decay=piecewise_constant" "--learning_rate_boundaries=${learning_rate_boundaries}" "--learning_rate=${learning_rate}" "--fine_tune=${fine_tune}" "--fine_tune_at=${fine_tune_at}" "--save_best_only=${save_best_only}" "--dropout=${dropout}" "--save_model=True" "--bash=False" "--trained_dir=${trained_dir}"
        # params="--train_step=${train_step} --batch_size=${batch_size} --dataset_format=tf-records --dataset=orchids52_data --dataset_version=v1 --model=mobilenet_v2_140_stn_v15 --total_epochs=${total_epochs} --learning_rate_decay=piecewise_constant --learning_rate_boundaries=${learning_rate_boundaries} --learning_rate=${learning_rate} --fine_tune=${fine_tune} --fine_tune_at=${fine_tune_at} --save_best_only=${save_best_only} --dropout=${dropout} --save_model=True --bash=False --trained_dir=${trained_dir}"
        # echo "${params}"
    done
}

if [[ -f "${script}" ]]; then
	echo "running for script: ${script}"
	training_model
else
	echo "Invalid script filename."
fi
