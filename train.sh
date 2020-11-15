#!/bin/bash

#ROOT_PATH="${WORKSPACE}"
#MODELS_DIR="${ROOT_PATH}/orchids-models"
TRAIN_DIR=$PWD

echo "${TRAIN_DIR}"

file="${TRAIN_DIR}/train.out"
if [[ -f "$file" ]]; then
    echo "delete ${file}"
	rm "${file}"
fi

script="${TRAIN_DIR}/train.py"
if [[ -f "${script}" ]]; then
	echo "running for script: ${script}"
	nohup python "${script}" "--bash=True" "--start_state=1" "--end_state=5" "--dataset=orchids52_v1_tf" "--model=mobilenet_v2_140_orchids52" "--learning_rate=0.001" "--total_epochs=100,100,20,100" --aug_method=slow > train.out 2>&1 &
    echo $! > train_pid.txt
else
	echo "Invalid script filename."
fi
