#!/bin/bash

TRAIN_DIR=$PWD
script="${TRAIN_DIR}/start_train.sh"

echo "${TRAIN_DIR}"

file="${TRAIN_DIR}/train.out"
if [[ -f "$file" ]]; then
    echo "delete ${file}"
	rm "${file}"
fi

nohup "${script}" > train.out 2>&1 &
echo $! > train_pid.txt
