#!/bin/bash

CUR_DIR=$PWD
script="tensorboard --logdir=$1 --bind_all"

echo "${script}"

file="${CUR_DIR}/board.out"
if [[ -f "$file" ]]; then
    echo "delete ${file}"
	rm "${file}"
fi

nohup "${script}" > board.out 2>&1 &
echo $! > train_pid.txt
