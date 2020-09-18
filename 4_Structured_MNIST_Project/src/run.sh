#!/bin/bash

# Example: $ sh run.sh dt_gini
# change model name after run.sh to run for other model defined in model_dispatcher

MODEL=$1
python3 train.py --fold 0 --model $MODEL
python3 train.py --fold 1 --model $MODEL
python3 train.py --fold 2 --model $MODEL
python3 train.py --fold 3 --model $MODEL
python3 train.py --fold 4 --model $MODEL