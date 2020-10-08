#!/bin/bash
PY_FILE=$1
MODEL=$2
echo "Training with" $MODEL

python3 -W ignore $PY_FILE --fold 0 --model $MODEL
python3 -W ignore $PY_FILE --fold 1 --model $MODEL
python3 -W ignore $PY_FILE --fold 2 --model $MODEL
python3 -W ignore $PY_FILE --fold 3 --model $MODEL
python3 -W ignore $PY_FILE --fold 4 --model $MODEL