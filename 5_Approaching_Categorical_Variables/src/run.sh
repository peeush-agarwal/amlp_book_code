#!/bin/bash

# Example: $ bash run.sh ohe_logres.py
# Change python file name after run.sh to run for other python files.

echo "Training for" $1
PY_FILE=$1
if [ "$PY_FILE" == "" ]
then
    echo "Enter python file name to run. Ex: ohe_logres.py"
else
    # Shell script to run {python file} for all folds
    python3 -W ignore $PY_FILE --fold 0
    python3 -W ignore $PY_FILE --fold 1
    python3 -W ignore $PY_FILE --fold 2
    python3 -W ignore $PY_FILE --fold 3
    python3 -W ignore $PY_FILE --fold 4
fi