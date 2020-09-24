#!/bin/bash

# Example: $ bash run.sh ohe_logres.py
# Change python file name after run.sh to run for other python files.

PY_FILE=$1
MODEL=$2
if [[ $PY_FILE == "" || ($PY_FILE == "lbl_rf.py" && $MODEL == "") ]]
then
    echo "Enter python file name and Model name to run. Ex: 'ohe_logres.py' or 'lbl_rf.py xgb'"
else
    echo "Training for" $1
    PARAM_2=""
    if [ $PY_FILE == "lbl_rf.py" ]
    then
        PARAM_2="--model "$MODEL
        echo $PARAM_2
    fi
    # Shell script to run {python file} for all folds
    python3 -W ignore $PY_FILE --fold 0 $PARAM_2
    python3 -W ignore $PY_FILE --fold 1 $PARAM_2
    python3 -W ignore $PY_FILE --fold 2 $PARAM_2
    python3 -W ignore $PY_FILE --fold 3 $PARAM_2
    python3 -W ignore $PY_FILE --fold 4 $PARAM_2
fi