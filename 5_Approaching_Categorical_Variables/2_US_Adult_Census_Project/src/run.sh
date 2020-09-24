#!/bin/bash
echo "Training with" $1

python3 -W ignore $1 --fold 0
python3 -W ignore $1 --fold 1
python3 -W ignore $1 --fold 2
python3 -W ignore $1 --fold 3
python3 -W ignore $1 --fold 4