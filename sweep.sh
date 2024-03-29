#!/bin/bash

timestamp=$(date +"%Y%m%d%H%M%S%N")  # %N for nanoseconds

random_suffix=$(($RANDOM % 1000))  # Generates a random number between 0 and 999

jobname="wandb_tutorial_${timestamp}_${random_suffix}"

qsub -N "${jobname}" -o "/projectnb/tianlabdl/jalido/wandb_tutorial/logs/${jobname}.qlog" "sweep.qsub"