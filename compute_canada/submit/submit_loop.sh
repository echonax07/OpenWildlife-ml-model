#!/bin/bash 
set -e
array=(
configs/DFO_whale_different_backbones/DFO_whale_deformable_detr_2048_convnext_T.py
)
for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train.sh ${array[i]}
   # bash test2.sh ${array[i]}
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 1
done