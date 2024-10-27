#!/bin/bash 
set -e
array=(
configs/mm_grounding_dino_animals/grouding_dino_swin-t_finetune_all.py
)
for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train.sh ${array[i]}
   # bash test2.sh ${array[i]}
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 1
done