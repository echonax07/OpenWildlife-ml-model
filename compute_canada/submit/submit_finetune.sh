#!/bin/bash 
set -e
array=(
configs/mm_grounding_dino_animals/finetune_configs_no_caption_new_split/DFO_whale_23.py #schedule 4hrs
# configs/mm_grounding_dino_animals/finetune_configs_no_caption_new_split/Eikelboom_dataset.py #schedule 3hrs
# configs/mm_grounding_dino_animals/finetune_configs_no_caption_new_split/Han_aerial_livestock_dataset.py #live 1 hrs
# configs/mm_grounding_dino_animals/finetune_configs_no_caption_new_split/michigan_od_dataset.py #schedule 7hrs
# configs/mm_grounding_dino_animals/finetune_configs_no_caption_new_split/Narwhal_2016_dataset.py #schedule 4 hrs
# configs/mm_grounding_dino_animals/finetune_configs_no_caption_new_split/NOAA_arctic_seal_dataset.py #schedule 1 day
# configs/mm_grounding_dino_animals/finetune_configs_no_caption_new_split/penguins_od_finetune.py #live 30 mins
# configs/mm_grounding_dino_animals/finetune_configs_no_caption_new_split/polar_bear_finetune.py #live 30 minns
# configs/mm_grounding_dino_animals/finetune_configs_no_caption_new_split/qian_finetune.py #live 30 mins
# configs/mm_grounding_dino_animals/finetune_configs_no_caption_new_split/tern_bioarxiv.py #live 15 mins
# configs/mm_grounding_dino_animals/finetune_configs_no_caption_new_split/WAID_livestock_dataset.py #schedule 10 hrs
)
for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch finetune.sh ${array[i]}
   # bash test2.sh ${array[i]}
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 1
done