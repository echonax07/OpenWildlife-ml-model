#!/bin/bash 
set -e
array=(
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/AED_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/aerial_livestock_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/Aerial_seabird_westafrica_od_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/Beluga_2014_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/Beluga_2015_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/Beluga_2017_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/birds_izembek_lagoon_od_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/birds_poland_od_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/Eikelboom_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/michigan_od_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/monash_od_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/Narwhal_2016_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/new_mexico_od_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/NOAA_artic_seal_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/NOAA_sealion_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/palmyra_od_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/penguins_od_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/pfeifer_od_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/qian_od_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/SAVMAP.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/seabirdwatch_od_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/turtle_dataset.py
/home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/WAID_livestock_dataset.py
configs/mm_grounding_dino_animals/test_configs_caption/Virunga_garamba_dataset.py

)
for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch test.sh ${array[i]}
   # bash test2.sh ${array[i]}
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 1
done