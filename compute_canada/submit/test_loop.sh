#!/bin/bash 
set -e
array=(
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/AED_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/aerial_livestock_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/Aerial_seabird_westafrica_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/Beluga_2014_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/Beluga_2015_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/Beluga_2017_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/birds_izembek_lagoon_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/birds_poland_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/Eikelboom_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/michigan_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/monash_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/Narwhal_2016_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/new_mexico_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/NOAA_artic_seal_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/NOAA_sealion_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/palmyra_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/penguins_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/pfeifer_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/qian_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/SAVMAP.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/seabirdwatch_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/turtle_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs/WAID_livestock_dataset.py
# configs/mm_grounding_dino_animals/test_configs/Virunga_garamba_dataset.py

)
for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch test.sh ${array[i]}
   # bash test2.sh ${array[i]}
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 1
done