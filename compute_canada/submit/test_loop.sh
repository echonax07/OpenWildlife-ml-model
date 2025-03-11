#!/bin/bash 
set -e
array=(
   # Caption
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/AED_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/aerial_livestock_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/Aerial_seabird_westafrica_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/Beluga_2014_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/Beluga_2015_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/Beluga_2017_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/birds_izembek_lagoon_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/birds_poland_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/Eikelboom_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/michigan_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/monash_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/Narwhal_2016_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/new_mexico_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/NOAA_artic_seal_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/NOAA_sealion_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/palmyra_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/penguins_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/pfeifer_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/qian_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/SAVMAP.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/seabirdwatch_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/turtle_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/WAID_livestock_dataset.py
# configs/mm_grounding_dino_animals/test_configs_caption/Virunga_garamba_dataset.py

# No caption

# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/AED_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/aerial_livestock_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/Aerial_seabird_westafrica_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/Beluga_2014_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/Beluga_2015_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/Beluga_2017_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/birds_izembek_lagoon_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/birds_poland_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/Eikelboom_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/michigan_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/monash_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/Narwhal_2016_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/new_mexico_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/NOAA_artic_seal_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/NOAA_sealion_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/palmyra_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/penguins_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/pfeifer_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/qian_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/SAVMAP.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/seabirdwatch_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/turtle_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/WAID_livestock_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/Virunga_garamba_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_no_caption/polar_bear.py
# configs/mm_grounding_dino_animals/test_configs_no_caption/DFO_Whale23.py
# configs/mm_grounding_dino_animals/test_configs_caption/DFO_Whale23.py


# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_caption/NOAA_artic_seal_dataset.py

# # Viz caption
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/AED_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/aerial_livestock_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/Aerial_seabird_westafrica_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/Beluga_2014_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/Beluga_2015_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/Beluga_2017_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/birds_izembek_lagoon_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/birds_poland_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/Eikelboom_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/michigan_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/monash_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/Narwhal_2016_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/new_mexico_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/NOAA_artic_seal_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/NOAA_sealion_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/palmyra_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/penguins_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/pfeifer_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/qian_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/SAVMAP.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/seabirdwatch_od_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/turtle_dataset.py
# /home/m32patel/projects/def-dclausi/whale/mmwhale2/configs/mm_grounding_dino_animals/test_configs_viz_caption/WAID_livestock_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption/Virunga_garamba_dataset.py


# No caption new split

# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/AED_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/aerial_livestock_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/Aerial_seabird_westafrica_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/Beluga_2014_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/Beluga_2015_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/Beluga_2017_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/birds_izembek_lagoon_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/birds_poland_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/create_config_files.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_sqsplit/demo_config_big_pipeline.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/demo_config_small_pipeline.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/DFO_Whale23.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/Eider_duck.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/Eikelboom_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/filname.txt
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/michigan_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/monash_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/Narwhal_2016_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/new_mexico_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/NOAA_artic_seal_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/NOAA_sealion_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/palmyra_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/penguins_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/pfeifer_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/polar_bear.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/qian_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/SAVMAP.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/seabirdwatch_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/turtle_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/Virunga_garamba_dataset.py
# configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/WAID_livestock_dataset.py
configs/mm_grounding_dino_animals/test_configs_no_caption_new_split/Narwhal_2016_dataset_all_images_of_survey.py


# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/AED_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/aerial_livestock_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/Beluga_2014_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/Beluga_2015_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/Beluga_2017_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/birds_izembek_lagoon_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/DFO_Whale23.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/Eider_duck.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/Eikelboom_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/michigan_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/monash_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/Narwhal_2016_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/new_mexico_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/NOAA_artic_seal_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/NOAA_sealion_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/palmyra_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/penguins_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/pfeifer_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/polar_bear.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/qian_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/SAVMAP.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/seabirdwatch_od_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/turtle_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/Virunga_garamba_dataset.py
# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/WAID_livestock_dataset.py


# configs/mm_grounding_dino_animals/test_configs_viz_caption_new_split/Narwhal_2016_dataset_all_survey.py
)

checkpoint_file=
for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch test.sh ${array[i]}
   # bash test2.sh ${array[i]}
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 1
done