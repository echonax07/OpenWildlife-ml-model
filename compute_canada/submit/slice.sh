#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=50G
#SBATCH --time=13:59:00
#SBATCH --output=../output3/%j.out
#SBATCH --account=def-l44xu-ab
#SBATCH --mail-user=muhammed.computecanada@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
set -e

module purge
module load python/3.10
echo "Loading module done"
module load StdEnv/2020 gcc/9.3.0 opencv/4.8.0 cuda/11.7

source ~/env_grounding_dino/bin/activate

echo "Activating virtual environment done"

cd $HOME/projects/def-dclausi/whale/mmwhale2

echo "starting training..."
# config=$1 
# # get the basename for the config file, basename is an inbuilt shell command
# config_basename=$(basename $config .py) 

export WANDB_MODE=offline

python slice_img.py --enable=True --data_root_whole=/home/m32patel/projects/rrg-dclausi/wildlife/datasets/birds_Izembek_Lagoon_Waterfowl --data_root_slice=/home/m32patel/scratch/animal_patches/birds_Izembek_Lagoon_Waterfowl/ --ann_file=train.json --img_dir=''  --slice_height=1024 --slice_width=1024 --overlap_height_ratio=0 --overlap_width_ratio=0 --save_only_positive_slices=True 


