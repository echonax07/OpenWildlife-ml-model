#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=4 # request a GPU
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=12 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=200G
#SBATCH --time=2:59:00
#SBATCH --output=../output4/%j.out
#SBATCH --account=rrg-dclausi
#SBATCH --mail-user=muhammed.computecanada@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
set -e

module --force purge
echo "Loading module done"
module load python/3.10 StdEnv/2023 opencv/4.10.0 cuda/12.2 rust/1.76.0 git-lfs
export WANDB_DATA_DIR='/home/m32patel/scratch/wandb'

source ~/env_mmgrounding_dino3/bin/activate

echo "Activating virtual environment done"

cd $HOME/projects/def-dclausi/whale/mmwhale2

echo "starting training..."
# config=$1 
# # get the basename for the config file, basename is an inbuilt shell command
# config_basename=$(basename $config .py) 


export WANDB_MODE=offline
export WANDB_DATA_DIR='/home/m32patel/scratch/wandb'

echo "Config file: $1"

# Generate a random 5-digit number
min=10000
max=99999
random_number=$(( RANDOM % (max - min + 1) + min ))
echo "Config file: $1"

# python tools/test.py $1 /home/m32patel/projects/def-dclausi/whale/mmwhale2/t/grouding_dino_swin-t_no_caption/epoch_20.pth

# python tools/test.py $1 /home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/grouding_dino_swin-t_caption/epoch_20.pth
# python tools/test.py $1 work_dir_grounding_dino/grouding_dino_swin-t_no_caption_new_split/epoch_20.pth
# srun --ntasks=2 --gres=gpu:2 --kill-on-bad-exit=1 --cpus-per-task=12 --nodes=1 python tools/test.py $1 work_dir_grounding_dino/grouding_dino_swin-t_no_caption_new_split/epoch_20.pth  --launcher slurm --cfg-options env_cfg.dist_cfg.port=${random_number}

srun --ntasks=4 --gres=gpu:4 --kill-on-bad-exit=1 --cpus-per-task=12 --nodes=1 python tools/test.py $1 /home/m32patel/projects/def-dclausi/whale/mmwhale2/work_dir_grounding_dino/finetune_no_caption_new_split/Narwhal_2016_dataset/epoch_10.pth --launcher slurm --cfg-options env_cfg.dist_cfg.port=${random_number}

# python tools/test.py $1 work_dir_grounding_dino/grouding_dino_swin-t_no_caption_new_split/epoch_20.pth

# python tools/test.py $1 work_dir_grounding_dino/grouding_dino_swin-t_viz_caption_new_split/epoch_20.pth


# python tools/analysis_tools/whale/plot_pr_confusion_matrix_year_wise.py --config $1 --save_year_wise=False

# python tools/test.py $1 work_dir_grounding_dino/grouding_dino_swin-t_finetune_all/epoch_20.pth
# python tools/dataset_converters/whale/convert_mmdet_pred_to_labelstudio_tasks.py /home/m32patel/projects/def-dclausi/whale/merged/test/test_non_whale.json work_dirs/$base_name/test_results_whale_non_whale.bbox.json work_dirs/$base_name/test_results_whale_non_whale_labelstudio.bbox.json --LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/media/pc2041
# # python tools/train.py $1

# # the above python script will generate a .env at the workdir/config-name/.env
# env=./work_dir/$config_basename/.env

# echo 'Reading environment file'
# # read the .env file and save them as environment variable
# while read line; do export $line; done < $env

# echo "Starting testing"
# python test_upload.py $1 $CHECKPOINT