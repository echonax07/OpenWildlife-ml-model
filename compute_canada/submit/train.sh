#!/bin/bash
#SBATCH --nodes 2
#SBATCH --gpus-per-node=4 # request a GPU
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=12 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=400G
#SBATCH --time=23:59:00
#SBATCH --output=../output3/%j.out
#SBATCH --account=rrg-dclausi
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
export WANDB_DATA_DIR='/home/m32patel/scratch/wandb'

# Generate a random 5-digit number
min=10000
max=99999
random_number=$(( RANDOM % (max - min + 1) + min ))
echo "Config file: $1"
# srun --ntasks=2 --gres=gpu:2  --kill-on-bad-exit=1 --cpus-per-task=12 python tools/train.py $1 --launcher slurm --resume
# srun --ntasks=2 --gres=gpu:2  --kill-on-bad-exit=1 --cpus-per-task=12 python tools/train.py $1 --launcher slurm
srun --ntasks=8 --gres=gpu:4 --kill-on-bad-exit=1 --cpus-per-task=12 --nodes=2 python tools/train.py $1 --launcher slurm --cfg-options env_cfg.dist_cfg.port=${random_number} --resume

# Extract the base name without extension
base_name=$(basename "$1" .py)
CHECKPOINT=$(cat work_dir_grounding_dino/$base_name/last_checkpoint)
echo "$CHECKPOINT"

srun --ntasks=8 --gres=gpu:4 --kill-on-bad-exit=1 --cpus-per-task=12 python tools/test.py $1 $CHECKPOINT --launcher slurm --out True --cfg-options env_cfg.dist_cfg.port=${random_number} class_name=('cat', ) num_classes=len(class_name) metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

# python tools/analysis_tools/whale/plot_pr_confusion_matrix_year_wise.py --config $1 --save_year_wise=False

# python tools/dataset_converters/whale/convert_mmdet_pred_to_labelstudio_tasks.py /home/m32patel/projects/def-dclausi/whale/merged/test/test_non_whale.json work_dirs/$base_name/test_results_whale_non_whale.bbox.json work_dirs/$base_name/test_results_whale_non_whale_labelstudio.bbox.json --LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/media/pc2041
# # python tools/train.py $1

# # the above python script will generate a .env at the workdir/config-name/.env
# env=./work_dir/$config_basename/.env

# echo 'Reading environment file'
# # read the .env file and save them as environment variable
# while read line; do export $line; done < $env

# echo "Starting testing"
# python test_upload.py $1 $CHECKPOINT