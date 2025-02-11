set -e
# deactivate



echo "Creating new virtualenv"

virtualenv ~/$1
source ~/$1/bin/activate

echo "Activating virtual env"

cd ../../

mmwhale_dir=$(pwd)

pip install --no-index --upgrade pip

pip install wandb
pip install opencv-python-headless
pip install numpy
pip install matplotlib
pip install torch
pip install torchvision
pip install tqdm
pip install scikit-learn
# pip install # ipywidgets==8.0.2
pip install jupyterlab
pip install ipywidgets
pip install icecream
pip install mmengine>=0.8.3
pip install mmcv
cd $mmwhale_dir
pip install -r requirements/multimodal.txt
pip install -v -e .
pip install albumentations==1.3.0
pip install sahi
cd faster_coco_eval_repo
pip install -v -e .
pip install rich
pip install mmpretrain
# # build mmcv from source-- NEED TO HAVE GPU FOR THIS
# git clone https://github.com/open-mmlab/mmcv.git
# cd ../mmcv
# MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -v -e .

# install mmcv from pip
# MMCV_WITH_OPS=1 FORCE_CUDA=1  mim install mmcv --no-deps

# install mmdet
# pip uninstall opencv-python
# pip uninstall opencv-python-headless
# pip install opencv-python-headless
