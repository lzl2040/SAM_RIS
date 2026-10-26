# download data
REFER_DATA_ROOT="${REFER_DATA_ROOT:-/home/v-zuoleili/Data/coco}"
PRETRAIN_WEIGHT_ROOT="${PRETRAIN_WEIGHT_ROOT:-/home/v-zuoleili/Pretrain}"
case "$PRETRAIN_WEIGHT_ROOT" in
    /*) ;;
    *) PRETRAIN_WEIGHT_ROOT="$PWD/$PRETRAIN_WEIGHT_ROOT" ;;
esac
sudo apt install unzip
mkdir -p "$REFER_DATA_ROOT"
cd "$REFER_DATA_ROOT"
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip -d ./ && rm train2014.zip
wget https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
wget https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
wget https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip
unzip refcoco.zip && rm refcoco.zip
unzip refcoco+.zip && rm refcoco+.zip
unzip refcocog.zip && rm refcocog.zip

# download pretrained weight
mkdir -p "$PRETRAIN_WEIGHT_ROOT"
cd "$PRETRAIN_WEIGHT_ROOT"
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth

# create env
# cd /home/v-zuoleili
# wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
# bash Anaconda3-2024.10-1-Linux-x86_64.sh
# source .bashrc
# conda create -n ris python=3.10
# conda activate ris
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
# pip install -r requirements.txt
# pip install numpy==1.26.3
# git config --global user.name "lzl2040"
# git config --global user.email "2040171586@qq.com"
