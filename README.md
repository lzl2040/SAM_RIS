## Preparation

## Run
```
export HF_HOME="/your/custom/cache/directory"
export TORCH_HOME="/your/custom/cache/directory"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py -model_cfg sam2.1_hiera_s.yaml -num_epochs 40 -batch_size 16 -train_visual -distributed
```