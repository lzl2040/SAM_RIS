## Preparation
download huggingface token:
```shell
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
```
## Run
### Train
```shell
export HF_HOME="/your/custom/cache/directory"
export TORCH_HOME="/your/custom/cache/directory"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py -model_cfg sam2.1_hiera_s.yaml -num_epochs 40 -batch_size 16 -image_size 1024 -train_visual -distributed
```
### Test
```shell
python test.py -model_cfg sam2.1_hiera_s.yaml -device 0 --split val
```