## Preparation
download huggingface token:
```shell
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
```

## qformer笔记
输入给qformer的是[query + text_tokens]的拼接，它们作为输入，图像特征会作为encoder_hidden_states，类似于kv

在每一层，[query + text_tokens]先进行self-attention，然后输出的query单独拿出来，跟图像特征进行cross-attention

## Run
batch size=16,
- img_size=1024: 60G
- img_size=384: bs=16, 12G; bs=64, 32G; bs=128, 62G
- img_size=480: bs=32, 27G; bs=96, 74G

### Train
```shell
export HF_HOME="/your/custom/cache/directory"
export TORCH_HOME="/your/custom/cache/directory"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py --model_cfg sam2.1_hiera_b+.yaml --num_epochs 40 --batch_size 16 --image_size 480 --train_visual --dataset all --task_name MD_SAM_img_size_480_b+ --distributed
```
### Test
```shell
python test.py --model_cfg sam2.1_hiera_l.yaml --device 0 --split val --dataset refcoco --image_size 480 --splitBy unc --refer_data_root /data/Data/coco
```

## Error
return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
NotImplementedError: Cannot copy out of meta tensor; no data!
使用如下代码检查是否有meta：
```
print(f"\n🔍 Checking meta tensors in blip2..")
has_meta = False
for name, param in self.blip2_qformer.named_parameters():
    if param.device.type == 'meta':
        print(f"[META PARAM] {name}, shape: {param.shape}")
        has_meta = True
if not has_meta:
    print("✅ No meta tensors found.")
```
然后删掉它