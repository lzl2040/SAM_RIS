# Ablation Study for AAAI

## 医学 referring segmentation 数据

`--dataset qata` / `--dataset mosmed` 可用于 `train_with_val.py`、
`test.py` 和 `benchmark_validation.py`；不改变现有 RefCOCO 数据加载。
原始数据分别来自用户指定的 Kaggle QaTa-COV19、MosMedData+，
文本描述和 train/val/test 划分来自 [LViT 官方公开标注](https://github.com/HUANGLIZI/LViT)。
QaTa 明确使用原始包的 **v2**，不混入同名的 v1 或无对应描述的对照组。

```bash
# 与 coco 同级下载，解压并生成 anns/{train,val,test}.json；需要 openpyxl。
python prepare_medical.py --data_root /data2/lizuolei/data
# 已下载时跳过联网；--dataset qata 或 mosmed 可只处理一个。
python prepare_medical.py --data_root /data2/lizuolei/data --skip_download

# 预训练权重、WORK_DIR 等仍沿用下文的环境变量配置。
export MEDICAL_DATA_ROOT=/data2/lizuolei/data
DATASET=qata ABLATION=full bash train_with_val_new.sh
DATASET=mosmed ABLATION=full bash train_with_val_new.sh
```

数据目录为 `$MEDICAL_DATA_ROOT/QaTa-COV19/`、`MosMedDataPlus/`，
各含 `dataset.zip`、`raw/`、`annotations/`、`anns/`。
划分数量（train/val/test）：QaTa **5716/1429/2113**，
MosMed **2183/273/273**。脚本拒绝缺失/歧义配对、尺寸不一致及跨划分同名图像。
这些是原公开划分，不额外承诺患者级互斥。
灰度 PNG 转三通道 RGB，掩码按 `>0` 二值化；保留原始文字，不随机生成描述，
不引入可能颠倒 left/right 语义的翻转。每张图一条描述，因此之前的
`val_batch_size` 句子批处理对这两套数据通常没有提速作用。

这是把公开医学图像、掩码和文本接入当前 SAM_RIS，**不是复现 MedMARS 模型或其完整实验协议**。
仍沿用当前训练损失和验证的 mIoU/oIoU/precision；未额外实现医学论文的 Dice 等报告口径。
不要直接将这些指标与采用不同预处理、划分或 Dice 定义的论文数字对比。

## 小修受控消融（新实验，不替换下方历史结果）

入口为 `train_with_val.py` / `train_with_val_new.sh`，四组均保留同一
InstructBLIP/Q-Former、SAM2 骨干及投影层，不再以 CLIP 替换代表关闭 TMFM。

| `--ablation` / `ABLATION` | MMA | TMFM | decoder 提示 |
|---|---|---|---|
| `baseline` | 无 | 无 | 第一次 Q-Former 输出 Q_e |
| `mma_only` | 有 | 无 | Q_e |
| `tmfm_only` | 无 | 有 | 第二次融合后的文本 token |
| `full` | 有 | 有 | 第二次融合后的文本 token |

```bash
export PRETRAIN_WEIGHT_ROOT=/data2/lizuolei/weights/pretrained
export REFER_DATA_ROOT=/data2/lizuolei/data/coco
export BERT_MODEL_PATH="$PRETRAIN_WEIGHT_ROOT/bert-base-uncased"
export EVA_VIT_PATH="$PRETRAIN_WEIGHT_ROOT/eva_vit_g.pth"
export WORK_DIR=/data2/lizuolei/weights/MM_SAM/work_dir
export LOG_PATH=/data2/lizuolei/weights/MM_SAM/log_path

# 四次独立训练：所有训练设置保持一致；batch_size 为每卡大小。
for mode in baseline mma_only tmfm_only full; do
    ABLATION="$mode" NPROC_PER_NODE=8 BATCH_SIZE=18 \
        bash train_with_val_new.sh
done

# 单数据集示例；RefCOCO/+ 使用 unc，RefCOCOg 使用 umd。
ABLATION=full DATASET=refcocog NPROC_PER_NODE=8 BATCH_SIZE=18 \
    bash train_with_val_new.sh --splitBy umd

# 相同数据/划分/图像尺寸、相同顺序下，导出四组预测概率图和掩码。
# 每组分别传入其 checkpoint，并使用各自的 --ablation。
python test.py --ablation baseline --resume /path/to/baseline/ris_sam_model_best_oiou.pth \
    --model_cfg sam2.1_hiera_l.yaml --image_size 480 \
    --dataset refcoco --split val --splitBy unc --controlled_visuals
```

`--tmfm_output`（脚本环境变量 `TMFM_OUTPUT`）可选择
`text`（默认）、`query`、`all`、`cls`，用于 TMFM 输出表示实验。
关闭 TMFM 时固定使用 Q_e；不要将输出选项实验当成新增独立模块。

新脚本默认启用 `--exclude_eval_images`，从训练数据中去掉所有已加载
数据集的非训练引用对应的图像；混合训练加载 RefCOCO/UNC、
RefCOCO+/UNC、RefCOCOg/UMD，不包含 Google 划分。
单集默认验证本集；混合默认验证 RefCOCO。
可通过 `--val_dataset`、`--val_split_by` 显式指定，但四组应保持一致。

验证默认将同一对象的最多 4 句描述合批（`VAL_BATCH_SIZE` /
`--val_batch_size`），仍遍历全部描述，在 GPU 上累计逐句 IoU 后计算
mIoU/oIoU/Precision，不保存图片。可设为 1 对照原逐句方式；显存不足时
降低该值。`--val_amp` 可选开启验证 AMP，默认关闭，开启可能产生指标数值
差异，四组需统一。验证目前仍由 rank 0 单卡执行，未做多卡分片。

已恢复旧版实际训练行为：四组的 Q-Former 和 query tokens 均冻结，
特征提取使用 `no_grad`，投影层、mask decoder、no-mask embedding 及启用的
MMA 仍可训练。不额外切换 Q-Former 到 eval 模式，保留旧版 train/eval 行为。
之前 `qformer_grad=True` 的受控实验与恢复后的设置不同，不能续训其异常权重；
需换新的 `TASK_NAME`（如 `1103_mm_sam_full_all_frozen`）从预训练权重重新开始，
也不要通过 `--pretrain` 加载这些异常任务权重。可训练 no-mask embedding 仍写入 checkpoint。
四组（包括 full）都应按统一设置新训；历史完整模型只可作为旧结果，不是本组 full。
checkpoint 保存实验设置，配置不同或缺少设置的旧文件不能用于续训。
`--pretrain` 仅用于有意初始化，不是续训；严格受控比较不要用完整任务模型
初始化某一消融组。SAM2.1 与论文原 SAM2 也不能混用为同一骨干。

`--controlled_visuals` 导出的是最终预测概率图，不是 MMA attention。
无 MMA 的模型没有 MMA attention；比较图像特征响应与预测响应时应明确区分。

full epoch, epoch=40, img_size=480, sam2-large, all data, test on refcoco val
## Components Study
| Epoch | mIoU | oIoU | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 |
|-------|---------|------------|---------------|---------------|---------------|---------------|---------------|
|   WO-Adapter   | 71.49 | 68.32 | 79.44% | 75.69% | 71.62% | 63.20% | 34.42% |
| WO-BLIP2 (replace it with CLIP)   | 78.38 | 75.39 | 87.75% | 85.50% | 82.42% | 74.20% | 44.82% |
|   Full: WO-CNN   | 79.44 | 77.49 | 88.90% | 86.58% | 83.25% | 75.24% | 45.24% |
<!-- |   Full   | 79.10 | 76.63 | 88.40% | 86.35% | 83.36% | 75.05% | 45.83% | -->
## Complexity and Effectiveness
| Method | Params | Flops | mIoU | oIoU |
|-------|---------|------------|---------------|---------------|


## Text-MultiModal Fusion Module Study
| Epoch | mIoU | oIoU | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 |
|-------|---------|------------|---------------|---------------|---------------|---------------|---------------|
|   All-Token   | 78.65 | 76.64 | 87.97% | 85.68% | 82.65% | 74.39% | 44.74% |
|   Text Cls Token   | 77.29 | 74.79 | 86.50% | 84.16% | 81.11% | 72.84% | 44.30% |
|   Use-Text-token   | 79.44 | 77.49 | 88.90% | 86.58% | 83.25% | 75.24% | 45.24% |

Multi-Modal Token, is running

## Supp: Query Num
| Epoch | mIoU | oIoU | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 |
|-------|---------|------------|---------------|---------------|---------------|---------------|---------------|
|   N=4   | 77.76 | 75.29 | 86.75% | 84.46% | 80.96% | 72.94% | 43.72% |
|   N=8   | 77.64 | 74.83 | 86.90% | 84.76% | 81.25% | 72.76% | 44.39% |
|   N=16   | 78.05 | 75.51 | 86.93% | 84.59% | 81.24% | 73.05% | 43.89% |
|   N=32   | 79.44 | 77.49 | 88.90% | 86.58% | 83.25% | 75.24% | 45.24% |
|   N=4, v2_ln   | 77.57 | 74.19 | 86.58% | 84.08% | 80.68% | 72.45% | 43.90% |
N=24， is running

## Supp: Gate Module
| Epoch | mIoU | oIoU | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 |
|-------|---------|------------|---------------|---------------|---------------|---------------|---------------|
|   WO-Gate   | 78.75 | 76.74 | 88.07% | 85.81% | 82.45% | 74.23% | 45.55% |
| Full   | 79.44 | 77.49 | 88.90% | 86.58% | 83.25% | 75.24% | 45.24% |

## Supp: Different SAM2
| Epoch | mIoU | oIoU | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 |
|-------|---------|------------|---------------|---------------|---------------|---------------|---------------|
| SAM2-S   | 77.67 | 75.63 | 87.10% | 84.69% | 81.26% | 72.23% | 42.39% |
| SAM2-B   | 78.39 | 76.87 | 87.45% | 83.79% | 82.03% | 69.90% | 43.28% |
| SAM2-L   | 79.44 | 77.49 | 88.90% | 86.58% | 83.25% | 75.24% | 45.24% |