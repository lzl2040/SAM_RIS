# Ablation Study for AAAI
full epoch, epoch=40, img_size=480, sam2-large, all data, test on refcoco
## Components Study
| Epoch | mIoU | oIoU | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 |
|-------|---------|------------|---------------|---------------|---------------|---------------|---------------|
| WO-Adapter     |     |        |          |          |          |         |         |
|   WO-CNN   | 79.44 | 77.49 | 88.90% | 86.58% | 83.25% | 75.24% | 45.24% |
| WO-BLIP2 (replace it with CLIP)     |     |        |          |          |          |         |         |
|   Full   | 79.10 | 76.63 | 88.40% | 86.35% | 83.36% | 75.05% | 45.83% |
## Complexity and Effectiveness
| Method | Params | Flops | mIoU | oIoU |
|-------|---------|------------|---------------|---------------|


## Gate Module
| Epoch | mIoU | oIoU | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 |
|-------|---------|------------|---------------|---------------|---------------|---------------|---------------|
| WO-Gate     |     |        |          |          |          |         |         |
| W-Gate    |     |        |          |          |          |         |         |