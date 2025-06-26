# Ablation Study for AAAI
full epoch, epoch=40, img_size=480, sam2-large, all data
## Components Study
| Epoch | mIoU | oIoU | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 |
|-------|---------|------------|---------------|---------------|---------------|---------------|---------------|
| WO-Adapter     |     |        |          |          |          |         |         |
| WO-CNN     |     |        |          |          |          |         |         |
| WO-BLIP2 (replace it with CLIP)     |     |        |          |          |          |         |         |
| Full     |     |        |          |          |          |         |         |
## Complexity and Effectiveness
| Method | Params | Flops | mIoU | oIoU |
|-------|---------|------------|---------------|---------------|


## Gate Module
| Epoch | mIoU | oIoU | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 |
|-------|---------|------------|---------------|---------------|---------------|---------------|---------------|
| WO-Gate     |     |        |          |          |          |         |         |
| W-Gate    |     |        |          |          |          |         |         |