## Exp Results
SAM2-Small, Instruct BIP2 Vicun7B, Trained on RefCOCO,Epoch=22, Batch=16, 50G
```shell
25-03-02 15:51:04.932 - INFO: Mean IoU is 73.33
25-03-02 15:51:04.933 - INFO: Overall IoU is 69.54
25-03-02 15:51:04.933 - INFO: Precision@0.5=82.24%
25-03-02 15:51:04.934 - INFO: Precision@0.6=78.80%
25-03-02 15:51:04.934 - INFO: Precision@0.7=74.28%
25-03-02 15:51:04.934 - INFO: Precision@0.8=65.08%
25-03-02 15:51:04.935 - INFO: Precision@0.9=36.56%
```
Epoch=35，从Epoch=22处中断开始，之后都差不多
```shell
Mean IoU is 73.90
Overall IoU is 70.33
Precision@0.5=83.01%
Precision@0.6=79.65%
Precision@0.7=75.24%
Precision@0.8=66.23%
Precision@0.9=37.59%
```

SAM2-S, 结合了CNN token来微调SAM2：epoch=22
```shell
Mean IoU is 73.96
Overall IoU is 70.33
Precision@0.5=83.01%
Precision@0.6=79.14%
Precision@0.7=74.49%
Precision@0.8=65.05%
Precision@0.9=37.70%
```
E=26:
```shell
Mean IoU is 74.38
Overall IoU is 71.08
Precision@0.5=83.51%
Precision@0.6=80.11%
Precision@0.7=75.49%
Precision@0.8=66.43%
Precision@0.9=37.96%
```

所有数据混合训练：
```shell
E=4:
Mean IoU is 73.18
Overall IoU is 70.50
Precision@0.5=82.12%
Precision@0.6=78.13%
Precision@0.7=72.48%
Precision@0.8=62.77%
Precision@0.9=34.98%
E=6:
Mean IoU is 74.16
Overall IoU is 71.55
Precision@0.5=83.35%
Precision@0.6=79.52%
Precision@0.7=74.74%
Precision@0.8=65.05%
Precision@0.9=35.88%
E=8:
Mean IoU is 75.16
Overall IoU is 72.55
Precision@0.5=84.15%
Precision@0.6=80.52%
Precision@0.7=75.88%
Precision@0.8=66.73%
Precision@0.9=38.21%
E=10:
Mean IoU is 75.05
Overall IoU is 72.98
Precision@0.5=84.50%
Precision@0.6=81.35%
Precision@0.7=76.31%
Precision@0.8=66.37%
Precision@0.9=37.11%
E=12:
Mean IoU is 75.89
Overall IoU is 73.59
Precision@0.5=85.31%
Precision@0.6=82.24%
Precision@0.7=78.00%
Precision@0.8=68.68%
Precision@0.9=39.65%
E=14:
Mean IoU is 75.88
Overall IoU is 73.60
Precision@0.5=84.95%
Precision@0.6=81.82%
Precision@0.7=77.09%
Precision@0.8=67.79%
Precision@0.9=39.11%
E=16:
Mean IoU is 76.43
Overall IoU is 74.41
Precision@0.5=85.95%
Precision@0.6=82.97%
Precision@0.7=78.53%
Precision@0.8=69.44%
Precision@0.9=40.14%
E=18:
Mean IoU is 76.64
Overall IoU is 74.68
Precision@0.5=86.04%
Precision@0.6=83.14%
Precision@0.7=78.68%
Precision@0.8=69.50%
Precision@0.9=40.71%
E=20:
Mean IoU is 76.77
Overall IoU is 74.71
Precision@0.5=86.20%
Precision@0.6=83.18%
Precision@0.7=78.92%
Precision@0.8=70.36%
Precision@0.9=40.54%
E=22:
Mean IoU is 76.97
Overall IoU is 74.88
Precision@0.5=86.52%
Precision@0.6=83.58%
Precision@0.7=79.47%
Precision@0.8=70.36%
Precision@0.9=40.99%
E=26:
Mean IoU is 77.39
Overall IoU is 75.11
Precision@0.5=86.77%
Precision@0.6=84.12%
Precision@0.7=80.28%
Precision@0.8=71.37%
Precision@0.9=41.54%
E=28:
Mean IoU is 77.47
Overall IoU is 75.51
Precision@0.5=86.71%
Precision@0.6=83.91%
Precision@0.7=80.32%
Precision@0.8=71.70%
Precision@0.9=42.12%
E=30:
Mean IoU is 78.00
Overall IoU is 76.37
Precision@0.5=87.49%
Precision@0.6=84.96%
Precision@0.7=81.16%
Precision@0.8=72.29%
Precision@0.9=42.62%
E=32:
Mean IoU is 77.61
Overall IoU is 75.82
Precision@0.5=87.05%
Precision@0.6=84.47%
Precision@0.7=80.71%
Precision@0.8=72.09%
Precision@0.9=42.79%
```

SAM2-B, 结合了CNN token来微调SAM2，Epoch=22：
```shell
Mean IoU is 75.05
Overall IoU is 71.85
Precision@0.5=83.77%
Precision@0.6=80.92%
Precision@0.7=76.93%
Precision@0.8=68.26%
Precision@0.9=40.47%
```
Epoch=26:
```shell
Mean IoU is 75.03
Overall IoU is 72.11
Precision@0.5=83.59%
Precision@0.6=81.00%
Precision@0.7=77.17%
Precision@0.8=68.62%
Precision@0.9=40.28%
```
E=30:
```shell
Mean IoU is 75.24
Overall IoU is 71.95
Precision@0.5=83.88%
Precision@0.6=81.16%
Precision@0.7=77.27%
Precision@0.8=69.02%
Precision@0.9=40.83%
```

E=34:
```shell
Mean IoU is 75.68
Overall IoU is 72.34
Precision@0.5=84.43%
Precision@0.6=81.90%
Precision@0.7=78.12%
Precision@0.8=69.69%
Precision@0.9=41.81%
```
E=40:
```shell
Mean IoU is 76.04
Overall IoU is 73.01
Precision@0.5=84.96%
Precision@0.6=82.58%
Precision@0.7=78.65%
Precision@0.8=70.36%
Precision@0.9=41.69%
```
