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