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