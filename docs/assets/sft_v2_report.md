### SFT (sft_v2)

![loss](assets/sft_v2_loss.png)

![health](assets/sft_v2_health.png)

![data](assets/sft_v2_data.png)

| epoch | step | training loss | held-out loss | gap |
|---|---|---|---|---|
| 1 | 88 | 0.4088 | 0.4412 | +0.0324 |
| 2 | 176 | 0.3861 | 0.4042 | +0.0182 |
| 3 | 264 | 0.3476 | 0.3985 | +0.0510 |

Held-out loss is measured on 86 examples from tasks absent from training; its standard error is about 0.013, so a gap below 0.037 is not distinguishable from zero.
