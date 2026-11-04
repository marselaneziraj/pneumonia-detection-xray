## split_table

| Split | NORMAL images (patients) | PNEUMONIA images (patients) | Total images | Distinct patients |
|---|---|---|---|---|
| train | 1080 (1079) | 2720 (1178) | 3800 | 2257 |
| val | 269 (269) | 706 (295) | 975 | 564 |
| test | 234 (234) | 390 (202) | 624 | 436 |

Pooled official train+val re-split by patient with `StratifiedGroupKFold` (seed 42, val fraction 0.2); official test untouched. Patient ids shared between the test folder and train/val: 170; train/val images dropped to keep the test patients unseen: 457. Unparsed file names: 144.

Metadata-only shortcut probe (logistic regression, val AUROC; 0.5 = no signal): image size only **0.9188**, border intensity only **0.7716**, both **0.9392**.

## results_table

| Model | Threshold | Recall [95% CI] | Specificity [95% CI] | False alarms (FP) | Missed (FN) | PPV | NPV | F2 | Bal. acc. | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *always PNEUMONIA* (baseline) | constant | 1.000 | 0.000 | 234 | 0 | 0.625 | — | 0.893 | 0.500 | 0.500 | 0.625 |
| *always NORMAL* (baseline) | constant | 0.000 | 1.000 | 0 | 390 | — | 0.375 | 0.000 | 0.500 | 0.500 | 0.625 |
| *random score* (baseline) | val-tuned (recall>=0.98): 0.018 | 0.982 | 0.030 | 227 | 7 | 0.628 | 0.500 | 0.882 | 0.506 | 0.482 | 0.608 |
| *random score* (baseline) | default 0.5: 0.500 | 0.485 | 0.491 | 119 | 201 | 0.614 | 0.364 | 0.506 | 0.488 | 0.482 | 0.608 |
| **densenet121** | val-tuned (recall>=0.98): 0.273 | 0.995 [0.987, 1.000] | 0.765 [0.709, 0.816] | 55 | 2 | 0.876 | 0.989 | 0.969 | 0.880 | 0.972 | 0.974 |
| **densenet121** | default 0.5: 0.500 | 0.992 [0.982, 1.000] | 0.812 [0.761, 0.859] | 44 | 3 | 0.898 | 0.984 | 0.972 | 0.902 | 0.972 | 0.974 |
| **resnet18** | val-tuned (recall>=0.98): 0.528 | 0.995 [0.987, 1.000] | 0.684 [0.624, 0.744] | 74 | 2 | 0.840 | 0.988 | 0.959 | 0.839 | 0.959 | 0.963 |
| **resnet18** | default 0.5: 0.500 | 0.995 [0.987, 1.000] | 0.675 [0.615, 0.735] | 76 | 2 | 0.836 | 0.988 | 0.958 | 0.835 | 0.959 | 0.963 |
| **resnet18_scratch** | val-tuned (recall>=0.98): 0.956 | 0.995 [0.987, 1.000] | 0.483 [0.419, 0.551] | 121 | 2 | 0.762 | 0.983 | 0.938 | 0.739 | 0.930 | 0.949 |
| **resnet18_scratch** | default 0.5: 0.500 | 1.000 [1.000, 1.000] | 0.150 [0.107, 0.197] | 199 | 0 | 0.662 | 1.000 | 0.907 | 0.575 | 0.930 | 0.949 |

Test split: 624 images (390 PNEUMONIA, 234 NORMAL). NPV is undefined (—) when the model never predicts NORMAL.

## val_table

| Model | Threshold | Recall [95% CI] | Specificity [95% CI] | False alarms (FP) | Missed (FN) | PPV | NPV | F2 | Bal. acc. | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *always PNEUMONIA* (baseline) | constant | 1.000 | 0.000 | 269 | 0 | 0.724 | — | 0.929 | 0.500 | 0.500 | 0.724 |
| *always NORMAL* (baseline) | constant | 0.000 | 1.000 | 0 | 706 | — | 0.276 | 0.000 | 0.500 | 0.500 | 0.724 |
| *random score* (baseline) | val-tuned (recall>=0.98): 0.018 | 0.980 | 0.011 | 266 | 14 | 0.722 | 0.176 | 0.915 | 0.496 | 0.509 | 0.741 |
| *random score* (baseline) | default 0.5: 0.500 | 0.501 | 0.517 | 130 | 352 | 0.731 | 0.283 | 0.535 | 0.509 | 0.509 | 0.741 |
| **densenet121** | val-tuned (recall>=0.98): 0.273 | 0.980 | 0.996 | 1 | 14 | 0.999 | 0.950 | 0.984 | 0.988 | 0.998 | 0.999 |
| **densenet121** | default 0.5: 0.500 | 0.965 | 0.996 | 1 | 25 | 0.999 | 0.915 | 0.971 | 0.980 | 0.998 | 0.999 |
| **resnet18** | val-tuned (recall>=0.98): 0.528 | 0.980 | 0.989 | 3 | 14 | 0.996 | 0.950 | 0.983 | 0.985 | 0.999 | 1.000 |
| **resnet18** | default 0.5: 0.500 | 0.983 | 0.989 | 3 | 12 | 0.996 | 0.957 | 0.986 | 0.986 | 0.999 | 1.000 |
| **resnet18_scratch** | val-tuned (recall>=0.98): 0.956 | 0.980 | 0.963 | 10 | 14 | 0.986 | 0.949 | 0.981 | 0.971 | 0.994 | 0.998 |
| **resnet18_scratch** | default 0.5: 0.500 | 0.997 | 0.714 | 77 | 2 | 0.901 | 0.990 | 0.976 | 0.855 | 0.994 | 0.998 |

Val split: 975 images (706 PNEUMONIA, 269 NORMAL). NPV is undefined (—) when the model never predicts NORMAL.

## sensitivity_table

| Recall target (val) | Threshold | Val recall | Val specificity | Test recall | Test specificity | Test FP | Test FN |
|---|---|---|---|---|---|---|---|
| 0.95 | 0.807 | 0.950 | 1.000 | 0.969 | 0.850 | 35 | 12 |
| 0.98 | 0.273 | 0.980 | 0.996 | 0.995 | 0.765 | 55 | 2 |
| 0.99 | 0.012 | 0.990 | 0.959 | 0.997 | 0.551 | 105 | 1 |

## headline

**densenet121**, threshold 0.273 chosen on val for recall >= 0.98: test recall **0.995** [0.987, 1.000] at specificity **0.765** [0.709, 0.816] - missed **2 of 390** pneumonia cases, **55 false alarms** on 234 normal films. The always-PNEUMONIA baseline scores recall 1.000 at specificity 0.000 (234 false alarms).

## resume

> Fine-tuned an ImageNet-pretrained DenseNet121 to detect pneumonia in 5.8k chest radiographs under patient-grouped splits, tuning the decision threshold for 99.5% recall at 76.5% specificity on the held-out test set against an always-positive baseline (recall 100%, specificity 0%), and used Grad-CAM to audit whether the model attends to lung fields rather than acquisition artefacts.

## error_analysis

- False negatives: **2 of 390** test pneumonias. Subtypes among misses: {'bacteria': 1, 'virus': 1} (all positives: {'bacteria': 242, 'virus': 148}).
- Miss rate by subtype: bacteria 0.004, virus 0.007. Viral share among misses 0.500 vs 0.379 among all positives.
- Exposure: mean pixel intensity of misses 0.493 vs 0.467 for true positives; contrast (pixel std) 0.216 vs 0.219.
- Lowest-scored misses (the confidently wrong ones): [0.0016, 0.1868]. Full list: `results/false_negatives.csv`.

## versions

Generated 2026-11-04T21:56:50Z (git a4bb440), seed 42; python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0+cu130, numpy 2.4.6, scikit-learn 1.6.1; GPU Tesla T4.

## gradcam_audit

| Case type | Maps | Auto: off-anatomy? | Auto: empty | By eye: lung | By eye: outside | By eye: mixed | By eye: empty | Not reviewed |
|---|---|---|---|---|---|---|---|---|
| True positives | 3 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| False negatives | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 0 |
| False positives | 3 | 0 | 0 | 1 | 0 | 2 | 0 | 0 |
| True negatives | 3 | 0 | 3 | 0 | 0 | 0 | 3 | 0 |
| **All** | 11 | 2 | 3 | 2 | 3 | 3 | 3 | 0 |

Auto-flag off-anatomy? = CAM peak in the outer 12% band, or >60% of CAM mass there (a spatially uniform map puts 42% there); empty = no positive PNEUMONIA evidence anywhere (expected for confident negatives). Mean border mass over the sampled maps: 0.422. Manual columns come from `results/gradcam_audit.csv`, filled by eye.