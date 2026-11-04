# Model card — `densenet121` pneumonia triage classifier

> **Not a medical device. Not validated for clinical use. Not a diagnostic aid.** Portfolio / research artefact.

## Model details

* ImageNet-pretrained DenseNet121 (torchvision `IMAGENET1K_V1`), classifier replaced by dropout(0.2) + a single linear logit.
* Input: frontal chest radiograph, greyscale replicated to 3 channels, 224×224, ImageNet normalisation.
* Output: P(PNEUMONIA) = sigmoid(logit). Decision: PNEUMONIA iff P ≥ the shipped threshold.
* Training: `BCEWithLogitsLoss(pos_weight = n_neg/n_pos)`; phase 1 head-only (3 epochs, lr 1e-3), phase 2 full fine-tune (≤12 epochs, AdamW, lr 1e-4, cosine), early stopping on val specificity at recall ≥ 0.98. Config: `configs/densenet121.yaml`, seed 42.

## Intended use

Research demonstration of a **recall-first triage operating point**: a binary "look at this film first" flag on **paediatric** frontal chest X-rays from the same distribution as the training data. Out of scope: adults, other centres or scanners, lateral views, any diagnostic or treatment decision, and any use without a clinician reading the film.

## Training data and provenance

Kaggle `paultimothymooney/chest-xray-pneumonia` (Kermany, Zhang, Goldbaum et al., *Cell* 2018; Mendeley Data, CC BY 4.0). The films are from Guangzhou Women and Children's Medical Center, patients aged 1–5, with labels from the original expert curation. Official train+val are pooled and re-split by patient id parsed from file names; the official test folder is held out.

<!-- AUTO:split_table -->
| Split | NORMAL images (patients) | PNEUMONIA images (patients) | Total images | Distinct patients |
|---|---|---|---|---|
| train | 1080 (1079) | 2720 (1178) | 3800 | 2257 |
| val | 269 (269) | 706 (295) | 975 | 564 |
| test | 234 (234) | 390 (202) | 624 | 436 |

Pooled official train+val re-split by patient with `StratifiedGroupKFold` (seed 42, val fraction 0.2); official test untouched. Patient ids shared between the test folder and train/val: 170; train/val images dropped to keep the test patients unseen: 457. Unparsed file names: 144.

Metadata-only shortcut probe (logistic regression, val AUROC; 0.5 = no signal): image size only **0.9188**, border intensity only **0.7716**, both **0.9392**.
<!-- /AUTO:split_table -->

## Operating point and why

A missed pneumonia (FN) sends a sick child home; a false alarm costs a second read. The threshold is therefore the **highest-specificity threshold whose val recall is ≥ 0.98** (`selection.target_recall`). It is chosen on the patient-grouped val split, frozen, and applied once to test. The loss is not distorted to buy recall.

<!-- AUTO:sensitivity_table -->
| Recall target (val) | Threshold | Val recall | Val specificity | Test recall | Test specificity | Test FP | Test FN |
|---|---|---|---|---|---|---|---|
| 0.95 | 0.807 | 0.950 | 1.000 | 0.969 | 0.850 | 35 | 12 |
| 0.98 | 0.273 | 0.980 | 0.996 | 0.995 | 0.765 | 55 | 2 |
| 0.99 | 0.012 | 0.990 | 0.959 | 0.997 | 0.551 | 105 | 1 |
<!-- /AUTO:sensitivity_table -->

## Measured performance (official test split, scored once)

<!-- AUTO:headline -->
**densenet121**, threshold 0.273 chosen on val for recall >= 0.98: test recall **0.995** [0.987, 1.000] at specificity **0.765** [0.709, 0.816] - missed **2 of 390** pneumonia cases, **55 false alarms** on 234 normal films. The always-PNEUMONIA baseline scores recall 1.000 at specificity 0.000 (234 false alarms).
<!-- /AUTO:headline -->

<!-- AUTO:results_table -->
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
<!-- /AUTO:results_table -->

## Explanation audit

<!-- AUTO:gradcam_audit -->
| Case type | Maps | Auto: off-anatomy? | Auto: empty | By eye: lung | By eye: outside | By eye: mixed | By eye: empty | Not reviewed |
|---|---|---|---|---|---|---|---|---|
| True positives | 3 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| False negatives | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 0 |
| False positives | 3 | 0 | 0 | 1 | 0 | 2 | 0 | 0 |
| True negatives | 3 | 0 | 3 | 0 | 0 | 0 | 3 | 0 |
| **All** | 11 | 2 | 3 | 2 | 3 | 3 | 3 | 0 |

Auto-flag off-anatomy? = CAM peak in the outer 12% band, or >60% of CAM mass there (a spatially uniform map puts 42% there); empty = no positive PNEUMONIA evidence anywhere (expected for confident negatives). Mean border mass over the sampled maps: 0.422. Manual columns come from `results/gradcam_audit.csv`, filled by eye.
<!-- /AUTO:gradcam_audit -->

## Limitations

* **Paediatric and single-centre.** Nothing here transfers to adult chest films, or to other hospitals or scanners, without revalidation.
* **Labels** come from the original curation, not a fresh consensus read; some label noise is likely.
* **Small, same-centre test set (~624 images).** The scores are an optimistic upper bound, so quote the CIs.
* **Acquisition shortcuts, measured.** Image size alone predicts the class (val AUROC 0.919). In the Grad-CAM audit, 6 of 8 non-empty maps have substantial activation outside the lungs, and 4 hit burnt-in text or the "R" laterality letter. The model partly relies on non-anatomical cues.
* **Operating point does not transfer cleanly.** Val specificity 0.996 at the shipped threshold vs 0.765 on test; expect specificity to drop again on any new data.
* **Known issue in this run:** a patient-ID parser bug let 39 val NORMAL images share a patient with train (test unaffected; fixed in code, re-run pending). Val-based selection is slightly optimistic.
* **Grad-CAM** is a coarse saliency heuristic, not a segmentation and not evidence of clinical reasoning.
* **Probabilities are not calibrated** (no temperature scaling); the threshold is not a calibrated risk.

## Ethical considerations

A recall-first threshold moves workload onto radiologists through false alarms; the false-alarm count is reported for this reason. A model that looks good on one centre's children could fail silently elsewhere. This model must not be deployed.

<!-- AUTO:versions -->
Generated 2026-11-04T21:56:50Z (git a4bb440), seed 42; python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0+cu130, numpy 2.4.6, scikit-learn 1.6.1; GPU Tesla T4.
<!-- /AUTO:versions -->
