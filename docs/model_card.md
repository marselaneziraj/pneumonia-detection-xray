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
_Run `pneumonia prepare` to generate the split table._
<!-- /AUTO:split_table -->

## Operating point and why

A missed pneumonia (FN) sends a sick child home; a false alarm costs a second read. The threshold is therefore the **highest-specificity threshold whose val recall is ≥ 0.98** (`selection.target_recall`). It is chosen on the patient-grouped val split, frozen, and applied once to test. The loss is not distorted to buy recall.

<!-- AUTO:sensitivity_table -->
_Pending._
<!-- /AUTO:sensitivity_table -->

## Measured performance (official test split, scored once)

<!-- AUTO:headline -->
_Headline pending: no trained checkpoint was evaluated yet._
<!-- /AUTO:headline -->

<!-- AUTO:results_table -->
_Pending._
<!-- /AUTO:results_table -->

## Explanation audit

<!-- AUTO:gradcam_audit -->
_Pending._
<!-- /AUTO:gradcam_audit -->

## Limitations

* **Paediatric and single-centre.** Nothing here transfers to adult chest films, or to other hospitals or scanners, without revalidation.
* **Labels** come from the original curation, not a fresh consensus read; some label noise is likely.
* **Small, same-centre test set (~624 images).** The scores are an optimistic upper bound, so quote the CIs.
* **Acquisition shortcuts.** The classes came through different pipelines (see the shortcut probe), so the model may partly key on non-anatomical cues; see the audit.
* **Grad-CAM** is a coarse saliency heuristic, not a segmentation and not evidence of clinical reasoning.
* **Probabilities are not calibrated** (no temperature scaling); the threshold is not a calibrated risk.

## Ethical considerations

A recall-first threshold moves workload onto radiologists through false alarms; the false-alarm count is reported for this reason. A model that looks good on one centre's children could fail silently elsewhere. This model must not be deployed.

<!-- AUTO:versions -->
_Pending._
<!-- /AUTO:versions -->
