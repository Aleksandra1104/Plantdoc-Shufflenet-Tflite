# Plantdoc-Shufflenet-Tflite

On-device plant **leaf disease classifier**: a ShuffleNetV2 model pretrained on
**PlantVillage**, fine-tuned on the real-world **PlantDoc** dataset, and exported to
**TFLite** for mobile inference. Identifies **27** leaf/disease classes from a single photo.

> **Held-out test accuracy: ~59%.** That's near the practical ceiling for PlantDoc.
> The limiting factor is data, not the model.

---

## Highlights

- **Mobile-first.** ShuffleNetV2 x1.0 (~1.3M parameters) keeps the exported model small
  and fast enough for real-time, on-device inference.
- **Honest evaluation.** Three-way split — the official PlantDoc test set is held out and
  scored exactly once, so the headline number isn't inflated by tuning against it.
- **Transfer learning that actually helps.** PlantVillage backbone pretraining roughly
  doubles accuracy versus fine-tuning from ImageNet alone on this dataset.
- **Reproducible.** Seeded data split and augmentation; a single Kaggle notebook runs the
  whole pipeline end to end and emits the `.tflite` model plus a `classes.json` label map.

---

## Results

| Metric | Value |
|---|---|
| Held-out **test** accuracy | **59.3%** |
| Best **validation** accuracy | 64.5% |
| Macro-avg F1 | 0.57 |
| Classes | 27 |

**Strong classes** (visually distinctive): grape leaf, strawberry leaf, corn rust leaf,
grape black rot, tomato late blight.

**Weak classes** (visually similar diseases): the tomato cluster (bacterial spot, mosaic
virus, mold) and the two corn classes, which the model tends to confuse with each other.

A full per-class breakdown and analysis lives in the training report.

---

## How it works

```
PlantVillage (lab images)        PlantDoc (real-world images)
        │  pretrain backbone              │  fine-tune + evaluate
        ▼                                 ▼
  ShuffleNetV2 backbone  ──────►  two-phase training  ──►  best checkpoint  ──►  TFLite
                                  (frozen-head warmup,         (early stopping        + classes.json
                                   then full fine-tune)         on validation)
```

1. **Pretrain** a throwaway model on PlantVillage, then keep only its feature-extracting
   backbone (the classifier head is discarded).
2. **Warm up** on PlantDoc with the backbone frozen — only the new classifier head learns.
3. **Fine-tune** the whole network at a lower learning rate, selecting the best epoch by
   validation accuracy with early stopping.
4. **Evaluate once** on the held-out PlantDoc test set, then **export** to TFLite.

Class imbalance is handled with a weighted sampler; training images get heavy augmentation
(crops, flips, color jitter, rotation, blur, perspective, random erasing).

---

## Datasets

| Dataset | Role | Character |
|---|---|---|
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) | train + validation + test | In-the-wild phone photos — varied backgrounds, lighting, angles |
| [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) | backbone pretraining only | Lab photos — single leaf on a uniform background |

PlantVillage is used **only** for pretraining. Its clean lab backgrounds don't resemble
real app photos, so the model is never validated or tested on it — doing so would produce a
misleadingly high number.

---

## Getting started

This is a single self-documenting Kaggle notebook; every cell opens with a plain-English
explanation of what it does.

1. Open the notebook on [Kaggle](https://www.kaggle.com/) and set **Settings → Accelerator → GPU T4 x1**.
2. (Optional but recommended) Add the **PlantVillage** dataset as an input and set
   `PLANTVILLAGE_DIR` in Step 0 to its `color` folder. To skip pretraining, set
   `USE_PLANTVILLAGE_PRETRAIN = False`.
3. **Run all.** PlantDoc is cloned automatically.

**Outputs** (in `/kaggle/working`):
- `plant_disease_model.tflite` — the mobile model
- `classes.json` — label names in the model's output order
- `training_curves.png` — accuracy/loss curves

Drop the two model files into your app (e.g. `assets/model/`). Step 12 prints the expected
input layout (NCHW vs NHWC) — your app's image preprocessing must match it.

---

## Why only ~59%?

This is roughly the intrinsic ceiling for PlantDoc with a mobile-sized model, not a tuning
failure. Published baselines on PlantDoc sit in the ~30–60% range. The main reasons:

- **Little data** — ~1,900 training images across 27 classes (some classes have ~35).
- **In-the-wild difficulty** — cluttered backgrounds, varied lighting and angles, occlusion.
- **Fine-grained classes** — many diseases of the same crop look near-identical in a phone photo.
- **Domain gap** — PlantVillage pretraining helps, but its lab images can't fully stand in
  for real field photos.

The biggest lever from here is **more and better realistic training data** — which is
exactly what the roadmap targets.

---

## Roadmap

**Next step: synthetic data to break the data ceiling.** The single biggest constraint is
the small, imbalanced set of real-world images. Plan:

- **Generate synthetic diseased leaves with [LeafGAN](https://github.com/IyatomiLab/LeafGAN).**
  LeafGAN is an attention-guided image-to-image translation model (built on CycleGAN) that
  turns healthy leaf photos into realistic diseased ones. Its label-free leaf-segmentation
  module (LFLSeg) focuses the transformation on the leaf itself rather than the background,
  which avoids the "bleeding" artifacts that plague vanilla CycleGAN. In the original paper,
  LeafGAN augmentation improved diagnostic accuracy by ~7.4% (vs ~0.7% for CycleGAN), so it's
  a strong candidate for boosting the weak, underrepresented classes here.
- **Target the failure cases first** — generate extra examples for the tomato cluster and the
  corn classes, which dominate the current error budget.
- **Treat synthetic data as augmentation, not a replacement** — mix it with real PlantDoc
  images for training, and keep the **real, held-out PlantDoc test set untouched** so the
  reported accuracy stays honest. (Evaluating on synthetic images would defeat the purpose.)
- **Re-measure** with the same three-way protocol to confirm the gain is real and not just
  the model learning the generator's artifacts.

Other possible directions: merging genuinely indistinguishable disease classes into coarser
labels, trying a larger backbone if device constraints allow, and adding top-3 predictions
with a confidence threshold in the app to gracefully handle uncertain cases.

---

## Limitations

- Not a diagnostic tool — predictions are probabilistic and often wrong on visually similar
  diseases.
- Trained and evaluated on a small dataset; performance on crops, lighting, or devices unlike
  the training data may be lower.

---

## References

- PlantDoc — Singh et al., *PlantDoc: A Dataset for Visual Plant Disease Detection.*
- PlantVillage — Hughes & Salathé, *An open access repository of images on plant health.*
- LeafGAN — Cap et al., *LeafGAN: An Effective Data Augmentation Method for Practical Plant
  Disease Diagnosis*, IEEE T-ASE, 2020. [Paper](https://arxiv.org/abs/2002.10100) ·
  [Code](https://github.com/IyatomiLab/LeafGAN)

## License

Add a license of your choice (e.g. MIT). Note that PlantDoc, PlantVillage, and LeafGAN carry
their own licenses — check each before redistributing data or generated images.
