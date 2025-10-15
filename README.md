# ML Challenge 2025 — Smart Product Pricing

**Multimodal solution for per‑product price regression (text + image + engineered numeric features)**

---

## 1. Quick summary

A compact, reproducible pipeline that fuses product `catalog_content`, product images and simple engineered numeric signals to predict per‑product price (trained/predicted in `log1p` space). Key ideas:

* Patch-aware Transformer fusion that attends image *patch tokens* together with pooled text and numeric tokens.
* Shared encoders (image + text) with auxiliary heads (image-only, text+numeric) to stabilize and regularize learning.
* Model predicts a Gaussian distribution (`mu`, `logvar`) on log-price for uncertainty-aware predictions and principled stacking.

---

## 2. What’s inside

* `train.py` — main training script (cross‑validation, OOF, optional ridge stacker).
* `data_preprocessing.py` — (expected) helpers for cleaning & extracting numeric features such as `amount_per_unit_log1p`.
* `config.yaml` — configuration for model/hyperparameters & paths.
* Model components: `ImageEncoder` (timm ViT backbone), `TextEncoder` (AutoModel), fusion Transformer, auxiliary heads.

---

## 3. Why this works (short)

* Text provides brand/spec details; images provide packaging/visual cues; numeric features capture parsed quantity/unit information that strongly correlates with per‑unit price.
* Training in `log1p` reduces heavy‑tail effects; Gaussian NLL produces calibrated variance estimates used by the stacker (`1/var` as a feature).
* Shared encoders + auxiliary heads give stronger gradients and better OOF generalization than training the fusion alone.

---

## 4. Model & training highlights

* **Inputs:** tokenized `catalog_content`, image tensor (ViT preprocessing), engineered numeric vector (6 dims by default).
* **Fusion:** optional learnable `fuse_token`, pooled text token, image patches or pooled image vector, numeric token → Transformer encoder → two readout heads (`mu`, `raw_logvar`) → stable `logvar` via `softplus` + floor.
* **Loss:** weighted sum of Gaussian NLL (on log1p targets), surrogate SMAPE (on raw prices) and a relative-MAE term, plus a small penalty to prevent variance collapse.
* **Optimization:** AdamW over *unique* parameters with named groups (`head`, `image`, `text`) so encoder LRs can be adjusted separately. Optional encoder freeze + gradual unfreeze with low‑LR fine‑tuning.
* **Practical:** mixed precision (AMP), gradient accumulation, gradient clipping, cosine LR schedule with warmup.

---

## 5. Getting started — quick run

1. Create a virtual environment and install requirements (example):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Edit `config.yaml` to point to your `data_csv`, `images_dir` and desired hyperparameters.

3. Run training (example):

```bash
python train.py --config path/to/config.yaml
```

Outputs (by default) go to `out_dir` configured in `config.yaml` and include per-fold CSVs, best checkpoints and OOF predictions.

---

## 6. Important config knobs (most impactful)

* `text_model` — transformer checkpoint for text (default: `sentence-transformers/all-mpnet-base-v2`).
* `image_backbone` — timm ViT (default: `vit_base_patch16_224`).
* `use_patch_tokens` — if `true`, image encoder returns patch tokens (enables patch-aware fusion).
* `encoder_freeze_initial_epochs` / `encoder_unfreeze_epochs` — freeze then gradually unfreeze encoders.
* `lr`, `image_lr_scale`, `text_lr_scale` — base LR and encoder scaling.
* `loss_lambda_nll`, `loss_lambda_smape`, `loss_lambda_rel` — control loss composition.

---

## 7. Typical outputs & evaluation

* Per-fold validation CSVs saved to `out_dir` (e.g. `fold1_val_results_main.csv`).
* OOF predictions: `oof_predictions_main_all_folds.csv` (and aux variants if enabled).
* (Optional) `oof_predictions_stacked_all_folds.csv` and `stacker_ridge.joblib` when `train_stacker=true`.
* Report OOF SMAPE as the main metric for leaderboard comparison.

---

## 8. Results (placeholder)

* **OOF SMAPE (main):** `45.1%`
* **Best fold example:** `smape=45.4%`

---