# Mechanistic Interpretability of a Vision Transformer Chest X-ray Classifier

A pilot study on whether a chest X-ray ViT decides from **pathology** or from a **shortcut**, answered causally at the level of internal components.

CSC 787, AI in Medical Imaging Informatics (University of South Dakota). Author: Shou-Tzu Han.

## Overview

A ViT-Base/16 is fine-tuned on the RSNA Pneumonia Detection Challenge with a **controlled synthetic marker** (a white square in one known patch token) correlated with the positive class. Because the confounder location is known in advance, three mechanistic probes can be applied with clean ground truth:

1. **Attention attribution** — how much each head attends to the marker patch (correlational).
2. **Activation patching** — copy a head's clean activation into the marked run and measure the change in prediction (causal).
3. **Component ablation** — zero the most causal heads and test whether pathology-based prediction returns (necessity).

Shortcut reliance is quantified with a **three-way test**: marker-consistent, clean (no marker), and marker-flipped.

## Key findings

- In-distribution (marker present): **0.999 AUC**, the model learns the task.
- Clean images (marker removed): recall collapses **0.96 to 0.07**, F1 drops to 0.13.
- Marker-flipped: AUC inverts to **0.19** (below chance).
- Attention attribution and activation patching converge on a single early head, **layer 1, head 7**, as the dominant marker carrier.
- Ablating the top-5 causal heads barely moves clean accuracy (0.790 to 0.786): an honest negative result, the reliance is distributed at the decision threshold.

## Repository structure

```
cxr_vit_shortcut_pilot/
├── src/
│   └── cxr_interp.py               # full pipeline: data, training, 3 probes, figures
├── scripts/
│   ├── run_cxr_full.sh             # SLURM job for the full GPU run (final results)
│   ├── run_cxr_vit.sh              # variant run script
│   └── run_cxr.sh                  # earlier run script
├── rsna/
│   ├── make_examples.py            # generates example / marker-injected figures
│   ├── stage_2_train_labels.csv
│   ├── stage_2_detailed_class_info.csv
│   └── stage_2_sample_submission.csv
├── out/
│   ├── out_full/                   # results: fig_*.png, results.json
│   └── cxr_examples/               # example images used in the slides
├── README.md
└── .gitignore
```

The RSNA DICOM image folders, the trained checkpoint (`vit_cxr.pt`), SLURM logs, local `data/`, and IDE files are not tracked in git; see the `.gitignore` note below.

## Dataset

[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge): 25,684 frontal chest radiographs, 1024 x 1024 8-bit grayscale DICOM, radiologist-annotated. Collapsed to a binary pneumonia task (~22% positive).

Accept the competition rules on Kaggle, then download and unzip into `rsna/`:

```bash
kaggle competitions download -c rsna-pneumonia-detection-challenge
```

Expected contents of `rsna/`:

```
rsna/
├── stage_2_train_images/           # *.dcm
├── stage_2_test_images/            # *.dcm
├── stage_2_train_labels.csv
├── stage_2_detailed_class_info.csv
└── stage_2_sample_submission.csv
```

## Setup

Tested with Python 3.11, PyTorch 2.5.1 + CUDA 12.1, on an NVIDIA Tesla V100.

```bash
conda create -n cxr python=3.11 -y
conda activate cxr
pip install --no-cache-dir \
  "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" \
  timm pydicom scikit-learn matplotlib pandas numpy pillow \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

## Usage

Full run (train + all probes + figures):

```bash
python src/cxr_interp.py --data_dir ./rsna --out_dir ./out/out_full --epochs 5 --batch 32
```

Quick smoke test on a subset:

```bash
python src/cxr_interp.py --data_dir ./rsna --out_dir ./out/out_smoke --subset 2000
```

Analysis only, reusing a saved checkpoint:

```bash
python src/cxr_interp.py --data_dir ./rsna --out_dir ./out/out_full --skip_train
```

On HPC, submit the SLURM job instead (activates the `cxr` env and runs the full pipeline on one GPU):

```bash
sbatch scripts/run_cxr_full.sh
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `./rsna` | Path to the RSNA folder (`stage_2_train_images/`, `stage_2_train_labels.csv`) |
| `--out_dir` | `./out` | Output directory for figures, `results.json`, and `vit_cxr.pt` |
| `--epochs` | `5` | Training epochs |
| `--lr` | `2e-5` | Learning rate (AdamW) |
| `--batch` | `32` | Batch size |
| `--subset` | `None` | Use only N images (quick test) |
| `--skip_train` | off | Skip training, load `vit_cxr.pt`, run analysis only |

### Example figures

The class examples and the original vs. marker-injected pair are produced by `rsna/make_examples.py` (written to `out/cxr_examples/`):

```bash
cd rsna
python make_examples.py
```

## Outputs

Written to `--out_dir` (for the full run, `out/out_full/`):

- `fig_confusion_clean.png`, `fig_roc_clean.png` — clean-test classification
- `fig_attention_attribution.png`, `fig_patching_causal.png` — per-head attention and causal effect
- `fig_ablation_recovery.png` — clean accuracy before/after ablation
- `results.json` — all metrics across the three conditions
- `vit_cxr.pt` — best-validation-AUC checkpoint

## Results

In-distribution (marker-consistent test set):

| Metric | Value |
|---|---|
| Accuracy | 0.988 |
| Precision | 0.983 |
| Recall | 0.961 |
| F1-score | 0.971 |
| Specificity | 0.995 |
| ROC-AUC | 0.999 |

Three-way shortcut test:

| Condition | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Marker-consistent | 0.988 | 0.961 | 0.971 | 0.999 |
| Clean (no marker) | 0.790 | 0.068 | 0.125 | 0.868 |
| Marker-flipped | 0.410 | 0.068 | 0.048 | 0.191 |

Top-5 causal heads (activation patching): (1,7), (8,3), (5,1), (11,11), (8,11).

Runtime: 5 epochs in about 21 minutes on a single V100.

## .gitignore

Suggested entries so the repo stays small (keep the `out_full` figures and `results.json`, which the README references):

```
rsna/stage_2_train_images/
rsna/stage_2_test_images/
data/
logs/
*.pt
.idea/
*.iml
.DS_Store
__pycache__/
*.pyc
```

## References

1. DeGrave, Janizek, Lee. AI for radiographic COVID-19 detection selects shortcuts over signal. *Nature Machine Intelligence* 3:610-619, 2021.
2. Dosovitskiy et al. An image is worth 16x16 words. ICLR 2021.
3. Vaswani et al. Attention is all you need. NeurIPS 2017.
4. Selvaraju et al. Grad-CAM. ICCV 2017.
5. Tang, Zhong, Shah, Liu. CXR-LanIC. arXiv:2510.21464, 2026.
6. Cooke et al. RoentMod. arXiv:2509.08640, 2025.

## Author

Shou-Tzu Han, Department of Computer Science, University of South Dakota.
CSC 787, AI in Medical Imaging Informatics, Summer 2026.

## License

Released for academic use. Add a `LICENSE` file (for example, MIT) if you want to set explicit terms.