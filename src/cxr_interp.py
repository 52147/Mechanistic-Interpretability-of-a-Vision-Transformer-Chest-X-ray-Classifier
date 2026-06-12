"""
Mechanistic interpretability of a ViT chest X-ray classifier (4-week pilot).

Pipeline:
  1. Load RSNA Pneumonia Detection Challenge (binary: lung-opacity vs not).
  2. Inject a controlled synthetic corner marker (known location) as a shortcut.
  3. Fine-tune an ImageNet-pretrained ViT-Base/16 (transfer learning).
  4. Evaluate (accuracy, precision, recall, F1, specificity, AUC, confusion matrix, ROC).
  5. Three-way shortcut test: marker-consistent / clean / marker-flipped.
  6. Mechanistic analysis on the ViT:
       (A) attention-head attribution (CLS attention on the marker patch),
       (B) activation patching (clean -> marked) per head,  (C) component ablation.
  7. Save all figures as PNG (works headless on an HPC node).

DEPENDENCIES (install once on the login node):
  pip install --no-cache-dir torch torchvision timm pydicom scikit-learn matplotlib pandas numpy pillow

DATA (download once where you have internet, e.g. login node, using a Kaggle token):
  pip install kaggle
  kaggle competitions download -c rsna-pneumonia-detection-challenge -p ./rsna
  cd rsna && unzip -q '*.zip'
  # you need: stage_2_train_images/*.dcm  and  stage_2_train_labels.csv
  # (you must accept the competition rules on the Kaggle website first)

RUN (on a GPU compute node):
  # full run (train then analyze):
  python cxr_interp.py --data_dir ./rsna --out_dir ./out --epochs 5
  # quick smoke test on a subset:
  python cxr_interp.py --data_dir ./rsna --out_dir ./out --epochs 1 --subset 2000
  # skip training and reuse a saved checkpoint for the interpretability part:
  python cxr_interp.py --data_dir ./rsna --out_dir ./out --skip_train

NOTE: hyperparameters here are reasonable starting points, not tuned. Expect to iterate.
"""

import os, argparse, random, json, contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
import pydicom
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)

# ----------------------------- config / repro -----------------------------
IMG_SIZE = 224
PATCH = 16
GRID = IMG_SIZE // PATCH            # 14
N_PATCH_TOK = GRID * GRID           # 196
MARKER_SIZE = 16                    # top-left square; 16 aligns to exactly one patch token
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def marker_token_indices():
    """Token indices (CLS = 0) covered by a MARKER_SIZE square at the top-left corner."""
    rmax = (MARKER_SIZE + PATCH - 1) // PATCH
    idx = []
    for r in range(rmax):
        for c in range(rmax):
            idx.append(1 + r * GRID + c)   # +1 for CLS token
    return idx

MARKER_TOKENS = marker_token_indices()

# ----------------------------- data -----------------------------
def load_labels(data_dir):
    csv = os.path.join(data_dir, "stage_2_train_labels.csv")
    df = pd.read_csv(csv)
    # one row per patient; Target is consistent within a patient
    lab = df.groupby("patientId")["Target"].max().reset_index()
    return lab  # columns: patientId, Target (1 = lung opacity / positive)

def add_marker(img_chw):
    """img_chw: float tensor (3,H,W) in [0,1]. Paint a white square at the top-left."""
    img_chw = img_chw.clone()
    img_chw[:, 0:MARKER_SIZE, 0:MARKER_SIZE] = 1.0
    return img_chw

class RSNADataset(Dataset):
    """
    marker_mode:
      'train_rule' : positive -> marker w.p. 0.9, negative -> marker w.p. 0.1 (learnable shortcut)
      'consistent' : positive -> marker, negative -> none      (model using marker scores high)
      'clean'      : never add a marker                        (model must use pathology)
      'flip'       : positive -> none, negative -> marker      (marker now indicates the wrong class)
    """
    def __init__(self, labels_df, data_dir, marker_mode="clean", seed=0):
        self.df = labels_df.reset_index(drop=True)
        self.img_dir = os.path.join(data_dir, "stage_2_train_images")
        self.mode = marker_mode
        rng = np.random.RandomState(seed)
        if marker_mode == "train_rule":
            p = np.where(self.df["Target"].values == 1, 0.9, 0.1)
            self.marker_flag = (rng.rand(len(self.df)) < p).astype(int)
        elif marker_mode == "consistent":
            self.marker_flag = (self.df["Target"].values == 1).astype(int)
        elif marker_mode == "flip":
            self.marker_flag = (self.df["Target"].values == 0).astype(int)
        else:  # clean
            self.marker_flag = np.zeros(len(self.df), dtype=int)

    def __len__(self):
        return len(self.df)

    def _read_image(self, patient_id):
        path = os.path.join(self.img_dir, f"{patient_id}.dcm")
        arr = pydicom.dcmread(path).pixel_array.astype(np.float32)
        arr = arr / (arr.max() + 1e-6)                      # -> [0,1]
        pil = Image.fromarray((arr * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
        x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0)  # (H,W)
        x = x.unsqueeze(0).repeat(3, 1, 1)                  # (3,H,W), grayscale -> 3 channels
        return x

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = self._read_image(row["patientId"])              # [0,1]
        if self.marker_flag[i]:
            x = add_marker(x)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD              # ImageNet normalization
        y = int(row["Target"])
        return x, y

def make_splits(labels_df, val_frac=0.15, test_frac=0.15, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(labels_df))
    n_test = int(len(idx) * test_frac)
    n_val = int(len(idx) * val_frac)
    test_df = labels_df.iloc[idx[:n_test]].reset_index(drop=True)
    val_df = labels_df.iloc[idx[n_test:n_test + n_val]].reset_index(drop=True)
    train_df = labels_df.iloc[idx[n_test + n_val:]].reset_index(drop=True)
    return train_df, val_df, test_df

# ----------------------------- model -----------------------------
def build_model(device):
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
    return model.to(device)

def class_weights(train_df, device):
    counts = train_df["Target"].value_counts().sort_index().values.astype(np.float32)
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, device=device)

def train_model(model, train_loader, val_loader, device, epochs, lr, out_dir, cw):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=cw)
    best_auc, ckpt = -1.0, os.path.join(out_dir, "vit_cxr.pt")
    for ep in range(epochs):
        model.train(); running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            running += loss.item() * x.size(0)
        _, probs, ys = run_inference(model, val_loader, device)
        try:
            vauc = roc_auc_score(ys, probs)
        except ValueError:
            vauc = float("nan")
        print(f"[epoch {ep+1}/{epochs}] train_loss={running/len(train_loader.dataset):.4f}  val_AUC={vauc:.4f}")
        if np.isfinite(vauc) and vauc > best_auc:
            best_auc = vauc
            torch.save(model.state_dict(), ckpt)
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    return model

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval(); preds, probs, ys = [], [], []
    for x, y in loader:
        out = model(x.to(device))
        p = F.softmax(out, dim=1)[:, 1]
        preds.append(out.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
    return (np.concatenate(preds), np.concatenate(probs), np.concatenate(ys))

# ----------------------------- evaluation -----------------------------
def compute_metrics(preds, probs, ys):
    cm = confusion_matrix(ys, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy_score(ys, preds),
        "precision": precision_score(ys, preds, zero_division=0),
        "recall": recall_score(ys, preds, zero_division=0),
        "f1": f1_score(ys, preds, zero_division=0),
        "specificity": spec,
        "roc_auc": roc_auc_score(ys, probs) if len(set(ys)) > 1 else float("nan"),
    }

def save_confusion(preds, ys, title, path):
    ConfusionMatrixDisplay.from_predictions(ys, preds, cmap="Blues",
                                            display_labels=["No opacity", "Lung opacity"])
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

def save_roc(probs, ys, title, path):
    fpr, tpr, _ = roc_curve(ys, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

def three_way_test(model, test_df, data_dir, device, batch, out_dir):
    rows = {}
    for mode in ["consistent", "clean", "flip"]:
        ds = RSNADataset(test_df, data_dir, marker_mode=mode, seed=0)
        ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)
        preds, probs, ys = run_inference(model, ld, device)
        m = compute_metrics(preds, probs, ys)
        rows[mode] = m
        if mode == "clean":
            save_confusion(preds, ys, "Confusion Matrix (clean test)",
                           os.path.join(out_dir, "fig_confusion_clean.png"))
            save_roc(probs, ys, "ROC (clean test)", os.path.join(out_dir, "fig_roc_clean.png"))
    print("\n=== Three-way shortcut test ===")
    print(f"{'condition':<14}{'acc':>8}{'AUC':>8}{'F1':>8}")
    for mode, m in rows.items():
        print(f"{mode:<14}{m['accuracy']:>8.3f}{m['roc_auc']:>8.3f}{m['f1']:>8.3f}")
    return rows

# ----------------------------- mechanistic interpretability -----------------------------
def disable_fused_attn(model):
    """Force explicit attention so we can read the softmax attention matrix via attn_drop."""
    for blk in model.blocks:
        blk.attn.fused_attn = False

def get_paired_batch(test_df, data_dir, device, n=32):
    """Take n positive test images; return (clean_batch, marked_batch, labels)."""
    pos = test_df[test_df["Target"] == 1].reset_index(drop=True).iloc[:n]
    clean_ds = RSNADataset(pos, data_dir, marker_mode="clean", seed=0)
    xs_clean = torch.stack([clean_ds[i][0] for i in range(len(pos))]).to(device)
    # marked = clean + marker (recompute on the [0,1] image, then renormalize)
    marked = []
    for i in range(len(pos)):
        raw = clean_ds._read_image(pos.iloc[i]["patientId"])      # [0,1]
        raw = add_marker(raw)
        marked.append((raw - IMAGENET_MEAN) / IMAGENET_STD)
    xs_marked = torch.stack(marked).to(device)
    return xs_clean, xs_marked

def attention_attribution(model, xs_marked, device, out_dir):
    """For each (layer, head): mean CLS->marker-token attention on marked images."""
    L = len(model.blocks)
    H = model.blocks[0].attn.num_heads
    store = {}
    handles = []
    def mk_hook(li):
        def pre_hook(module, args):              # args[0] = attn weights (B, H, N, N)
            store[li] = args[0].detach()
            return None
        return pre_hook
    for li, blk in enumerate(model.blocks):
        handles.append(blk.attn.attn_drop.register_forward_pre_hook(mk_hook(li)))
    model.eval()
    with torch.no_grad():
        _ = model(xs_marked)
    for h in handles:
        h.remove()
    mat = np.zeros((L, H))
    for li in range(L):
        attn = store[li]                          # (B, H, N, N)
        cls_to_marker = attn[:, :, 0, MARKER_TOKENS].sum(dim=-1)   # (B, H)
        mat[li] = cls_to_marker.mean(dim=0).cpu().numpy()
    plt.figure(figsize=(7, 6))
    plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(label="CLS attention on marker patch")
    plt.xlabel("Head"); plt.ylabel("Layer"); plt.title("Attention-head attribution (marker region)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig_attention_attribution.png"),
                                    dpi=150, bbox_inches="tight"); plt.close()
    return mat

def capture_proj_inputs(model, x):
    """Run model on x; return {layer_idx: input tensor to attn.proj} (concatenated heads)."""
    store, handles = {}, []
    def mk(li):
        def pre_hook(module, args):
            store[li] = args[0].detach().clone()
            return None
        return pre_hook
    for li, blk in enumerate(model.blocks):
        handles.append(blk.attn.proj.register_forward_pre_hook(mk(li)))
    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()
    return store

def diff_logit(model, x):
    with torch.no_grad():
        out = model(x)
    return (out[:, 1] - out[:, 0])               # (B,)  positive-vs-negative logit gap

def activation_patching(model, xs_clean, xs_marked, device, out_dir):
    """
    For each (layer, head): patch the CLEAN head activation into the MARKED run and measure
    the mean change in P(positive). Large change => that head carries the marker signal.
    """
    L = len(model.blocks); H = model.blocks[0].attn.num_heads
    head_dim = model.blocks[0].attn.head_dim
    clean_proj = capture_proj_inputs(model, xs_clean)
    with torch.no_grad():
        base_p = F.softmax(model(xs_marked), dim=1)[:, 1]    # (B,)
    effect = np.zeros((L, H))
    for li in range(L):
        for h in range(H):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            def pre_hook(module, args, _sl=sl, _li=li):
                inp = args[0].clone()
                inp[:, :, _sl] = clean_proj[_li][:, :, _sl]   # inject clean head slice
                return (inp,)
            handle = model.blocks[li].attn.proj.register_forward_pre_hook(pre_hook)
            with torch.no_grad():
                p = F.softmax(model(xs_marked), dim=1)[:, 1]
            handle.remove()
            effect[li, h] = (p - base_p).abs().mean().item()
    plt.figure(figsize=(7, 6))
    plt.imshow(effect, aspect="auto", cmap="magma")
    plt.colorbar(label="mean |\u0394 P(positive)| when patching clean head")
    plt.xlabel("Head"); plt.ylabel("Layer"); plt.title("Activation patching: causal effect per head")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "fig_patching_causal.png"),
                                    dpi=150, bbox_inches="tight"); plt.close()
    return effect

@contextlib.contextmanager
def ablate_heads(model, heads):
    """heads: list of (layer, head). Zero those head slices at attn.proj input."""
    head_dim = model.blocks[0].attn.head_dim
    handles = []
    for (li, h) in heads:
        sl = slice(h * head_dim, (h + 1) * head_dim)
        def pre_hook(module, args, _sl=sl):
            inp = args[0].clone(); inp[:, :, _sl] = 0.0
            return (inp,)
        handles.append(model.blocks[li].attn.proj.register_forward_pre_hook(pre_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()

def ablation_recovery(model, effect, test_df, data_dir, device, batch, out_dir, top_k=5):
    """Ablate the top-k most causal heads; compare clean-test accuracy before vs after."""
    flat = [(li, h, effect[li, h]) for li in range(effect.shape[0]) for h in range(effect.shape[1])]
    flat.sort(key=lambda t: t[2], reverse=True)
    top = [(li, h) for (li, h, _) in flat[:top_k]]
    clean_ds = RSNADataset(test_df, data_dir, marker_mode="clean", seed=0)
    ld = DataLoader(clean_ds, batch_size=batch, shuffle=False, num_workers=4)
    preds, _, ys = run_inference(model, ld, device)
    acc_before = accuracy_score(ys, preds)
    with ablate_heads(model, top):
        preds2, _, ys2 = run_inference(model, ld, device)
    acc_after = accuracy_score(ys2, preds2)
    plt.figure(figsize=(5, 5))
    plt.bar(["before", f"after ablating\ntop-{top_k} heads"], [acc_before, acc_after],
            color=["#4477aa", "#cc6677"])
    for i, v in enumerate([acc_before, acc_after]):
        plt.text(i, v + 0.005, f"{v:.3f}", ha="center")
    plt.ylabel("Clean-test accuracy"); plt.ylim(0, 1)
    plt.title("Ablation recovery"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_ablation_recovery.png"), dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nTop-{top_k} causal heads (layer, head): {top}")
    print(f"Clean accuracy before ablation: {acc_before:.3f}  | after: {acc_after:.3f}")
    return {"top_heads": top, "acc_before": acc_before, "acc_after": acc_after}

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="./out")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--subset", type=int, default=0, help="use only N patients (quick test)")
    ap.add_argument("--skip_train", action="store_true", help="load ./out/vit_cxr.pt and only analyze")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| marker tokens:", MARKER_TOKENS)

    labels = load_labels(args.data_dir)
    if args.subset:
        labels = labels.sample(n=min(args.subset, len(labels)), random_state=0).reset_index(drop=True)
    train_df, val_df, test_df = make_splits(labels)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"(positive rate {labels['Target'].mean():.3f})")

    model = build_model(device)
    ckpt = os.path.join(args.out_dir, "vit_cxr.pt")
    if args.skip_train and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("loaded checkpoint", ckpt)
    else:
        train_ds = RSNADataset(train_df, args.data_dir, marker_mode="train_rule", seed=0)
        val_ds = RSNADataset(val_df, args.data_dir, marker_mode="train_rule", seed=1)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
        cw = class_weights(train_df, device)
        model = train_model(model, train_loader, val_loader, device, args.epochs, args.lr, args.out_dir, cw)

    # Baseline (marker-consistent test ~ the data the model was trained on) + three-way test
    results = three_way_test(model, test_df, args.data_dir, device, args.batch, args.out_dir)

    # Mechanistic analysis on paired positive images
    disable_fused_attn(model)
    xs_clean, xs_marked = get_paired_batch(test_df, args.data_dir, device, n=32)
    attn_mat = attention_attribution(model, xs_marked, device, args.out_dir)
    effect = activation_patching(model, xs_clean, xs_marked, device, args.out_dir)
    ablation = ablation_recovery(model, effect, test_df, args.data_dir, device, args.batch, args.out_dir)

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"three_way": results, "ablation": ablation,
                   "marker_tokens": MARKER_TOKENS}, f, indent=2)
    print("\nDone. Figures + results.json saved in", args.out_dir)

if __name__ == "__main__":
    main()
