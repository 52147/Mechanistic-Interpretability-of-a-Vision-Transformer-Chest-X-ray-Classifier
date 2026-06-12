import os, numpy as np, pandas as pd, pydicom
from PIL import Image

IMG = "stage_2_train_images"
OUT = os.path.expanduser("~/cxr_examples"); os.makedirs(OUT, exist_ok=True)

# one Normal + one Lung Opacity (uses detailed class info if present)
if os.path.exists("stage_2_detailed_class_info.csv"):
    d = pd.read_csv("stage_2_detailed_class_info.csv").drop_duplicates("patientId")
    normal_id  = d[d["class"] == "Normal"]["patientId"].iloc[0]
    opacity_id = d[d["class"] == "Lung Opacity"]["patientId"].iloc[0]
else:
    l = pd.read_csv("stage_2_train_labels.csv").drop_duplicates("patientId")
    normal_id  = l[l["Target"] == 0]["patientId"].iloc[0]
    opacity_id = l[l["Target"] == 1]["patientId"].iloc[0]

def load(pid, size=None):
    ds = pydicom.dcmread(os.path.join(IMG, pid + ".dcm"))
    a = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.max() - a
    a = (a - a.min()) / (a.max() - a.min() + 1e-8) * 255.0
    im = Image.fromarray(a.astype(np.uint8)).convert("L")
    return im.resize((size, size)) if size else im

load(normal_id).save(os.path.join(OUT, "ex_normal.png"))
load(opacity_id).save(os.path.join(OUT, "ex_lung_opacity.png"))

orig = load(opacity_id, size=224); orig.save(os.path.join(OUT, "ex_original.png"))
m = np.array(orig.convert("RGB")); m[0:16, 0:16, :] = 255   # one-patch white marker
Image.fromarray(m).save(os.path.join(OUT, "ex_marker_injected.png"))

print("wrote 4 PNGs to", OUT, "| normal:", normal_id, "| opacity:", opacity_id)
