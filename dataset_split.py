import os, shutil
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


KVASIR_ROOT = Path("/input/datasets/kvasir/Kvasir-SEG")          
CVC_ROOT    = Path("/input/datasets/cvcclinic/PNG")               
OUT_ROOT    = Path("/input/datasets/PolypM")

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
SEED = 42
MASK_THRESHOLD = 127   


assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Ratios must sum to 1"

def find_dir(root, *names):
    """Find a sub-folder by case-insensitive name."""
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in [n.lower() for n in names]:
            return child
    raise FileNotFoundError(f"None of {names} found in {root}: {[c.name for c in root.iterdir()]}")

def collect_pairs(img_dir, msk_dir, prefix):
    """Match image and mask by file stem."""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in exts}
    msks = {p.stem: p for p in msk_dir.iterdir() if p.suffix.lower() in exts}
    common = sorted(set(imgs) & set(msks))
    missing = set(imgs) ^ set(msks)
    if missing:
        print(f"[{prefix}] WARNING: {len(missing)} unmatched files ignored")
    return [(prefix, stem, imgs[stem], msks[stem]) for stem in common]


kv_pairs  = collect_pairs(find_dir(KVASIR_ROOT, "images"),
                          find_dir(KVASIR_ROOT, "masks"), "kvasir")
cvc_pairs = collect_pairs(find_dir(CVC_ROOT, "Original"),
                          find_dir(CVC_ROOT, "Ground Truth", "GroundTruth", "Ground_Truth"), "cvc")

all_pairs = kv_pairs + cvc_pairs
N = len(all_pairs)
print(f"Kvasir-SEG: {len(kv_pairs)} | CVC-ClinicDB: {len(cvc_pairs)} | Total: {N}")


n_train = round(N * TRAIN_RATIO)
n_val   = int(N * VAL_RATIO)
n_test  = N - n_train - n_val         
print(f"Split ratios {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO} -> train {n_train}, val {n_val}, test {n_test}")


source = [p[0] for p in all_pairs]
trainval, test, src_tv, _ = train_test_split(
    all_pairs, source, test_size=n_test, stratify=source, random_state=SEED)
train, val = train_test_split(
    trainval, test_size=n_val, stratify=src_tv, random_state=SEED)

splits = {"train": train, "val": val, "test": test}


if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

for split, items in splits.items():
    img_out = OUT_ROOT / split / "images"
    msk_out = OUT_ROOT / split / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    for src, stem, img_path, msk_path in tqdm(items, desc=split):
        name = f"{src}_{stem}.png"          

        Image.open(img_path).convert("RGB").save(img_out / name)

        m = np.array(Image.open(msk_path).convert("L"))
        m = ((m > MASK_THRESHOLD) * 255).astype(np.uint8)
        Image.fromarray(m).save(msk_out / name)


print("\nSummary")
for split, items in splits.items():
    n_img = len(os.listdir(OUT_ROOT / split / "images"))
    n_msk = len(os.listdir(OUT_ROOT / split / "masks"))
    n_kv  = sum(1 for it in items if it[0] == "kvasir")
    n_cvc = sum(1 for it in items if it[0] == "cvc")
    print(f"{split:5s}: {n_img} images, {n_msk} masks  ({n_img/N:.1%} of total; Kvasir {n_kv}, CVC {n_cvc})")




