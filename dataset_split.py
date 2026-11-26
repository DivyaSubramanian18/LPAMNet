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

import os
from pathlib import Path
from PIL import Image
import imagehash
import pandas as pd


ROOT = Path("input/datasets/PolypM/")

TRAIN_DIR = ROOT / "train" / "images"
VAL_DIR   = ROOT / "val"   / "images"
TEST_DIR  = ROOT / "test"  / "images"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_images(folder):
    return sorted([
        p for p in folder.rglob("*")
        if p.suffix.lower() in EXTENSIONS
    ])


train_images = get_images(TRAIN_DIR)
val_images   = get_images(VAL_DIR)
test_images  = get_images(TEST_DIR)

print("Train:", len(train_images))
print("Val  :", len(val_images))
print("Test :", len(test_images))

def get_hashes(path):

    img = Image.open(path).convert("RGB")

    transforms = {
        "original": img,
        "hflip": img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "vflip": img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        "rot90": img.rotate(90, expand=True),
        "rot180": img.rotate(180, expand=True),
        "rot270": img.rotate(270, expand=True),
    }

    return {
        name: imagehash.phash(im)
        for name, im in transforms.items()
    }


datasets = {
    "train": train_images,
    "val": val_images,
    "test": test_images
}

hash_db = {}

for split, images in datasets.items():

    hash_db[split] = []

    for path in images:

        try:
            hashes = get_hashes(path)

            hash_db[split].append({
                "path": str(path),
                "hashes": hashes
            })

        except Exception as e:
            print("Error:", path, e)


def compare_splits(split_a, split_b, max_distance=5):

    results = []

    A = hash_db[split_a]
    B = hash_db[split_b]

    for item_a in A:

        for item_b in B:

            best_distance = 999
            best_transform = None

            for trans_a, hash_a in item_a["hashes"].items():
                for trans_b, hash_b in item_b["hashes"].items():

                    distance = hash_a - hash_b

                    if distance < best_distance:
                        best_distance = distance
                        best_transform = f"{trans_a} ↔ {trans_b}"

            if best_distance <= max_distance:

                results.append({
                    "split_a": split_a,
                    "image_a": item_a["path"],
                    "split_b": split_b,
                    "image_b": item_b["path"],
                    "hash_distance": best_distance,
                    "transformation": best_transform
                })

    return pd.DataFrame(results)

train_val = compare_splits("train", "val")
train_test = compare_splits("train", "test")
val_test = compare_splits("val", "test")



print(train_val)
print(train_test)
print(val_test)


OUTPUT_DIR = Path("input/datasets/similarity_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_val.to_csv(
    OUTPUT_DIR / "train_val_similarity.csv",
    index=False
)

train_test.to_csv(
    OUTPUT_DIR / "train_test_similarity.csv",
    index=False
)

val_test.to_csv(
    OUTPUT_DIR / "val_test_similarity.csv",
    index=False
)

print("Results saved to:")
print(OUTPUT_DIR)

print("\nFiles:")
for f in OUTPUT_DIR.iterdir():
    print(f)

import pandas as pd
from pathlib import Path

files = {
    "Train-Val": "input/datasets/similarity_analysis/train_val_similarity.csv",
    "Train-Test": "input/datasets/similarity_analysis/train_test_similarity.csv",
    "Val-Test": "input/datasets/similarity_analysis/val_test_similarity.csv"
}


summary = []

for comparison, file_path in files.items():

    df = pd.read_csv(file_path)

    print("\n" + "=" * 70)
    print(comparison)
    print("=" * 70)

    print(f"Total flagged pairs : {len(df)}")

    for threshold in [0, 2, 4]:

        matches = df[df["hash_distance"] <= threshold]

        unique_a = matches["image_a"].nunique()
        unique_b = matches["image_b"].nunique()

        print(
            f"Hash distance <= {threshold}: "
            f"{len(matches)} pairs | "
            f"{unique_a} unique images in split A | "
            f"{unique_b} unique images in split B"
        )

        summary.append({
            "comparison": comparison,
            "threshold": threshold,
            "matching_pairs": len(matches),
            "unique_images_split_A": unique_a,
            "unique_images_split_B": unique_b
        })

summary_df = pd.DataFrame(summary)

print("\n\nFINAL SUMMARY")
print("=" * 70)
print(summary_df.to_string(index=False))


output_dir = Path("input/datasets/similarity_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "similarity_summary.csv"

summary_df.to_csv(output_file, index=False)

print("\nSaved to:")
print(output_file)

for comparison, file_path in files.items():

    df = pd.read_csv(file_path)

    print("\n" + "=" * 70)
    print(comparison)
    print("=" * 70)

    for threshold in [0, 2]:

        matches = df[df["hash_distance"] <= threshold]

        images_a = set(matches["image_a"])
        images_b = set(matches["image_b"])

        print(f"\nDistance <= {threshold}")

        print("Unique images in split A:", len(images_a))
        print("Unique images in split B:", len(images_b))

        # Show filenames
        print("\nSplit A:")
        for x in sorted(images_a):
            print(Path(x).name)

        print("\nSplit B:")
        for x in sorted(images_b):
            print(Path(x).name)
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import math

csv_files = {
    "Train-Val": "input/datasets/similarity_analysis/train_val_similarity.csv",
    "Train-Test": "input/datasets/similarity_analysis/train_test_similarity.csv",
    "Val-Test": "input/datasets/similarity_analysis/val_test_similarity.csv"
}

output_dir = Path("input/datasets/similarity_analysis/contact_sheets")
output_dir.mkdir(parents=True, exist_ok=True)

all_pairs = []

for comparison, csv_path in csv_files.items():

    df = pd.read_csv(csv_path)
    matches = df[df["hash_distance"] == 0].copy()

    matches["comparison"] = comparison

    all_pairs.append(matches)

pairs = pd.concat(all_pairs, ignore_index=True)

print("Total distance-0 pairs:", len(pairs))
print(pairs[[
    "comparison",
    "image_a",
    "image_b",
    "hash_distance",
    "transformation"
]].to_string(index=False))

def load_image(path, size=(350, 350)):

    img = Image.open(path).convert("RGB")
    img.thumbnail(size)

    canvas = Image.new("RGB", size, "white")

    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2

    canvas.paste(img, (x, y))

    return canvas


for comparison in pairs["comparison"].unique():

    subset = pairs[pairs["comparison"] == comparison].reset_index(drop=True)

    n = len(subset)

    fig, axes = plt.subplots(
        n,
        2,
        figsize=(12, 5 * n)
    )
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"{comparison} — Distance-0 Similarity Matches",
        fontsize=18,
        fontweight="bold"
    )

    for i, row in subset.iterrows():

        img_a = load_image(row["image_a"])
        img_b = load_image(row["image_b"])

        axes[i][0].imshow(img_a)
        axes[i][0].axis("off")

        axes[i][1].imshow(img_b)
        axes[i][1].axis("off")

        name_a = Path(row["image_a"]).name
        name_b = Path(row["image_b"]).name

        axes[i][0].set_title(
            f"{name_a}",
            fontsize=12
        )

        axes[i][1].set_title(
            f"{name_b}",
            fontsize=12
        )
        fig.text(
            0.5,
            1 - (i + 0.5) / n,
            f"Hash distance: {row['hash_distance']} | "
            f"Transformation: {row['transformation']}",
            ha="center",
            fontsize=10
        )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    safe_name = comparison.lower().replace("-", "_")

    output_path = output_dir / f"{safe_name}_distance0.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


print("\nAll contact sheets saved in:")
print(output_dir)



import os
import cv2
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np

SRC_ROOT = Path(
    "/input/datasets/PolypM"
)

OUT_ROOT = Path(
    "/input/datasets/PolypMS"
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

SPLITS = ["train", "val", "test"]


HASH_THRESHOLD = 2

TARGET_RATIO = {
    "train": 0.70,
    "val":   0.15,
    "test":  0.15
}

def get_images(split):

    image_dir = SRC_ROOT / split / "images"

    files = []

    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)

    return sorted(files)


all_images = {}

for split in SPLITS:
    all_images[split] = get_images(split)
    print(f"{split:5s}: {len(all_images[split])} images")

print()

def dhash(image_path, hash_size=16):

    img = Image.open(image_path).convert("L")
    img = img.resize((hash_size + 1, hash_size))

    arr = np.asarray(img)

    diff = arr[:, 1:] > arr[:, :-1]

    return diff.flatten()


def hamming_distance(h1, h2):

    return int(np.sum(h1 != h2))

class UnionFind:

    def __init__(self, n):

        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):

        while self.parent[x] != x:

            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:

            self.parent[ra] = rb

        elif self.rank[ra] > self.rank[rb]:

            self.parent[rb] = ra

        else:

            self.parent[rb] = ra
            self.rank[ra] += 1


all_files = []

for split in SPLITS:

    for p in all_images[split]:

        all_files.append(
            {
                "path": p,
                "original_split": split
            }
        )


print("Total images:", len(all_files))

hashes = []

for i, item in enumerate(all_files):

    try:
        hashes.append(dhash(item["path"]))

    except Exception as e:

        print("Hash error:", item["path"], e)
        hashes.append(None)

print("Hash calculation completed.")


uf = UnionFind(len(all_files))

similar_pairs = []

print()
print("Searching for visually similar images...")
print()

N = len(all_files)

for i in range(N):

    if hashes[i] is None:
        continue

    for j in range(i + 1, N):

        if hashes[j] is None:
            continue

        if (
            all_files[i]["original_split"]
            == all_files[j]["original_split"]
        ):
            continue

        dist = hamming_distance(hashes[i], hashes[j])

        if dist <= HASH_THRESHOLD:

            similar_pairs.append(
                (
                    i,
                    j,
                    dist
                )
            )

            uf.union(i, j)


print("Similar cross-split pairs:", len(similar_pairs))

groups = defaultdict(list)

for i in range(N):

    root = uf.find(i)

    groups[root].append(i)

multi_groups = [
    members
    for members in groups.values()
    if len(members) > 1
]

print("Total similarity groups:", len(multi_groups))

print()


cross_split_groups = []

for members in multi_groups:

    original_splits = set(
        all_files[i]["original_split"]
        for i in members
    )

    if len(original_splits) > 1:

        cross_split_groups.append(members)


print(
    "Groups crossing original train/val/test splits:",
    len(cross_split_groups)
)

print()


for k, members in enumerate(cross_split_groups[:20]):

    print("=" * 70)
    print(f"GROUP {k + 1}")

    for i in members:

        print(
            all_files[i]["original_split"],
            "->",
            all_files[i]["path"].name
        )


total_images = len(all_files)

target_counts = {
    split: int(total_images * TARGET_RATIO[split])
    for split in SPLITS
}

print()
print("Target counts:")
print(target_counts)

new_assignment = {}

assigned = set()

sorted_groups = sorted(
    groups.values(),
    key=len,
    reverse=True
)

current_counts = {
    "train": 0,
    "val": 0,
    "test": 0
}


def choose_split(group_size):

    remaining = {
        split:
        target_counts[split] - current_counts[split]
        for split in SPLITS
    }

    feasible = [
        split
        for split in SPLITS
        if remaining[split] >= group_size
    ]

    if feasible:

        return max(
            feasible,
            key=lambda x: remaining[x]
        )

    return max(
        SPLITS,
        key=lambda x: remaining[x]
    )


for group in sorted_groups:

    split = choose_split(len(group))

    for i in group:

        new_assignment[i] = split

        assigned.add(i)

    current_counts[split] += len(group)


print()
print("=" * 70)
print("NEW SPLIT DISTRIBUTION")
print("=" * 70)

for split in SPLITS:

    count = sum(
        1
        for i in range(N)
        if new_assignment[i] == split
    )

    print(
        f"{split:5s}: {count:5d} "
        f"({100 * count / N:.2f}%)"
    )

for split in SPLITS:

    (OUT_ROOT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUT_ROOT / split / "masks").mkdir(
        parents=True,
        exist_ok=True
    )


print()
print("Copying files...")
print()

missing_masks = []

for i, item in enumerate(all_files):

    src_img = item["path"]

    new_split = new_assignment[i]

    dst_img = (
        OUT_ROOT /
        new_split /
        "images" /
        src_img.name
    )

    shutil.copy2(src_img, dst_img)

    mask_candidates = [
        SRC_ROOT / item["original_split"] / "masks" / src_img.name,
        SRC_ROOT / item["original_split"] / "masks" / (src_img.stem + ".png"),
        SRC_ROOT / item["original_split"] / "masks" / (src_img.stem + ".jpg"),
    ]

    src_mask = None

    for candidate in mask_candidates:

        if candidate.exists():

            src_mask = candidate
            break

    if src_mask is None:

        missing_masks.append(src_img)

        continue

    dst_mask = (
        OUT_ROOT /
        new_split /
        "masks" /
        src_mask.name
    )

    shutil.copy2(src_mask, dst_mask)


print("Copy completed.")

if missing_masks:

    print()
    print("WARNING: Missing masks:", len(missing_masks))

    for p in missing_masks[:20]:

        print(p)

print()
print("=" * 70)
print("FINAL DATASET")
print("=" * 70)

for split in SPLITS:

    img_count = len(
        list(
            (OUT_ROOT / split / "images").glob("*")
        )
    )

    mask_count = len(
        list(
            (OUT_ROOT / split / "masks").glob("*")
        )
    )

    print(
        f"{split:5s}: "
        f"{img_count} images | "
        f"{mask_count} masks"
    )

print()
print("Output:")
print(OUT_ROOT)


from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(
    "/input/datasets/PolypMS"
)

SPLITS = ["train", "val", "test"]


def dhash(image_path, hash_size=16):

    img = Image.open(image_path).convert("L")
    img = img.resize((hash_size + 1, hash_size))

    arr = np.asarray(img)

    diff = arr[:, 1:] > arr[:, :-1]

    return diff.flatten()


def hamming_distance(h1, h2):

    return int(np.sum(h1 != h2))


split_images = {}

for split in SPLITS:

    split_images[split] = sorted(
        [
            p
            for p in (ROOT / split / "images").iterdir()
            if p.is_file()
        ]
    )

    print(
        split,
        len(split_images[split])
    )


hashes = {}

for split in SPLITS:

    hashes[split] = {}

    for p in split_images[split]:

        hashes[split][p] = dhash(p)

comparisons = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]


for split_a, split_b in comparisons:

    print()
    print("=" * 70)
    print(f"{split_a.upper()} - {split_b.upper()}")
    print("=" * 70)

    matches_0 = []
    matches_2 = []
    matches_4 = []

    for path_a, hash_a in hashes[split_a].items():

        for path_b, hash_b in hashes[split_b].items():

            d = hamming_distance(hash_a, hash_b)

            if d <= 4:

                matches_4.append(
                    (path_a, path_b, d)
                )

            if d <= 2:

                matches_2.append(
                    (path_a, path_b, d)
                )

            if d == 0:

                matches_0.append(
                    (path_a, path_b, d)
                )

    print(
        "Distance <= 0:",
        len(matches_0)
    )

    print(
        "Distance <= 2:",
        len(matches_2)
    )

    print(
        "Distance <= 4:",
        len(matches_4)
    )


    if matches_2:

        print()
        print("WARNING: remaining <=2 matches:")

        for a, b, d in matches_2[:20]:

            print(
                d,
                a.name,
                "<->",
                b.name
            )

    else:

        print("✓ No cross-split matches with distance <= 2")

from pathlib import Path
import shutil

ROOT = Path("/input/datasets/PolypMS")

filename = "kvasir_cju8amfdtqi4x09871tygrgqe.png"

src_img = ROOT / "val" / "images" / filename
dst_img = ROOT / "train" / "images" / filename

src_mask = ROOT / "val" / "masks" / filename
dst_mask = ROOT / "train" / "masks" / filename


shutil.move(src_img, dst_img)


if src_mask.exists():
    shutil.move(src_mask, dst_mask)
else:
    print("WARNING: mask not found:", src_mask)

print("Moved:")
print(filename)
print("VAL -> TRAIN")


