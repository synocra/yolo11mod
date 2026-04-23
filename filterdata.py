import os
import random
import shutil

random.seed(42)

# ===== CONFIG =====
classes_target = [
    "deformation",
    "obstacle",
    "rupture",
    "disconnect",
    "misalignment",
    "deposition"
]

# mapping class name -> index (sesuaikan dengan dataset kamu)
class_map = {
    "deformation": 0,
    "obstacle": 1,
    "rupture": 2,
    "disconnect": 3,
    "misalignment": 4,
    "deposition": 5
}

IMG_DIR = "/content/dataset/images"
LBL_DIR = "/content/dataset/labels"

OUT_DIR = "/content/filtered_dataset"

# jumlah per class
TRAIN_PER_CLASS = 150
VAL_PER_CLASS = 100

# ==================

os.makedirs(OUT_DIR, exist_ok=True)

splits = ["train", "val"]

for split in splits:
    os.makedirs(f"{OUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/labels/{split}", exist_ok=True)

# ===== STEP 1: grouping image per class =====
class_images = {c: [] for c in classes_target}

for file in os.listdir(LBL_DIR):
    label_path = os.path.join(LBL_DIR, file)

    with open(label_path, "r") as f:
        lines = f.readlines()

    classes_in_img = set([int(l.split()[0]) for l in lines])

    for cname, cid in class_map.items():
        if cid in classes_in_img:
            class_images[cname].append(file.replace(".txt", ".jpg"))

# ===== STEP 2: sampling =====
selected = {"train": [], "val": []}

for cname in classes_target:
    imgs = list(set(class_images[cname]))  # unique
    random.shuffle(imgs)

    train_imgs = imgs[:TRAIN_PER_CLASS]
    val_imgs = imgs[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS]

    selected["train"].extend(train_imgs)
    selected["val"].extend(val_imgs)

# ===== STEP 3: remove duplicates antar class =====
selected["train"] = list(set(selected["train"]))
selected["val"] = list(set(selected["val"]))

# ===== STEP 4: copy files =====
def copy_files(file_list, split):
    for img_name in file_list:
        label_name = img_name.replace(".jpg", ".txt")

        src_img = os.path.join(IMG_DIR, img_name)
        src_lbl = os.path.join(LBL_DIR, label_name)

        dst_img = f"{OUT_DIR}/images/{split}/{img_name}"
        dst_lbl = f"{OUT_DIR}/labels/{split}/{label_name}"

        if os.path.exists(src_img) and os.path.exists(src_lbl):
            shutil.copy(src_img, dst_img)
            shutil.copy(src_lbl, dst_lbl)

copy_files(selected["train"], "train")
copy_files(selected["val"], "val")

print("✅ Dataset filtering selesai!")