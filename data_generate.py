from pathlib import Path
import random
import shutil

from torchvision.datasets import CIFAR100
from PIL import Image


# =========================
# Basic settings
# =========================

ROOT = Path("data")
OUT_DIR = ROOT / "cifar100_images"

TRAIN_DIR = OUT_DIR / "train"
VAL_DIR = OUT_DIR / "val"
TEST_DIR = OUT_DIR / "test"

TARGET_SIZE = (224, 224)

SEED = 42
VAL_PER_CLASS = 50

EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# If True, remove the existing cifar100_images folder and regenerate everything.
# This is recommended to avoid duplicated or inconsistent train/val splits.
RESET_OUTPUT = True


# =========================
# Utility functions
# =========================

def list_images(directory: Path):
    if not directory.exists():
        return []
    return [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    ]


def export_cifar100_split(split_name: str, train_flag: bool):
    """
    Download CIFAR-100 through torchvision and export images
    into ImageFolder-style class folders.
    """
    dataset = CIFAR100(
        root=str(ROOT),
        train=train_flag,
        download=False
    )

    for i, (img, label) in enumerate(dataset):
        class_name = dataset.classes[label]

        save_dir = OUT_DIR / split_name / class_name
        save_dir.mkdir(parents=True, exist_ok=True)

        img = img.resize(TARGET_SIZE, resample=Image.BICUBIC)
        img.save(save_dir / f"{i:06d}.png", optimize=True)

    print(f"Finished exporting CIFAR-100 {split_name} split.")


def create_validation_split():
    """
    Move VAL_PER_CLASS images from each training class folder
    into the corresponding validation class folder.
    """
    random.seed(SEED)

    class_dirs = sorted([p for p in TRAIN_DIR.iterdir() if p.is_dir()])

    if not class_dirs:
        raise RuntimeError(f"No class folders found in {TRAIN_DIR.resolve()}")

    moved_total = 0

    for cls_dir in class_dirs:
        class_name = cls_dir.name

        train_imgs = sorted(list_images(cls_dir))

        val_cls_dir = VAL_DIR / class_name
        val_cls_dir.mkdir(parents=True, exist_ok=True)

        val_imgs = list_images(val_cls_dir)

        need = max(0, VAL_PER_CLASS - len(val_imgs))

        if need == 0:
            continue

        if len(train_imgs) < need:
            print(
                f"[WARN] Class {class_name}: "
                f"only {len(train_imgs)} training images available, "
                f"but need {need}. Moving all available images."
            )
            selected_imgs = train_imgs
        else:
            selected_imgs = random.sample(train_imgs, need)

        for src in selected_imgs:
            dst = val_cls_dir / src.name

            # Avoid accidental overwriting.
            if dst.exists():
                dst = val_cls_dir / f"{src.stem}__moved{src.suffix}"

            shutil.move(str(src), str(dst))
            moved_total += 1

    print(f"Finished creating validation split. moved_total={moved_total}")


def print_summary():
    print("\nDataset preparation finished.")
    print(f"Output directory: {OUT_DIR.resolve()}")
    print(f"Train directory:  {TRAIN_DIR.resolve()}")
    print(f"Val directory:    {VAL_DIR.resolve()}")
    print(f"Test directory:   {TEST_DIR.resolve()}")

    train_classes = sorted([p for p in TRAIN_DIR.iterdir() if p.is_dir()])
    if train_classes:
        first_class = train_classes[0].name
        print("\nExample class statistics:")
        print(f"Class: {first_class}")
        print(f"Train images: {len(list_images(TRAIN_DIR / first_class))}")
        print(f"Val images:   {len(list_images(VAL_DIR / first_class))}")
        print(f"Test images:  {len(list_images(TEST_DIR / first_class))}")


# =========================
# Main process
# =========================

if __name__ == "__main__":
    if RESET_OUTPUT and OUT_DIR.exists():
        print(f"Removing existing directory: {OUT_DIR.resolve()}")
        shutil.rmtree(OUT_DIR)

    # Step 1: export CIFAR-100 train and test images.
    export_cifar100_split(split_name="train", train_flag=True)
    export_cifar100_split(split_name="test", train_flag=False)

    # Step 2: create validation split from the training set.
    create_validation_split()

    # Step 3: print final information.
    print_summary()