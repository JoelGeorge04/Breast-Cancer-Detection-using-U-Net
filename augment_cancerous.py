"""
Augment class-1 (cancerous) images and their paired masks to reduce class imbalance.

Source  : final_dataset/images/1  +  final_dataset/masks/1
Output  : same folders  (new augmented files are added alongside originals)

Default target: match class-0 count (27 944) so the dataset becomes ~1:1
Run:  python augment_cancerous.py
      python augment_cancerous.py --target 20000   # custom target
"""

import os
import cv2
import numpy as np
import argparse
import random
from tqdm import tqdm

IMAGES_DIR = "final_dataset/images/1"
MASKS_DIR  = "final_dataset/masks/1"
TARGET_DEFAULT = 27944   # match class-0 count


# ---------------------------------------------------------------------------
# Augmentation helpers (applied identically to image + mask)
# ---------------------------------------------------------------------------

def aug_hflip(img, mask):
    return cv2.flip(img, 1), cv2.flip(mask, 1)

def aug_vflip(img, mask):
    return cv2.flip(img, 0), cv2.flip(mask, 0)

def aug_rot90(img, mask):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

def aug_rot180(img, mask):
    return cv2.rotate(img, cv2.ROTATE_180), cv2.rotate(mask, cv2.ROTATE_180)

def aug_rot270(img, mask):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

def aug_hflip_rot90(img, mask):
    img, mask = aug_hflip(img, mask)
    return aug_rot90(img, mask)

def aug_brightness(img, mask):
    """Adjust brightness/contrast of image only (mask unchanged)."""
    alpha = random.uniform(0.75, 1.25)   # contrast
    beta  = random.randint(-30, 30)       # brightness
    img_out = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    return img_out, mask.copy()

def aug_noise(img, mask):
    """Add Gaussian noise to image only."""
    noise = np.random.normal(0, 10, img.shape).astype(np.float32)
    img_out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img_out, mask.copy()


AUGMENTATIONS = [
    ("hflip",       aug_hflip),
    ("vflip",       aug_vflip),
    ("rot90",       aug_rot90),
    ("rot180",      aug_rot180),
    ("rot270",      aug_rot270),
    ("hflip_rot90", aug_hflip_rot90),
    ("bright",      aug_brightness),
    ("noise",       aug_noise),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET_DEFAULT,
                        help="Target number of class-1 images after augmentation")
    args = parser.parse_args()

    # Collect original image filenames (skip files already prefixed with "aug_")
    orig_files = [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        and not f.startswith("aug_")
    ]

    current_total = len(os.listdir(IMAGES_DIR))
    needed = args.target - current_total

    if needed <= 0:
        print(f"Already have {current_total} class-1 images (target={args.target}). Nothing to do.")
        return

    print(f"Current class-1 images : {current_total}")
    print(f"Target                 : {args.target}")
    print(f"Images to generate     : {needed}")

    # Build an infinite cycling list of (filename, aug_name, aug_fn) pairs
    # by cycling through originals and all augmentation types
    task_queue = []
    aug_cycle = 0
    while len(task_queue) < needed:
        aug_name, aug_fn = AUGMENTATIONS[aug_cycle % len(AUGMENTATIONS)]
        for fname in orig_files:
            if len(task_queue) >= needed:
                break
            task_queue.append((fname, aug_name, aug_fn, aug_cycle // len(AUGMENTATIONS)))
        aug_cycle += 1

    random.shuffle(task_queue)

    print(f"\nGenerating {len(task_queue)} augmented images...\n")
    saved = 0
    skipped = 0

    for fname, aug_name, aug_fn, repeat_idx in tqdm(task_queue):
        img_path  = os.path.join(IMAGES_DIR, fname)
        mask_path = os.path.join(MASKS_DIR,  fname)

        img  = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            skipped += 1
            continue

        # Apply augmentation
        aug_img, aug_mask = aug_fn(img, mask if mask is not None else np.zeros(img.shape[:2], dtype=np.uint8))

        # Build unique output filename:  aug_<aug_name>_r<repeat>_<original>
        base, ext = os.path.splitext(fname)
        out_name = f"aug_{aug_name}_r{repeat_idx}_{base}{ext}"

        out_img_path  = os.path.join(IMAGES_DIR, out_name)
        out_mask_path = os.path.join(MASKS_DIR,  out_name)

        # Skip if already exists (resume support)
        if os.path.exists(out_img_path):
            skipped += 1
            continue

        cv2.imwrite(out_img_path, aug_img)
        if mask is not None:
            cv2.imwrite(out_mask_path, aug_mask)
        saved += 1

    final_count = len(os.listdir(IMAGES_DIR))
    print(f"\nDone.")
    print(f"  Saved    : {saved}")
    print(f"  Skipped  : {skipped}")
    print(f"  Class-1 images now : {final_count}")
    print(f"  Class-0 images     : 27944")
    ratio = 27944 / max(final_count, 1)
    print(f"  Imbalance ratio    : {ratio:.2f}:1")


if __name__ == "__main__":
    main()
