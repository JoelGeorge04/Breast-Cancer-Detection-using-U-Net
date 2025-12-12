import os
import cv2
import numpy as np
import torch

TARGET_SIZE = 128   # or 256

def load_dataset(root):
    images = []
    masks = []

    img_root = os.path.join(root, "images")
    mask_root = os.path.join(root, "masks")

    print("Loading:", img_root)

    if not os.path.exists(img_root):
        print("Missing:", img_root)
        return None, None

    for class_name in ["0", "1"]:
        img_dir = os.path.join(img_root, class_name)
        mask_dir = os.path.join(mask_root, class_name)

        if not os.path.exists(img_dir):
            print("Skipping missing folder:", img_dir)
            continue

        for file in os.listdir(img_dir):

            img_path = os.path.join(img_dir, file)
            mask_path = os.path.join(mask_dir, file)

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0)

            if img is None or mask is None:
                continue

            # resize both
            img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
            mask = cv2.resize(mask, (TARGET_SIZE, TARGET_SIZE))

            # preprocess
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype("float32") / 255.0
            img = np.transpose(img, (2, 0, 1))

            mask = mask.astype("float32") / 255.0
            mask = np.expand_dims(mask, axis=0)

            images.append(img)
            masks.append(mask)

    if len(images) == 0:
        print("No data found")
        return None, None

    return torch.tensor(np.array(images)), torch.tensor(np.array(masks))
