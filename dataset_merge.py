import os
import shutil

SOURCE_IMG = "dataset"
SOURCE_MASK = "dataset_masks"
OUTPUT = "final_dataset"

def ensure(path):
    if not os.path.exists(path):
        os.makedirs(path)

def merge_dataset():
    print("Merging hierarchical dataset...")

    classes = ["0", "1"]

    # Create final folders
    for split in ["images", "masks"]:
        for cls in classes:
            ensure(os.path.join(OUTPUT, split, cls))

    parent_folders = os.listdir(SOURCE_IMG)

    for parent in parent_folders:
        img_parent_path = os.path.join(SOURCE_IMG, parent)
        mask_parent_path = os.path.join(SOURCE_MASK, parent)

        # Skip if not a directory
        if not os.path.isdir(img_parent_path):
            continue

        print(f"Processing folder: {parent}")

        for cls in classes:
            img_class_path = os.path.join(img_parent_path, cls)
            mask_class_path = os.path.join(mask_parent_path, cls)

            if not os.path.exists(img_class_path) or not os.path.exists(mask_class_path):
                print(f"Missing class folder in: {parent}/{cls}")
                continue

            img_files = os.listdir(img_class_path)

            for fname in img_files:
                src_img = os.path.join(img_class_path, fname)
                src_mask = os.path.join(mask_class_path, fname)

                if not os.path.exists(src_mask):
                    print(f"Mask missing for: {parent}/{cls}/{fname}")
                    continue

                dest_img = os.path.join(OUTPUT, "images", cls, fname)
                dest_mask = os.path.join(OUTPUT, "masks", cls, fname)

                shutil.copy(src_img, dest_img)
                shutil.copy(src_mask, dest_mask)

    print("\nDataset merged successfully.")

if __name__ == "__main__":
    merge_dataset()
    print("Final dataset ready.")
