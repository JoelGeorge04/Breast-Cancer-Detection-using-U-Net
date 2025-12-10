import os
import shutil
import random

DATASET = "final_dataset"
OUTPUT = "dataset_split"
SPLIT = [0.7, 0.2, 0.1]  # train(70), val(20), test(10)

def ensure(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_data():
    classes = ["0", "1"]

    for split in ["train", "val", "test"]:
        for sub in ["images", "masks"]:
            for cls in classes:
                ensure(os.path.join(OUTPUT, split, sub, cls))

    for cls in classes:
        img_folder = os.path.join(DATASET, "images", cls)
        files = os.listdir(img_folder)
        random.shuffle(files)

        total = len(files)
        train_end = int(total * SPLIT[0])
        val_end = train_end + int(total * SPLIT[1])

        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:]
        }

        for split, file_list in splits.items():
            for fname in file_list:
                src_img = os.path.join(DATASET, "images", cls, fname)
                src_mask = os.path.join(DATASET, "masks", cls, fname)

                dst_img = os.path.join(OUTPUT, split, "images", cls, fname)
                dst_mask = os.path.join(OUTPUT, split, "masks", cls, fname)

                shutil.copy(src_img, dst_img)
                shutil.copy(src_mask, dst_mask)

    print("Dataset successfully split!")

if __name__ == "__main__":
    split_data()
