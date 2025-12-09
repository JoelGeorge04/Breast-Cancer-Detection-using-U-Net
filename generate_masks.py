import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def generate_mask(image_path, K=3):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # reshape to (num_pixels, 3)
    pixel_vals = img_rgb.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    # K-means
    kmeans = KMeans(n_clusters=K, random_state=42).fit(pixel_vals)
    labels = kmeans.labels_.reshape(img_rgb.shape[:2])

    # select tumor cluster (max intensity or lowest intensity)
    unique, counts = np.unique(labels, return_counts=True)
    tumor_cluster = unique[np.argmax(counts)]

    # create mask
    mask = np.uint8(labels == tumor_cluster) * 255

    # clean mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 7)

    return mask

def process_folder(main_folder):
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                
                img_path = os.path.join(root, file)
                output_path = img_path.replace("dataset", "dataset_masks")

                # Skip already processed images
                if os.path.exists(output_path):
                    print("Skipping already processed:", output_path)
                    continue

                print("Processing:", img_path)

                # generate mask
                mask = generate_mask(img_path)

                # ensure mask directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # save mask
                cv2.imwrite(output_path, mask)

process_folder("dataset")
