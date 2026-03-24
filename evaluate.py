import torch
import numpy as np
from loader import load_dataset
from model import UNet
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = UNet().to(device)
state_dict = torch.load(
    "unet_model.pth",
    map_location=device,
    weights_only=True
)
model.load_state_dict(state_dict)
model.eval()

# Load validation data
images, masks = load_dataset("dataset_split/val")
images = images.to(device)
masks = masks.to(device)

y_true = []
y_pred = []

with torch.no_grad():
    for i in range(len(images)):
        img = images[i].unsqueeze(0)
        gt = (masks[i] > 0.5).int()
        
        pred = torch.sigmoid(model(img))
        pred = (pred > 0.5).int()

        # Store pixel-wise labels
        y_true.append(gt.cpu().numpy().flatten())
        y_pred.append(pred.cpu().numpy().flatten())

# Convert to numpy arrays
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# Required metrics only
cm = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

print("Confusion Matrix:\n", cm)
print("Precision :", precision)
print("Recall :", recall)

