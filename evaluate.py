import torch
import numpy as np
from loader import load_dataset
from model import UNet
from sklearn.metrics import confusion_matrix

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

def dice_score(pred, gt):
    """Calculate Dice coefficient between predicted and ground truth masks"""
    intersection = (pred * gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)

def iou_score(pred, gt):
    """Calculate Intersection over Union (IoU)"""
    intersection = (pred * gt).sum()
    union = (pred + gt).sum() - intersection
    return intersection / (union + 1e-8)

dice_scores = []
iou_scores = []
img_true_labels = []  # 1 if image has tumor, 0 if only background
img_pred_labels = []  # 1 if predicted mask has tumor, 0 if only background

with torch.no_grad():
    for i in range(len(images)):
        img = images[i].unsqueeze(0)
        gt = (masks[i] > 0.5).int()
        
        pred = torch.sigmoid(model(img))
        pred = (pred > 0.5).int()
        
        # Flatten for metric calculation
        gt_flat = gt.cpu().numpy().flatten()
        pred_flat = pred.cpu().numpy().flatten()
        
        # Calculate per-image metrics
        dice = dice_score(pred_flat, gt_flat)
        iou = iou_score(pred_flat, gt_flat)
        
        dice_scores.append(dice)
        iou_scores.append(iou)
        
        # Image-level classification (has tumor or not)
        img_true_labels.append(1 if gt_flat.sum() > 0 else 0)
        img_pred_labels.append(1 if pred_flat.sum() > 0 else 0)

# Image-wise confusion matrix
cm = confusion_matrix(img_true_labels, img_pred_labels, labels=[0, 1])

# Calculate average metrics
avg_dice = np.mean(dice_scores)
avg_iou = np.mean(iou_scores)
std_dice = np.std(dice_scores)
std_iou = np.std(iou_scores)

print("=" * 60)
print("IMAGE-WISE SEGMENTATION EVALUATION")
print("=" * 60)
print(f"Number of validation images: {len(images)}")
print()
print("CONFUSION MATRIX (Image-level: Has Tumor vs No Tumor)")
print("                    No Tumor (Pred)  |  Has Tumor (Pred)")
print(f"No Tumor (True)            {cm[0,0]:5d}        |        {cm[0,1]:5d}")
print(f"Has Tumor (True)           {cm[1,0]:5d}        |        {cm[1,1]:5d}")
print()
print("IMAGE-LEVEL METRICS")
print("-" * 60)
print(f"Average Dice Coefficient:  {avg_dice:.4f} (±{std_dice:.4f})")
print(f"Average IoU (Jaccard):     {avg_iou:.4f} (±{std_iou:.4f})")
print()
print("PER-IMAGE METRIC RANGES")
print("-" * 60)
print(f"Dice Score  - Min: {np.min(dice_scores):.4f}, Max: {np.max(dice_scores):.4f}")
print(f"IoU Score   - Min: {np.min(iou_scores):.4f}, Max: {np.max(iou_scores):.4f}")
print("=" * 60)

