import torch
from loader import load_dataset
from model import UNet

# DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# LOAD DATA (KEEP ON CPU)
train_images, train_masks = load_dataset("dataset_split/train")
val_images, val_masks = load_dataset("dataset_split/val")

if train_images is None or len(train_images) == 0:
    print("No training data found")
    exit()

train_images = train_images.float()
train_masks = train_masks.float()

if val_images is not None:
    val_images = val_images.float()
    val_masks = val_masks.float()

# MODEL
model = UNet().to(device)

# LOSS & OPTIMIZER
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 5
batch_size = 4

print("Training started...")

# TRAIN LOOP
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    step_count = 0

    for i in range(0, len(train_images), batch_size):
        imgs = train_images[i:i + batch_size].to(device)
        masks = train_masks[i:i + batch_size].to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step_count += 1

        if step_count % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Step {step_count}")

    avg_loss = total_loss / step_count
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

    # SAVE CHECKPOINT
    torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

# SAVE FINAL MODEL
torch.save(model.state_dict(), "unet_model.pth")
print("Training complete. Final model saved as unet_model.pth")
