import torch
from loader import load_dataset
from model import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD DATA
train_images, train_masks = load_dataset("dataset_split/train")
val_images, val_masks = load_dataset("dataset_split/val")

if train_images is None or len(train_images) == 0:
    print("No training data found")
    exit()

train_images = train_images.to(device)
train_masks = train_masks.to(device)

# MODEL
model = UNet().to(device)

# LOSS + OPTIMIZER
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 5

print("Training started...")

for epoch in range(epochs):
    model.train()

    total_loss = 0
    for i in range(len(train_images)):
        img = train_images[i].unsqueeze(0)
        mask = train_masks[i].unsqueeze(0)

        pred = model(img)
        loss = criterion(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss / len(train_images)}")
    
    torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

print("Training complete!")
