import torch
from loader import load_dataset
from model import UNet
import torch.nn as nn
import torch.optim as optim

root = "dataset_split"

train_images, train_masks = load_dataset(root, "train")
val_images, val_masks = load_dataset(root, "val")

if train_images is None:
    raise Exception("Training data not found!")

train_images = train_images.float()
train_masks = train_masks.float()

model = UNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(train_images)):
        img = train_images[i].unsqueeze(0)
        mask = train_masks[i].unsqueeze(0)

        output = model(img)
        loss = criterion(output, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss / len(train_images)}")
