import cv2
import numpy as np
import torch
from model import UNet

IMAGE_SIZE = 48  

def predict(image_path, model_path="unet_model.pth"):
    img = cv2.imread(image_path)

    if img is None:
        print("Image not found:", image_path)
        return

    # resize to training size
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # preprocess
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.tensor(img).unsqueeze(0)

    # load model
    model = UNet()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        pred = pred[0, 0].cpu().numpy()

    # binary mask
    mask = (pred > 0.5).astype("uint8") * 255

    cv2.imwrite("prediction_mask.png", mask)
    print("Saved prediction_mask.png")

if __name__ == "__main__":
    predict("sample_test_image.png")
