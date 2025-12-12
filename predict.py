import cv2
import numpy as np
import torch
from model import UNet

def predict(image_path, model_path="unet_model.pth"):
    # load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype("float32") / 255.0
    img_rgb = np.transpose(img_rgb, (2, 0, 1))
    img_tensor = torch.tensor(img_rgb).unsqueeze(0)

    # load model
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        pred = pred[0,0].cpu().numpy()

    # convert to mask
    mask = (pred > 0.5).astype("uint8") * 255

    out_path = "prediction_mask.png"
    cv2.imwrite(out_path, mask)
    print("Saved:", out_path)

if __name__ == "__main__":
    predict("sample_test_image.png")
