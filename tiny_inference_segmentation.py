import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from unet import TinyUNet

class TinySegmenter:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TinyUNet().to(self.device)
        self.model.eval()

        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✅ Segmentation model loaded.")
        else:
            print("⚠️ No valid segmentation model found.")

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            output = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (output > 0.5).astype(np.uint8) * 255
            return Image.fromarray(mask)
