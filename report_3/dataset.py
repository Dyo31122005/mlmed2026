import os
import cv2
import torch
from torch.utils.data import Dataset

class InfectionDataset(Dataset):
    def __init__(self, root, split="train"):
        self.samples = []
        classes = ["COVID-19", "Non-COVID"]

        for cls in classes:
            img_dir  = os.path.join(root, cls, "images")
            mask_dir = os.path.join(root, cls, "infection masks")

            if not os.path.exists(img_dir):
                continue

            for name in os.listdir(img_dir):
                img_path  = os.path.join(img_dir, name)
                mask_path = os.path.join(mask_dir, name)

                if not os.path.exists(mask_path):
                    continue

                self.samples.append((img_path, mask_path))

        print(f"[{split.upper()}] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        image = image.astype("float32") / 255.0
        mask  = (mask > 0).astype("float32")

        image = torch.from_numpy(image).unsqueeze(0)
        mask  = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
