import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import InfectionDataset
from model import AttentionUNet
from tqdm import tqdm

DATA_ROOT = "Infection Segmentation Data"
TEST_DIR = os.path.join(DATA_ROOT, "test")
MODEL_PATH = "infection_attention_unet.pth"
SAVE_DIR = "test_results"
IMG_SIZE = 256

os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

def dice_iou(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * inter + eps) / (union + eps)

    iou = (inter + eps) / (pred.sum() + target.sum() - inter + eps)
    return dice.item(), iou.item()

test_ds = InfectionDataset(TEST_DIR, split="val")
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

model = AttentionUNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print(f"[TEST] Loaded {len(test_ds)} samples")

dice_scores = []
iou_scores = []

with torch.no_grad():
    for idx, (img, gt) in enumerate(tqdm(test_loader, desc="Testing")):
        img = img.to(device)
        gt = gt.to(device)

        pred = model(img)
        pred_bin = (pred > 0.5).float()

        dice, iou = dice_iou(pred_bin, gt)
        dice_scores.append(dice)
        iou_scores.append(iou)

        img_np = img[0, 0].cpu().numpy() * 255
        gt_np = gt[0, 0].cpu().numpy() * 255
        pred_np = pred_bin[0, 0].cpu().numpy() * 255

        img_np = img_np.astype(np.uint8)
        gt_np = gt_np.astype(np.uint8)
        pred_np = pred_np.astype(np.uint8)

        overlay = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        overlay[gt_np > 0] = [0, 255, 0]      
        overlay[pred_np > 0] = [0, 0, 255]    

        cv2.imwrite(f"{SAVE_DIR}/img_{idx}_image.png", img_np)
        cv2.imwrite(f"{SAVE_DIR}/img_{idx}_gt.png", gt_np)
        cv2.imwrite(f"{SAVE_DIR}/img_{idx}_pred.png", pred_np)
        cv2.imwrite(f"{SAVE_DIR}/img_{idx}_overlay.png", overlay)

if len(dice_scores) > 0:
    print("========== TEST RESULT ==========")
    print(f"Mean Dice : {np.mean(dice_scores):.4f}")
    print(f"Mean IoU  : {np.mean(iou_scores):.4f}")
    print(f"Saved results : {SAVE_DIR}")
else:
    print("No test samples found")
