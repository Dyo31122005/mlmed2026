import torch
import torch.nn as nn
import csv
import os
from torch.utils.data import DataLoader
from dataset import InfectionDataset
from model import AttentionUNet
from losses import DiceBCELoss
from tqdm import tqdm
import torch.multiprocessing as mp

DATA_ROOT = "Infection Segmentation Data"
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
CSV_LOG = "training_log.csv"


def dice_per_sample(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    B = preds.size(0)
    dices = []

    for i in range(B):
        p = preds[i].view(-1)
        t = target[i].view(-1)

        if t.sum() == 0:
            dice = torch.tensor(
                1.0 if p.sum() == 0 else 0.0,
                device=logits.device
            )
        else:
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            dice = (2 * inter + eps) / (union + eps)

        dices.append(dice)

    return dices   


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = InfectionDataset(
        os.path.join(DATA_ROOT, "train"),
        split="train"
    )
    val_ds = InfectionDataset(
        os.path.join(DATA_ROOT, "val"),
        split="val"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = AttentionUNet().to(device)
    criterion = DiceBCELoss(
        dice_weight=0.5,
        bce_weight=0.5,
        pos_weight=5.0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    with open(CSV_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_dice",
            "val_loss",
            "val_covid_dice",
            "val_noncovid_dice"
        ])

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training Epochs")

    for epoch in epoch_bar:

        model.train()
        train_loss = 0.0
        train_dices = []

        for x, y in tqdm(
            train_loader,
            leave=False,
            desc=f"Epoch {epoch}/{EPOCHS} [TRAIN]"
        ):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dices.extend(dice_per_sample(logits, y))

        train_loss /= len(train_loader)
        train_dice = (
            torch.stack(train_dices).mean().item()
            if len(train_dices) > 0 else 0.0
        )

        model.eval()
        val_loss = 0.0
        covid_dices = []
        noncovid_dices = []

        with torch.no_grad():
            for x, y in tqdm(
                val_loader,
                leave=False,
                desc=f"Epoch {epoch}/{EPOCHS} [VAL]"
            ):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item()
                dices = dice_per_sample(logits, y)

                for d, t in zip(dices, y):
                    if t.sum() > 0:
                        covid_dices.append(d)
                    else:
                        noncovid_dices.append(d)

        val_loss /= len(val_loader)

        covid_dice = (
            torch.stack(covid_dices).mean().item()
            if len(covid_dices) > 0 else 0.0
        )

        noncovid_dice = (
            torch.stack(noncovid_dices).mean().item()
            if len(noncovid_dices) > 0 else 0.0
        )

        epoch_bar.set_postfix({
            "Train Dice": f"{train_dice:.4f}",
            "Val COVID": f"{covid_dice:.4f}",
            "Val NonCOVID": f"{noncovid_dice:.4f}"
        })

        with open(CSV_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(train_loss, 6),
                round(train_dice, 6),
                round(val_loss, 6),
                round(covid_dice, 6),
                round(noncovid_dice, 6)
            ])

    torch.save(model.state_dict(), "infection_attention_unet.pth")
    print("Training completed!")
    print(f"Metrics saved to {CSV_LOG}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
