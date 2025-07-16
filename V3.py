#!/usr/bin/env python3
import os
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from tqdm import tqdm
import torchmetrics


class CSVLogger:

    def __init__(
            
            self, 
            
            path, 
            
            headers
            
            ):

        self.path = Path(path)
        write_hdr = not self.path.exists()

        self.f = open(
            
            self.path, 

            'a', 

            newline=''
            
            )
        
        if write_hdr:

            self.f.write(','.join(headers) + '\n')

    def log(
            
            self, 
            
            values
            
            ):

        line = ','.join(

            f"{v:.4f}" if isinstance(v, float) else str(v)

            for v in values

        )

        self.f.write(line + '\n')
        self.f.flush()

    def close(self):

        self.f.close()


class EarlyStopping:

    def __init__(
            
            self, 
            
            patience=5, 
            
            min_delta=1e-4
            
            ):

        self.patience = patience
        self.min_delta = min_delta
        self.best = -float('inf')
        self.wait = 0

    def step(
            
            self, 
            
            metric
            
            ):
        
        if metric - self.best > self.min_delta:

            self.best = metric
            self.wait = 0

            return False  
        
        # do not stop
        
        else:

            self.wait += 1
            return self.wait >= self.patience


def get_data_loaders(data_dir, batch_size, num_workers, device):
    
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    pin = (device.type == "cuda")
    persistent = (num_workers > 0)

    train_tf = transforms.Compose([

        transforms.Grayscale(3),
        transforms.Pad(4),
        transforms.RandomCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1),
                                 ratio=(0.3, 3.3), value='random'),

    ])

    test_tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=train_tf
    )
    val_ds = datasets.FashionMNIST(
        data_dir, train=False, download=True, transform=test_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent
    )
    return train_loader, val_loader


def make_model(num_classes=10, dropout_p=0.3):
    model = models.efficientnet_b0(pretrained=True)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_feat, num_classes),
    )
    return model


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def rand_bbox(size, lam):
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
    lam_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                        (x.size(2) * x.size(3)))
    return x, y, y[idx], lam_adjusted


def train_one_epoch(model, loader, opt, criterion, device, scaler,
                    scheduler, epoch, total_epochs,
                    mixup_alpha, cutmix_alpha, cutmix_prob,
                    clip_grad=None):
    model.train()
    loss_accum, correct, total = 0.0, 0.0, 0
    top3 = torchmetrics.Accuracy(
        task="multiclass", num_classes=10, top_k=3
    ).to(device)
    pbar = tqdm(loader, desc=f"Train [{epoch}/{total_epochs}]", ncols=120)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()

        if random.random() < cutmix_prob:
            imgs, y1, y2, lam = cutmix_data(imgs, labels, cutmix_alpha)
        else:
            imgs, y1, y2, lam = mixup_data(imgs, labels, mixup_alpha)

        with autocast():
            outputs = model(imgs)
            loss = lam * criterion(outputs, y1) + (1 - lam) * criterion(outputs, y2)

        scaler.scale(loss).backward()
        if clip_grad is not None:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        preds = outputs.argmax(dim=1)
        loss_accum += loss.item() * imgs.size(0)
        correct += lam * (preds == y1).sum().item() + (1 - lam) * (preds == y2).sum().item()
        total += labels.size(0)
        top3.update(outputs, labels)

        pbar.set_postfix({
            "loss": f"{loss_accum/total:.4f}",
            "acc":  f"{100 * correct/total:.2f}%",
            "top3": f"{100 * top3.compute().item():.2f}%"
        })

    return loss_accum / total, correct / total, top3.compute().item()


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    loss_accum, correct, total = 0.0, 0, 0
    top3 = torchmetrics.Accuracy(
        task="multiclass", num_classes=10, top_k=3
    ).to(device)
    pbar = tqdm(loader, desc=f"Valid [{epoch}/{total_epochs}]", ncols=120)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        loss_accum += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        top3.update(outputs, labels)

        pbar.set_postfix({
            "loss": f"{loss_accum/total:.4f}",
            "acc":  f"{100 * correct/total:.2f}%",
            "top3": f"{100 * top3.compute().item():.2f}%"
        })

    return loss_accum / total, correct / total, top3.compute().item()


def save_checkpoint(state, ckpt_path):
    torch.save(state, ckpt_path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer:
        optimizer.load_state_dict(ckpt['opt'])
    if scheduler:
        scheduler.load_state_dict(ckpt['sched'])
    if scaler:
        scaler.load_state_dict(ckpt['scaler'])
    return ckpt['epoch'], ckpt.get('best_acc', 0.0)


def main():
    parser = argparse.ArgumentParser("Fashion-MNIST v3+")
    parser.add_argument('--data-dir',     type=str,   default='./data')
    parser.add_argument('--output-dir',   type=str,   default='./outputs')
    parser.add_argument('--batch-size',   type=int,   default=128)
    parser.add_argument('--num-workers',  type=int,   default=4,
                        help='number of DataLoader workers')
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--wd',           type=float, default=1e-4)
    parser.add_argument('--mixup-alpha',  type=float, default=0.2)
    parser.add_argument('--cutmix-alpha', type=float, default=1.0)
    parser.add_argument('--cutmix-prob',  type=float, default=0.5)
    parser.add_argument('--swa-start',    type=int,   default=30)
    parser.add_argument('--patience',     type=int,   default=5)
    parser.add_argument('--clip-grad',    type=float, default=1.0)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--resume',       type=str,   default=None,
                        help="path to checkpoint to resume")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader = get_data_loaders(
        args.data_dir, args.batch_size, args.num_workers, device
    )

    model     = make_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.wd)
    total_steps = args.epochs * len(train_loader)
    scheduler   = OneCycleLR(optimizer, max_lr=args.lr,
                             total_steps=total_steps, pct_start=0.3,
                             anneal_strategy='cos')
    scaler        = GradScaler()
    swa_model     = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.lr)
    early_stop    = EarlyStopping(patience=args.patience)
    writer        = SummaryWriter(log_dir=args.output_dir)
    csv_logger    = CSVLogger(
        Path(args.output_dir) / 'V3_metrics.csv',
        ['epoch','train_loss','train_acc','train_top3',
         'val_loss','val_acc','val_top3']
    )

    start_epoch, best_acc = 1, 0.0
    if args.resume is not None:
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        print(f"Resumed from epoch {start_epoch-1}, best_acc={best_acc:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_acc, tr_top3 = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler, scheduler, epoch, args.epochs,
            args.mixup_alpha, args.cutmix_alpha, args.cutmix_prob,
            clip_grad=args.clip_grad
        )
        val_loss, val_acc, val_top3 = validate(
            model, val_loader, criterion, device, epoch, args.epochs
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                {
                    'epoch':      epoch,
                    'model':      model.state_dict(),
                    'opt':        optimizer.state_dict(),
                    'sched':      scheduler.state_dict(),
                    'scaler':     scaler.state_dict(),
                    'best_acc':   best_acc,
                },
                Path(args.output_dir) / 'best_V3_model.pth'
            )
            print(f"Epoch {epoch:02d}: 🎉 New best val_acc = {best_acc*100:.2f}%, saved best_V3_model.pth")


        if epoch > args.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        writer.add_scalars('Loss',   {'train': tr_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Acc@1',  {'train': tr_acc,  'val': val_acc},  epoch)
        writer.add_scalars('Acc@3',  {'train': tr_top3,'val': val_top3}, epoch)
        csv_logger.log([
            epoch, tr_loss, tr_acc, tr_top3,
            val_loss, val_acc, val_top3
        ])

        print(
            f"Epoch {epoch:02d} | "
            f"Train L={tr_loss:.4f} A1={tr_acc*100:.2f}% A3={tr_top3*100:.2f}% | "
            f"Val   L={val_loss:.4f} A1={val_acc*100:.2f}% A3={val_top3*100:.2f}%"
        )

        if early_stop.step(val_acc):
            print(f"No improvement for {args.patience} epochs, stopping.")
            break

    update_bn(train_loader, swa_model)
    torch.save(swa_model.module.state_dict(),
               Path(args.output_dir) / 'V3_swa_model.pth')

    csv_logger.close()
    writer.close()
    print(f"Done. Best val acc: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
