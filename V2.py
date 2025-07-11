#!/usr/bin/env python3
import os
import argparse
from multiprocessing import freeze_support

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from tqdm import tqdm
import torch

def get_data_loaders(
        
        data_dir, 
        
        batch_size=128, 
        
        num_workers=4, 
        
        device=None):
    
    mean, std = [0.2860]*3, [0.3530]*3

    pin = device is not None and device.type == "cuda"

    train_tf = transforms.Compose([

        transforms.Grayscale(num_output_channels=3),

        transforms.Pad(4),

        transforms.RandomCrop(28),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(15),

        transforms.RandomAffine(

            0, 
            
            translate=(

                0.1,
                
                0.1
                
                ), 
                
            scale=(
                
                0.9,

                1.1
                
                ), 

                shear=10
                
                ),
        
        transforms.RandomPerspective(
            
            distortion_scale=0.2, 
            
            p=0.5
            
            ),
        
        transforms.ToTensor(),

        transforms.Normalize(
            
            mean,

            std
            
            ),

        transforms.RandomErasing(
            
            p=0.5, 
            
            scale=(
                
                 0.02,
                 
                 0.1
                 
                 ), ratio=(0.3,3.3), value='random'),

    ])
    test_tf = transforms.Compose([

        transforms.Grayscale(num_output_channels=3),

        transforms.ToTensor(),

        transforms.Normalize(mean, std),

    ])

    train_ds = datasets.FashionMNIST(
        
        data_dir, 
        
        train=True,  
        
        download=True, 
        
        transform=train_tf
        
        )
    
    test_ds  = datasets.FashionMNIST(
        
        data_dir, 
        
        train=False, 
        
        download=True, 
        
        transform=test_tf
        
        )

    train_loader = DataLoader(
        
        train_ds, 
        
        batch_size=batch_size, 
        
        shuffle=True,
        
        num_workers=num_workers, 
        
        pin_memory=pin, 
        
        persistent_workers=True
        
        )
    
    test_loader  = DataLoader(

        test_ds,  
        
        batch_size=batch_size, 
        
        shuffle=False,
        
        num_workers=num_workers, 
        
        pin_memory=pin, 
        
        persistent_workers=True
        
        )
    
    return (
        
        train_loader, 
        
        test_loader
        
        )

def make_model(
        
        num_classes=10
        
        ):
    
    # EfficientNet-B0 backbone

    model = models.efficientnet_b0(
        
        pretrained=True
        
        )
    
    # replace classifier head

    in_feat = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        
        in_feat, 
        
        num_classes
        
        )
    
    return model

def mixup_data(x, y, alpha=0.2):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def rand_bbox(size, lam):

    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w//2, 0, W)
    bby1 = np.clip(cy - cut_h//2, 0, H)
    bbx2 = np.clip(cx + cut_w//2, 0, W)
    bby2 = np.clip(cy + cut_h//2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2-bbx1)*(bby2-bby1)/(x.size(2)*x.size(3)))
    return x, y, y[idx], lam

def train_epoch(model, loader, optimizer, criterion, device, scaler,
                scheduler, epoch, total_epochs, mixup_alpha, cutmix_alpha, cutmix_prob):
    model.train()
    running_loss, running_correct, total = 0., 0., 0
    pbar = tqdm(loader, desc=f"Train [{epoch}/{total_epochs}]", ncols=120)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if np.random.rand() < cutmix_prob:
            imgs, y1, y2, lam = cutmix_data(imgs, labels, cutmix_alpha)
        else:
            imgs, y1, y2, lam = mixup_data(imgs, labels, mixup_alpha)

        with autocast():
            outputs = model(imgs)
            loss = lam * criterion(outputs, y1) + (1-lam) * criterion(outputs, y2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        preds = outputs.argmax(dim=1)
        running_loss    += loss.item() * imgs.size(0)
        running_correct += lam * (preds==y1).sum().item() + (1-lam)*(preds==y2).sum().item()
        total           += labels.size(0)

        pbar.set_postfix({
            "loss": f"{running_loss/total:.4f}",
            "acc":  f"{100*running_correct/total:.2f}%"
        })

    return running_loss/total, running_correct/total

@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    val_loss, val_correct, total = 0., 0, 0
    pbar = tqdm(loader, desc=f"Valid [{epoch}/{total_epochs}]", ncols=120)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        val_loss    += loss.item() * imgs.size(0)
        val_correct += (preds==labels).sum().item()
        total       += labels.size(0)
        pbar.set_postfix({
            "loss": f"{val_loss/total:.4f}",
            "acc":  f"{100*val_correct/total:.2f}%"
        })
    return val_loss/total, val_correct/total

def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST v2")
    parser.add_argument('--data-dir',    type=str,   default='./data')
    parser.add_argument('--batch-size',  type=int,   default=128)
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--weight-decay',type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int,   default=4)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--cutmix-alpha',type=float, default=1.0)
    parser.add_argument('--cutmix-prob', type=float, default=0.5)
    parser.add_argument('--swa-start',   type=int,   default=40)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loaders(
        args.data_dir, args.batch_size, args.num_workers, device
    )

    model     = make_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.3, anneal_strategy='cos'
    )
    scaler      = GradScaler()

    swa_model    = AveragedModel(model)
    swa_scheduler= SWALR(optimizer, swa_lr=args.lr)
    swa_start    = args.swa_start

    writer = SummaryWriter()
    metrics_file = open('V2_metrics.csv', 'w')
    metrics_file.write('Epoch,TrainLoss,TrainAcc,ValLoss,ValAcc\n')
    best_val_acc = 0.0

    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            scheduler, epoch, args.epochs,
            args.mixup_alpha, args.cutmix_alpha, args.cutmix_prob
        )
        val_loss, val_acc = validate(
            model, test_loader, criterion, device, epoch, args.epochs
        )

        # SWA update
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # log & save
        metrics_file.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},"
                           f"{val_loss:.4f},{val_acc:.4f}\n")
        metrics_file.flush()
        writer.add_scalars('Loss',     {'train':train_loss, 'val':val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train':train_acc,  'val':val_acc}, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_v2_model.pth")

        print(f"Epoch {epoch:02d}: Train L={train_loss:.4f}, A={train_acc*100:.2f}% | "
              f"Val   L={val_loss:.4f}, A={val_acc*100:.2f}%")

    # finalize SWA
    update_bn(train_loader, swa_model)
    torch.save(swa_model.module.state_dict(), "v2_swa_model.pth")

    metrics_file.close()
    writer.close()
    print(f"Training complete. Best val acc: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    freeze_support()
    main()
