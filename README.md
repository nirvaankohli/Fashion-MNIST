# Fashion-MNIST Classification with Transfer Learning and Advanced Augmentations

## Overview

This repository demonstrates three iterations of a Fashion-MNIST classification pipeline using PyTorch, pretrained backbones, and modern training techniques. Each version (V1, V2, V3) builds on the previous, introducing new augmentations, schedulers, regularization, and logging to squeeze maximum performance on a more challenging dataset than the classic MNIST.

## What Makes Fashion-MNIST Different

- **More Complex Classes**  
  Fashion-MNIST consists of 28×28 grayscale images of 10 clothing categories (e.g. T-shirt, Trouser, Bag). Unlike MNIST digits, these items exhibit richer intra-class variability (different poses, textures) and inter-class similarity (e.g. shirts vs. pullovers), making classification substantially harder.
- **Higher Difficulty**  
  Benchmarks on standard MNIST regularly top 99.5% accuracy with simple CNNs. On Fashion-MNIST, even deep CNNs hover around 93–94% and require stronger augmentations and regularization to break 95%+.

## How We “Hurdled” the Challenges

1. **Pretrained Backbones**  
   - **V1**: MobileNetV2 with modified first conv-layer for 3-channel input  
   - **V2 & V3**: EfficientNet-B0 for higher capacity and better feature extraction  
2. **Strong Data Augmentation**  
   - Grayscale → 3-channel conversion  
   - Random crops, horizontal flips, rotations, affine transforms, perspective  
   - RandAugment (in V3) for automated, diverse augmentation policies  
   - RandomErasing to simulate occlusions  
3. **MixUp & CutMix**  
   - On-the-fly sample interpolation and patch mixing to regularize and enrich training distribution  
4. **Label Smoothing**  
   - Softens hard targets to improve calibration and generalization  
5. **Advanced Schedulers & SWA**  
   - **V1**: CosineAnnealingLR + Stochastic Weight Averaging (SWA)  
   - **V2 & V3**: OneCycleLR + SWA  
6. **Early Stopping & Checkpointing** (V3)  
   - Monitors validation accuracy to halt training when improvements plateau, saving compute  
7. **Extended Metrics**  
   - Top-1 & Top-3 accuracy (V3 via torchmetrics)  
   - TensorBoard scalars + CSV logging for offline analysis  


## Usage

Train any version with:

```bash
python V1.py --data-dir ./data --batch-size 128 --epochs 50 --lr 0.1
python V2.py --data-dir ./data --batch-size 128 --epochs 50 --lr 1e-3
python V3.py --data-dir ./data --output-dir ./outputs --batch-size 128 --epochs 50 --lr 1e-3
```

Additional flags:

* `--mixup-alpha`, `--cutmix-alpha`, `--cutmix-prob` to tune mixup/cutmix
* `--swa-start` to delay SWA start epoch
* `--patience` (V3) for early stopping

## Results

Each run outputs:

* A `.pth` model checkpoint (`best_*.pth`, `*_swa_model.pth`)
* A metrics CSV (`V*_metrics.csv`) logging per-epoch train/val loss & accuracy
* TensorBoard logs for visualizing learning curves

Typical best validation accuracies:

* **V1 (MobileNetV2)**: \~89%
* **V2 (EfficientNet-B0)**: \~92%
* **V3 (+ RandAugment, EarlyStopping)**: \~93%

#### This is part of Shipwrecked

<div align="center">
  <a href="https://shipwrecked.hackclub.com/?t=ghrm" target="_blank">
    <img src="https://hc-cdn.hel1.your-objectstorage.com/s/v3/739361f1d440b17fc9e2f74e49fc185d86cbec14_badge.png" 
         alt="This project is part of Shipwrecked, the world's first hackathon on an island!" 
         style="width: 70%;">
  </a>
</div>
