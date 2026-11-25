# EuroSAT Land-Cover Classification from 13-Band Satellite Images

[![tests](https://github.com/ZIADAYMAN81/eurosat-multispectral-classification/actions/workflows/tests.yml/badge.svg)](https://github.com/ZIADAYMAN81/eurosat-multispectral-classification/actions/workflows/tests.yml)

> Deep Learning coursework, The British University in Egypt, 2025. Author: Ziad Ayman.

Classifies **Sentinel-2 satellite images** into 10 land-cover types (crops, forest, highway, industrial, residential, river, sea and more) using **all 13 spectral bands**, not just red, green and blue. It compares the fully connected deep network explored in the coursework with a compact convolutional network added in this version, both in **PyTorch**.

## Results (held-out test set, 2,759 images)

| Model | Parameters | **Test accuracy** | Weighted F1 | Macro ROC-AUC | CPU training time |
|---|---|---|---|---|---|
| **Spectral CNN** (added) | **0.30 million** | **98.7%** | **0.987** | **0.9997** | 54 min |
| Deep fully connected network (2048-1024) | 111 million | 85.2% | 0.850 | 0.983 | 72 min |

**The convolutional network is 13.5 percentage points more accurate with about 370 times fewer parameters.** Flattening a 13 × 64 × 64 image into 53,248 numbers discards which pixels are next to each other; the CNN keeps that spatial structure and learns texture (field patterns, road lines, building blocks). Every class reaches at least 97% F1 with the CNN; full per-class reports and confusion matrices are in [`results/`](results/).

### Coursework comparison of network sizes
The original coursework compared five fully connected networks (20 epochs, Adam):

| Hidden layers | Learning rate | Test accuracy | Weighted F1 |
|---|---|---|---|
| 4096-2048 | 0.001 | 80.6% | 0.802 |
| **2048-1024** | **0.0005** | **82.9%** | **0.829** |
| 8192-4096 | 0.001 | 80.8% | 0.804 |
| 4096-4096 | 0.001 | 80.9% | 0.808 |
| 2048-4096 | 0.0005 | 81.7% | 0.822 |

Wider networks did not help, so the best configuration (2048-1024) was retrained here with AdamW, cosine learning-rate decay and best-validation checkpointing, which raised it from 82.9% to 85.2%.

## What was improved in this version
- **Accurate model naming.** The coursework notebook called the network a "Deep Boltzmann Machine", but it is a fully connected network trained with backpropagation, so it is named `DeepMLP` here.
- **Correct ROC-AUC.** The original computed AUC from hard class predictions; it is now computed from predicted probabilities (one-vs-rest, macro-averaged).
- **New convolutional model** (`SpectralCNN`): a 1 × 1 convolution mixes the 13 bands per pixel, followed by three convolutional blocks with batch normalisation and global average pooling.
- **Fast data pipeline.** The 27,000 GeoTIFFs are decoded once into a compact `uint16` cache and read through memory-mapped arrays, instead of re-reading every file each epoch.
- **Reproducible training script** with seeding, best-validation checkpointing, optional early stopping, and automatic export of metrics, training history, per-class report and confusion matrix.
- **Unit tests** for the models, augmentation and metrics.

## Data
[EuroSAT](https://github.com/phelber/EuroSAT) "all bands" release: 27,000 labelled 64 × 64 Sentinel-2 tiles across 10 classes, split into 19,317 training, 5,519 validation and 2,759 test images. Training uses flips and 90-degree rotations (satellite tiles have no "up").

## Getting started
```bash
pip install -r requirements.txt

# point --data-root at the folder containing train.csv, validation.csv, test.csv and the class folders
python -m eurosat.train --data-root path/to/EuroSATallBands --model cnn --epochs 20 --patience 6
python -m eurosat.train --data-root path/to/EuroSATallBands --model mlp --hidden 2048 1024 --lr 5e-4 --weight-decay 0

python -m pytest -q
```
The first run builds the cache in `data/cache/` (about 2.9 GB); later runs start immediately.

## Project structure
```
eurosat/
├── data.py     # GeoTIFF decoding, uint16 cache, dataset and augmentation
├── models.py   # DeepMLP and SpectralCNN
└── train.py    # training, checkpointing, evaluation and exports (command line)
results/        # metrics, history, per-class reports and confusion matrices for both models
tests/
```

## Tech stack
Python · PyTorch · NumPy · pandas · scikit-learn · rasterio · pytest

## License
MIT, see [LICENSE](LICENSE).
