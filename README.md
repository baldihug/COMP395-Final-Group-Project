# COMP395 Final Project — FNO vs U-Net on 2D Navier-Stokes

Comparing the Fourier Neural Operator (FNO) and U-Net on predicting the time evolution of 2D incompressible Navier-Stokes equations.

## Repository Structure

```
docs/                  — reports, presentation, and project log
  report.tex / .pdf        technical report
  explanation.tex / .pdf   plain-English companion report
  presentation.html        reveal.js slideshow (open in any browser)
  project_log.txt          full session-by-session project log
  COMP 395 Final - Latex Report Draft.pdf
  CS_395_2_Deep_Learning_Final_Project (3).pdf

models/                — model definitions
  fno.py                   Fourier Neural Operator (4.7M params)
  unet.py                  U-Net baseline (31M params)

figures/               — all generated figures
  predictions.png          side-by-side FNO vs U-Net predictions
  error_maps.png           absolute error heatmaps
  super_res.png            zero-shot super-resolution comparison
  error_histogram.png      per-sample error distribution
  sample_pairs.png         dataset input/target examples
  sample_diffs.png         target minus input visualisation
  stats.png                dataset statistics

checkpoints/           — best model weights (gitignored)
  fno_best.pt
  unet_best.pt

data/                  — dataset files (gitignored, download below)
  nsforcing_train_128.pt
  nsforcing_test_128.pt

dataset.py             — PyTorch Dataset wrapper
train.py               — training loop (relative L2, cosine LR, MLflow)
evaluate.py            — evaluation: accuracy, throughput, super-resolution
visualize_data.py      — dataset visualisation figures
visualize_results.py   — model prediction and error figures
main.py                — entry point
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/baldihug/COMP395-Final-Group-Project.git
cd COMP395-Final-Group-Project
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Download the dataset

The dataset is hosted on Zenodo (too large for GitHub):

```bash
mkdir -p data && cd data
wget https://zenodo.org/records/12825163/files/nsforcing_128.tgz
tar -xzf nsforcing_128.tgz
cd ..
```

This produces:
- `data/nsforcing_train_128.pt` — training set (1.3 GB, 10,000 samples)
- `data/nsforcing_test_128.pt` — test set (251 MB, 2,000 samples)

## Usage

```bash
# Train
uv run python train.py --model fno
uv run python train.py --model unet

# Evaluate (accuracy + speed)
uv run python evaluate.py --model fno  --checkpoint checkpoints/fno_best.pt
uv run python evaluate.py --model unet --checkpoint checkpoints/unet_best.pt

# Zero-shot super-resolution
uv run python evaluate.py --model fno  --checkpoint checkpoints/fno_best.pt  --super_res
uv run python evaluate.py --model unet --checkpoint checkpoints/unet_best.pt --super_res

# Regenerate result figures
uv run python visualize_results.py

# Compile reports (ACM format — bibtex pass required)
cd docs
pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex
pdflatex explanation.tex && pdflatex explanation.tex && pdflatex explanation.tex
```

## Results

| Model  | Params | Test Rel-L2 | Throughput (samp/s) | Latency (ms/samp) |
|--------|--------|-------------|---------------------|-------------------|
| FNO    | 4.7M   | **0.0348**  | **499.2**           | **2.003**         |
| U-Net  | 31M    | 0.1029      | 227.0               | 4.406             |

Zero-shot super-resolution (trained only at 128×128):

| Resolution      | FNO Rel-L2   | U-Net Rel-L2 |
|-----------------|--------------|--------------|
| 64×64           | **0.0185**   | 0.4764       |
| 128×128 (train) | **0.0190**   | 0.1029       |
| 256×256         | **0.0188**   | 0.5941       |

## References

- Li et al. (2021) — [Fourier Neural Operator for Parametric PDEs](https://arxiv.org/abs/2010.08895)
- Ronneberger et al. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation
- Dataset DOI: [10.5281/zenodo.12825163](https://zenodo.org/records/12825163)
- NeuralOperator repo: https://github.com/neuraloperator/neuraloperator
