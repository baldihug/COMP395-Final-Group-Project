# COMP395-Final-Group-Project

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/baldihug/COMP395-Final-Group-Project.git
cd COMP395-Final-Group-Project
```

### 2. Download the dataset

The dataset is hosted on Zenodo (too large for GitHub). Download and extract it into the `data/` directory:

```bash
mkdir -p data
cd data
wget https://zenodo.org/records/12825163/files/nsforcing_128.tgz
tar -xzf nsforcing_128.tgz
cd ..
```

This produces:
- `data/nsforcing_train_128.pt` — training set (1.3 GB, 10,000 samples)
- `data/nsforcing_test_128.pt` — test set (251 MB)

### 3. Install dependencies

```bash
uv sync
```

---

## References

- NeuralOperator repo (Li et al. 2021, source of dataset): https://github.com/neuraloperator/neuraloperator
- Dataset DOI: https://zenodo.org/records/12825163
- HUFNO repo (Li et al. data): https://github.com/esaka-forever/HUFNO
- FourierFlow readme: https://github.com/alasdairtran/fourierflow/blob/main/README.md
