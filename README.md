# A systematic benchmarking framework and dual-view optimization strategy for single-cell DNA methylation imputation

This repository provides the code, example workflows, and reproducibility resources for our systematic benchmark of single-cell DNA methylation (scDNAm) imputation methods and the proposed dual-view optimization strategies.

scDNAm data are highly sparse, which substantially limits accurate methylation-state recovery and downstream analysis. To provide practical guidance for method selection, we benchmarked five representative scDNAm imputation methods across 13 real-world datasets and evaluated their performance from seven perspectives: prediction accuracy, sensitivity to intrinsic data characteristics, scalability, robustness to data splitting strategies, inter-dataset generalizability, convergence behavior, and computational efficiency.

Based on the benchmarking results, we further developed two complementary optimization strategies. From the model view, **BridgeCpG** uses a LightGBM-based meta-learner to integrate the complementary strengths of CpG Transformer, GraphCpG, and MambaCpG. From the data view, an adaptive **divide-and-conquer (D&C)** strategy partitions heterogeneous datasets into smaller subsets with similar cell-level methylation and coverage profiles before cluster-wise imputation.

This repository includes scripts for data preprocessing, running baseline methods, reproducing **benchmark experiments**, applying **BridgeCpG**, and performing the **adaptive D&C strategy**. 

**A lightweight toy example** is also provided to demonstrate how to run the **BridgeCpG pipeline** on a small subset of cells and CpG sites.

![Overview](all_overview.png)

## Repository structure

```text
.
├── CODE/
│   ├── DeepCpG/                              # DeepCpG preprocessing, training, and imputation
│   ├── CaMelia/
│   │   ├── CaMelia_model_origin/             # Original CaMelia cross-validation workflow
│   │   └── CaMelia_chromesome_based/         # Chromosome-based CaMelia workflow
│   ├── cpg_transformer/                      # CpG Transformer preprocessing, training, and imputation
│   ├── GraphCpG/                             # GraphCpG training and imputation
│   ├── MambaCpG/                             # MambaCpG training and imputation
│   ├── BridgeCpG/
│   │   ├── CpG_Transformer_module/           # CpG Transformer module used by BridgeCpG
│   │   ├── GraphCpG_module/                  # GraphCpG module used by BridgeCpG
│   │   ├── MambaCpG_module/                  # MambaCpG module used by BridgeCpG
│   │   └── Ensemble/                         # LightGBM ensemble workflow and toy example
│   └── Divide_and_Conquer/                   # Adaptive divide-and-conquer workflow
├── all_overview.png
├── LICENSE
└── README.md
```

The paths shown above match the directory names included in this repository.

## Data preparation

For each dataset, the basic input files include:

- Tab-separated methylation files (`.tsv`) containing observed methylation states for each cell
- Corresponding genomic FASTA files (`.fa`) for sequence-based models

Different methods require different processed formats. In particular, CpG Transformer, GraphCpG, and MambaCpG use the unified `.npz` format generated during the CpG Transformer preprocessing step. Detailed preprocessing scripts are provided in the corresponding method folders.

## Running the methods

This repository contains three major components:

1. Workflows for the five benchmarked scDNAm imputation methods
2. The BridgeCpG ensemble-learning pipeline
3. The adaptive divide-and-conquer strategy

Please configure the environment of each method according to its corresponding `requirements.txt` file. Because different baseline methods depend on different deep-learning framework versions, we recommend using separate conda environments for different methods to ensure reproducibility.

## Run benchmark models

### DeepCpG

- **Installation**: DeepCpG can be installed as a Python package. The required environment is provided in `CODE/DeepCpG/requirements.txt`.
- **Usage**: Example scripts are provided in `CODE/DeepCpG/` for preprocessing, training, and imputation:
  - `data.py`: data preprocessing
  - `run.sh`: model training and imputation

Please modify the input paths and parameters in these scripts according to your dataset.

### CaMelia

- **Installation**: Configure the environment using the `requirements.txt` file in the corresponding method folder.
- **Usage**: Two workflows are provided:
  - `CODE/CaMelia/CaMelia_model_origin/`: the original cross-validation workflow
  - `CODE/CaMelia/CaMelia_chromesome_based/`: the chromosome-based hold-out workflow used in our benchmark
  - Use `CODE/CaMelia/run.py` as a reference script for model execution.

### CpG Transformer

- **Preprocessing**: CpG Transformer preprocessing is used to generate the unified `.npz` input files required by CpG Transformer, GraphCpG, and MambaCpG. The `X.npz` file should be generated from the corresponding genomic FASTA file.
- **Genome references**: Pre-built reference files for mm9, mm10, hg19, and hg38 are available via Google Drive: [reference files](https://drive.google.com/drive/folders/1NdQcvru1X7ttNcNy_RUfs4wm4gvHqe4R?usp=sharing).
- **Usage**: A complete preprocessing-to-imputation example is provided in `CODE/cpg_transformer/run.sh`.

### GraphCpG

- **Installation**: Configure the environment using the `requirements.txt` file in the corresponding method folder. To improve reproducibility, we provide a tested environment configuration used in our benchmark.
- **Preprocessing**: Follow the CpG Transformer preprocessing workflow to generate the required `.npz` files.
- **Training and imputation**: Run `CODE/GraphCpG/train_graph_cpg.py` to train the model and generate imputation results.

### MambaCpG

- **Installation**: Configure the environment using the `requirements.txt` file in the corresponding method folder.
- **Preprocessing**: Follow the CpG Transformer preprocessing workflow to generate the required `.npz` files.
- **Usage**: Refer to `CODE/MambaCpG/run.sh` for model training and imputation.

## Run BridgeCpG

BridgeCpG is a LightGBM-based ensemble framework that integrates predictions from CpG Transformer, GraphCpG, and MambaCpG. The meta-learner dynamically combines base-model predictions and auxiliary features to improve imputation robustness while retaining feature-importance-based interpretability.

### Step 1: Run the three base imputation modules

First, run the CpG Transformer preprocessing workflow to generate the unified `.npz` files required by the three base models. All base-model outputs were generated from identical data groups and evaluation masks.

1. Train CpG Transformer module and generate:
   - the prediction file `CpGTransformer.csv`
   - the checkpoint file used to record the masked or random site indices
2. Train MambaCpG module using the same `.npz` files and checkpoint information, and generate `MambaCpG.csv`.
3. Run `CODE/BridgeCpG/GraphCpG_module/ckpt_change.py` to convert the checkpoint for GraphCpG compatibility, then train GraphCpG module and generate `GraphCpG.csv`.
4. Run `CODE/BridgeCpG/Ensemble/script/data_alignment_check.py` to validate structural consistency of three base-model outputs and generate audit reports; only passed results can be used for BridgeCpG ensemble training.

### Step 2: Run the LightGBM-based ensemble module

Configure this module using `CODE/BridgeCpG/Ensemble/requirements.txt`. All BridgeCpG-related files are located in the `CODE/BridgeCpG/Ensemble/` folder.

### Toy example

A lightweight toy example is provided to demonstrate the complete BridgeCpG workflow on a small subset of cells and CpG sites. This example is intended for quick testing of the environment, input format, base-model prediction files, and LightGBM-based ensemble module.

Please see:

```text
CODE/BridgeCpG/Ensemble/Example.ipynb
```

Equivalent Python scripts are also provided to improve readability and facilitate command-line execution.

## Run the divide-and-conquer clustering workflow

Configure this workflow using `CODE/Divide_and_Conquer/requirements.txt`. The divide-and-conquer workflow partitions an scDNAm dataset into subsets before cluster-wise imputation.

In the current implementation, each cell is represented by its global methylation rate and log-transformed observed CpG count. Ward hierarchical clustering is then applied to group cells with similar methylation and coverage profiles.

For the standard workflow, cluster sizes are constrained to an adjustable range of 10–100 cells. Cells belonging to clusters smaller than the minimum size are reassigned to the nearest eligible cluster in the weighted feature space, whereas clusters larger than the maximum size are recursively subdivided until no cluster exceeds the upper limit.

### Step 1: Compute cell-level statistics

Run `CODE/Divide_and_Conquer/stat.py` to calculate the cell-level methylation and coverage statistics used for clustering.

### Step 2: Perform hierarchical clustering

Select the script according to the intended analysis:

- `CODE/Divide_and_Conquer/cluster_for_standard_dataset.py`: standard workflow with adjustable lower and upper cluster-size limits
- `CODE/Divide_and_Conquer/cluster_without_predefined_constraints.py`: workflow without the standard 10–100 cell-size range, used for the corresponding small-dataset analysis

Example output tables are provided as `example_on_hEmbryo_standard_dataset.csv` and `example_on_mESc_2i_without_district.csv`.

Before downstream imputation, users are encouraged to inspect the resulting cluster sizes and record the parameter settings used for each dataset.

All scripts related to this workflow are located in the `CODE/Divide_and_Conquer/` folder.

> Note: These scripts implement the clustering and partitioning stage only. The resulting clusters are intended to facilitate cluster-wise imputation, but performance improvements may vary across datasets and downstream imputation models.

## Reproducibility notes

Different baseline methods rely on different software environments and deep-learning framework versions. To ensure reproducibility, we provide method-specific environment files whenever possible. Runtime and memory usage may vary across CUDA, PyTorch, TensorFlow, compiler settings, and hardware configurations. Therefore, we recommend creating separate environments for different methods and using the reported framework versions when reproducing the benchmark results.

In addition, CaMelia uses bulk methylation profiles as supplementary input, which are not required by the other methods and may provide an advantage in direct performance comparisons. Therefore, its results should be interpreted with this input difference in mind.



