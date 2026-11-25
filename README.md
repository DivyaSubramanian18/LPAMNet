# LPAMNet
Objective: To achieve accurate and robust colorectal polyp segmentation by enhancing low-contrast regions and improving segmentation reliability under challenging image conditions.

## Repository Contents and Reproducibility

To facilitate transparency and reproducibility, we provide the dataset preparation procedure, the resulting dataset partitions, and the complete implementation of the proposed LPAMNet framework.

### 1. Dataset Preparation and Leakage-Free Splitting

**File:** `Dataset_Split.py`

This file describes the procedure used to combine the Kvasir-SEG and CVC datasets into a unified dataset. Duplicate and near-duplicate images are identified and handled before partitioning to minimize potential data leakage between the training, validation, and testing subsets.

The resulting leakage-controlled dataset is saved as **PolypMS**, which contains the following three mutually exclusive partitions:

| Dataset split | Number of images |
|---|---:|
| Training | 1,128 |
| Validation | 241 |
| Testing | 243 |
| **Total** | **1,612** |

The PolypMS dataset is organized into separate training, validation, and testing folders, each containing the corresponding images and ground-truth masks. These partitions are used consistently in the reported experiments.

### 2. Complete LPAMNet Implementation

**File:** `Full_Code.py`

This file provides the complete implementation of the proposed LPAMNet framework and the associated experimental procedures. It includes the model architecture and its constituent modules, including Channel Attention, the complete loss function, data loading and preprocessing, training procedure, and evaluation methodology.

The implementation is provided to enable independent examination of the proposed architecture and reproduction of the experiments reported in the manuscript.

### Reproducibility

The dataset preparation procedure, leakage-controlled PolypMS dataset partitions, and complete LPAMNet implementation are made publicly available to improve the transparency, accessibility, and reproducibility of the proposed approach.
Datasets
The framework was evaluated using the following publicly available datasets:
 Kvasir-SEG: https://datasets.simula.no/kvasir-seg/
 CVC-ClinicDB: https://polyp.grand-challenge.org/CVCClinicDB/
The datasets are not included in this repository and must be downloaded from their respective official sources.


