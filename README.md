# LPAMNet
This repository contains the core implementation of LPAMNet, a fault-tolerant deep learning framework for colorectal polyp segmentation.

The repository provides the implementation of the following proposed components:

DEM (Detail Enhancement Module): Identifies low-contrast regions using a Contrast Feature Importance (CFI) map and enhances their feature representation.

LPAM (Low-Contrast Polyp Attention Module): Uses CFI-guided feature amplification to emphasize low-contrast regions.

HiFiDecoder: Refines segmentation features using distance-guided feature modulation.

LPAMNet: Integrates the proposed modules into the overall segmentation architecture.

Loss Functions: Provides the loss functions used in the proposed framework.

Datasets

The framework was evaluated using the following publicly available datasets:

Kvasir-SEG: https://datasets.simula.no/kvasir-seg/

CVC-ClinicDB: https://polyp.grand-challenge.org/CVCClinicDB/

The datasets are not included in this repository and must be downloaded from their respective official sources.

Repository Contents

The repository includes the source code for the proposed network architecture, its individual modules, and the associated loss functions.
