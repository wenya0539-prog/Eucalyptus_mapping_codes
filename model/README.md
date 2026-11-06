
# Trained Random Forest Model

This directory contains the trained Random Forest model used for annual *Eucalyptus* plantation mapping across southern China from 2001 to 2020.

## File Description

| File | Description |
|---|---|
| `eucalyptus_rf_southern_china_2020.rds` | Trained Random Forest classification model saved in RDS format. |

## Model Information

- **Algorithm:** Random Forest
- **Implementation:** R (`ranger`)
- **Model type:** Classification with probability estimation
- **Training data:** Reference samples collected across the study area
- **Spatial coverage:** 11 provincial-level regions in southern China
- **Application:** Annual prediction of *Eucalyptus* plantation probability maps (2001–2020).

## Usage

The model can be loaded in R using:

```r
rf_model <- readRDS(
  "model/eucalyptus_rf_southern_china_2020.rds"
)
```

The model is used by `05_annual_prediction.R` to generate annual probability and initial classification maps.

**Important:** The predictor names and their order must be consistent with those used during model training and annual feature extraction. In particular, the vegetation index `NRGDI` is defined as:

\[
NRGDI = \frac{Red-Green}{Red+Green}
\]

## File Storage

The model file is tracked using Git Large File Storage (Git LFS). Make sure Git LFS is installed and enabled when cloning or downloading the repository.

