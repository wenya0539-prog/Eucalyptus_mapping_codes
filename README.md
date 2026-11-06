# Annual Eucalyptus Plantation Mapping Codes (2001–2020)

This repository contains the scripts and trained Random Forest model used to produce annual 30-m *Eucalyptus* plantation maps across southern China from 2001 to 2020.

The workflow integrates Landsat time-series imagery, spectral and textural features, reference samples, Random Forest classification, provincial area constraints, and temporal post-processing.

## 1. Repository Structure

```text
Eucalyptus_mapping_codes/
│
├── README.md
├── .gitattributes
│
├── model/
│   ├── eucalyptus_rf_southern_china_2020.rds
│   └── README.md
│
├── scripts/
│   ├── 01_area_reconstruction.R
│   ├── 02_GEE_feature_extraction.js
│   ├── 03_sample_extraction.R
│   ├── 04_random_forest_training.R
│   ├── 05_annual_prediction.R
│   ├── README.md
│   │
│   └── postprocessing/
│       ├── 001_reproject_to_south_china_albers.py
│       ├── 002_temporal_postprocessing_albers_neighbor.py
│       ├── 003_merge_11_provinces.py
│       ├── 004_compress_probability_to_netcdf.py
│       └── README.md
│
└── evaluation/
    ├── rf_feature_importance.csv
    ├── regional_accuracy_table.csv
    └── README.md
```

## 2. Workflow Overview

The complete mapping workflow consists of the following stages:

1. **Area reconstruction:** Reconstruct annual provincial *Eucalyptus* plantation area estimates for 2001–2020.
2. **Feature extraction:** Process Landsat imagery in Google Earth Engine (GEE) and generate annual spectral, vegetation-index, textural, tasseled-cap, and terrain predictors.
3. **Sample extraction:** Extract predictor values from reference polygons to construct the classification dataset.
4. **Random Forest training:** Train and evaluate the Random Forest classifier using provincial training and validation samples.
5. **Annual prediction:** Apply the trained model to annual predictor rasters and generate probability and initial classification maps.
6. **Post-processing:** Reproject the annual maps to a common South China Albers Equal Area grid, apply temporal and area constraints, and prepare the final datasets.

The scripts are numbered according to their main execution sequence. The four post-processing scripts are stored separately in `scripts/postprocessing/`.

## 3. Study Area and Temporal Coverage

The study covers 11 provincial-level regions in southern China:

Guangxi, Guangdong, Hainan, Yunnan, Sichuan, Guizhou, Chongqing, Fujian, Jiangxi, Hunan, and Zhejiang.

The annual mapping period is 2001–2020, with a spatial resolution of 30 m.

## 4. Data and Model Access

The released datasets are available through Zenodo:

- **Final and initial binary maps:** [Zenodo – Annual Eucalyptus plantation maps](https://zenodo.org/uploads/22867686)
- **Random Forest probability maps:** [Zenodo – Annual Eucalyptus plantation probability maps](https://zenodo.org/uploads/22867667)

The trained Random Forest model is provided in the `model/` directory using Git Large File Storage (Git LFS).

## 5. Software and Dependencies

The workflow uses the following software and packages:

- R and R packages for area reconstruction, sample extraction, Random Forest training, and annual prediction.
- Google Earth Engine (GEE) JavaScript API for Landsat feature extraction.
- Python 3 with `rasterio`, `numpy`, `pandas`, `pyproj`, `affine`, and `netCDF4` for raster processing and NetCDF generation.

The required R and Python packages should be installed before running the corresponding scripts.

## 6. Configuration and Execution

Before execution, update the project root directory and input/output paths in the scripts to match your local environment.

The main scripts are designed to process individual provinces, with the province identifier specified in the configuration section. Repeat the corresponding processing steps for all 11 regions.

The reference polygons, Landsat-derived predictor rasters, forest masks, and other required input data must be prepared separately. The annual provincial area table (`Eucalyptus_area.csv`) is also required for area-constrained prediction and temporal post-processing.

For detailed instructions, see the README files in the `scripts/`, `scripts/postprocessing/`, `model/`, and `evaluation/` directories.

## 7. Map Classification

The final binary maps use the following classification values:

- `1`: *Eucalyptus* plantation.
- `0`: Other forest.
- `255`: NoData.

The probability products contain Random Forest classification probabilities.

## 8. License and Citation

If you use these codes or the associated datasets, please cite the corresponding data publication.

