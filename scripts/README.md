# Processing Scripts

This directory contains the complete processing workflow for annual *Eucalyptus* plantation mapping across southern China from 2001 to 2020. It includes five main processing scripts and four raster post-processing scripts.

## 1. Script Overview

### Main Processing Scripts

| Script | Description |
|---|---|
| `01_area_reconstruction.R` | Reconstruct annual provincial *Eucalyptus* plantation area estimates (2001–2020). |
| `02_GEE_feature_extraction.js` | Extract Landsat-based spectral, vegetation-index, textural, tasseled-cap, and terrain predictors using Google Earth Engine. |
| `03_sample_extraction.R` | Extract predictor values from reference polygons to construct the classification sample dataset. |
| `04_random_forest_training.R` | Train and evaluate the Random Forest classifier and save the trained model. |
| `05_annual_prediction.R` | Generate annual probability and initial classification maps using the trained model. |

### Raster Post-processing Scripts

Located in `postprocessing/`.

| Script | Description |
|---|---|
| `001_reproject_to_south_china_albers.py` | Reproject annual prediction rasters to a common 30-m South China Albers Equal Area grid. |
| `002_temporal_postprocessing_albers_neighbor.py` | Apply temporal occurrence filtering, provincial area constraints, and neighbourhood-based tie-breaking to generate final binary maps. |
| `003_merge_11_provinces.py` | Merge the 11 provincial binary maps and organize the initial maps and probability rasters. |
| `004_compress_probability_to_netcdf.py` | Convert annual provincial probability rasters into compressed 20-year NetCDF-4 files. |

## 2. Processing Workflow

The main processing sequence is:

**Step 1. Annual Area Reconstruction**

Run `01_area_reconstruction.R` to reconstruct annual provincial *Eucalyptus* plantation area estimates using National Forest Inventory statistics and temporal interpolation.

Output: `Eucalyptus_area.csv`.

**Step 2. Landsat Feature Extraction**

Run `02_GEE_feature_extraction.js` in the Google Earth Engine Code Editor. Landsat 5 TM, Landsat 7 ETM+, and Landsat 8 OLI surface reflectance imagery are processed to generate annual predictor layers.

The predictor set includes surface reflectance bands, vegetation indices, GLCM textural features, tasseled-cap components, and terrain variables.

Output: Annual predictor rasters exported through Google Drive.

**Step 3. Reference Sample Extraction**

Run `03_sample_extraction.R` to extract predictor values from reference polygons and assign the corresponding classification labels.

Output: A tabular sample dataset containing predictor variables and the `EUCA` class label.

**Step 4. Random Forest Training**

Run `04_random_forest_training.R` to train the probability-based Random Forest classifier using provincial samples and a 70/30 training-validation split.

Outputs include the trained model, classification accuracy statistics, and feature importance results.

**Step 5. Annual Prediction**

Run `05_annual_prediction.R` to apply the trained model to annual predictor rasters and generate provincial probability and initial binary classification maps.

The annual provincial area estimates and forest masks are used to constrain the predictions.

## 3. Raster Post-processing Workflow

The post-processing scripts are located in `postprocessing/`.

**Step 6. Raster Reprojection**

Run `001_reproject_to_south_china_albers.py` to reproject annual probability and initial classification rasters to a common South China Albers Equal Area projection at 30-m resolution.

Projection parameters:

- Standard parallels: 18°N and 34°N;
- Latitude of origin: 26°N;
- Central meridian: 110°E.

**Step 7. Temporal Post-processing**

Run `002_temporal_postprocessing_albers_neighbor.py` to apply cumulative temporal occurrence rules, provincial area constraints, and neighbourhood-based tie-breaking.

Output: Final annual post-processed binary maps for each province.

**Step 8. Provincial Map Assembly**

Run `003_merge_11_provinces.py` to combine the 11 provincial maps into annual maps covering the entire study area. The script also organizes the initial binary maps and provincial probability rasters.

Outputs:

- `Eucalyptus_YYYY.tif`: Final annual binary maps;
- `Eucalyptus_initial_YYYY.tif`: Initial annual binary maps;
- Provincial annual probability rasters.

**Step 9. Probability Data Conversion**

Run `004_compress_probability_to_netcdf.py` to convert the 20 annual probability rasters for each province into a single three-dimensional NetCDF-4 file.

The output follows CF-1.8 conventions and uses `uint16` storage with a scale factor of `0.0001` and a NoData value of `65535`.

**Note:** Step 9 is a separate processing branch. It uses the reprojected probability rasters generated in Step 6 and does not depend on Steps 7 or 8.

## 4. Required Input Data

The following data should be prepared before running the corresponding scripts:

- Landsat-derived annual predictor rasters;
- Reference polygon data with classification attributes;
- Annual forest masks derived from CLCD;
- Provincial annual area estimates (`Eucalyptus_area.csv`);
- Trained Random Forest model for annual prediction.

The trained model is provided in the `model/` directory. Evaluation tables are available in the `evaluation/` directory.

## 5. Software Requirements

- R and the required R packages for spatial processing, data manipulation, and Random Forest modelling;
- Google Earth Engine JavaScript API;
- Python 3 with `numpy`, `pandas`, `rasterio`, `pyproj`, `affine`, and `netCDF4`.

## 6. Configuration

Before execution, update the project root directory and input/output paths in the configuration sections of the scripts according to your local environment.

The scripts support province-wise processing. Set the corresponding province identifier and repeat the processing steps for all 11 provincial-level regions.

Ensure that predictor names and definitions remain consistent throughout feature extraction, sample extraction, model training, and annual prediction. In particular, `NRGDI` is defined as `(Red - Green) / (Red + Green)`.

