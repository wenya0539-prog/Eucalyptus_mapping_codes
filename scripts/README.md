Source code for annual Eucalyptus plantation mapping.
# Main Processing Scripts

This directory contains the five main scripts used to reconstruct provincial area estimates, extract Landsat predictors, prepare reference samples, train the Random Forest classifier, and generate annual *Eucalyptus* plantation maps.

## 1. Script Overview

| Script | Main Function |
|---|---|
| `01_area_reconstruction.R` | Reconstruct annual provincial *Eucalyptus* plantation area estimates (2001–2020). |
| `02_GEE_feature_extraction.js` | Extract annual Landsat-based spectral, vegetation-index, textural, tasseled-cap, and terrain predictors in Google Earth Engine. |
| `03_sample_extraction.R` | Extract predictor values from reference polygons and generate the sample dataset for model training. |
| `04_random_forest_training.R` | Train and evaluate the Random Forest classifier and save the trained model. |
| `05_annual_prediction.R` | Generate annual probability and initial classification maps using the trained model. |

## 2. Execution Order

The recommended processing sequence is:

### Step 1. Annual Area Reconstruction

**Script:** `01_area_reconstruction.R`

Reconstruct annual provincial *Eucalyptus* plantation area estimates using available National Forest Inventory statistics and temporal interpolation.

**Output:** `Eucalyptus_area.csv`

This table provides provincial annual area constraints for subsequent prediction and temporal post-processing.

### Step 2. Landsat Feature Extraction

**Script:** `02_GEE_feature_extraction.js`

Run the script in the Google Earth Engine Code Editor.

The script processes Landsat 5 TM, Landsat 7 ETM+, and Landsat 8 OLI surface reflectance imagery, harmonizes the earlier sensors to the Landsat 8 reference system, and generates annual predictor layers.

The predictor set includes:

- Surface reflectance bands;
- Vegetation indices;
- GLCM textural features;
- Tasseled-cap components;
- Elevation, slope, and aspect.

**Output:** Annual predictor rasters exported through Google Drive.

### Step 3. Reference Sample Extraction

**Script:** `03_sample_extraction.R`

Extract raster predictor values within reference polygons and assign the corresponding polygon class labels.

**Output:** A tabular sample dataset containing predictor variables and the `EUCA` classification label.

### Step 4. Random Forest Model Training

**Script:** `04_random_forest_training.R`

Combine provincial samples, perform a stratified training/validation split, train the probability-based Random Forest classifier, and evaluate classification performance.

**Outputs:**

- Trained Random Forest model (`.rds`);
- Feature importance statistics;
- Provincial classification accuracy statistics.

The trained model is saved in the `model/` directory, while the evaluation tables are stored in `evaluation/`.

### Step 5. Annual Prediction

**Script:** `05_annual_prediction.R`

Apply the trained model to annual predictor rasters and generate probability-based predictions for each province and year.

The annual provincial area estimates and forest masks are used to constrain the predictions.

**Outputs:**

- Annual Random Forest probability rasters;
- Initial binary classification rasters.

These outputs are used as inputs for the subsequent post-processing workflow.

## 3. Required Input Data

The following inputs must be prepared before execution:

- Annual Landsat-derived predictor rasters;
- Reference polygon data with classification attributes;
- Annual forest masks derived from CLCD;
- Provincial annual area estimates (`Eucalyptus_area.csv`);
- The trained Random Forest model for annual prediction.

The input and output directories are configured within each script and should be updated according to the local working environment.

## 4. Software Requirements

- R with the required spatial analysis, data processing, and Random Forest packages;
- Google Earth Engine JavaScript API;
- Python is used separately for the subsequent post-processing scripts.

## 5. Notes

The scripts are configured for province-wise processing. Update the province identifier and relevant file paths before running each script.

The predictor names must remain consistent across feature extraction, sample extraction, model training, and annual prediction.

In particular, `NRGDI` must retain the definition `(Red - Green) / (Red + Green)` throughout the workflow.

For raster reprojection, temporal post-processing, province merging, and NetCDF conversion, see `postprocessing/README.md`.

