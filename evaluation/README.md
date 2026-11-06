# Model Evaluation and Feature Importance

This directory contains supplementary tables describing the performance of the Random Forest classifier and the contribution of predictor variables.

## 1. File Description

| File | Description |
|---|---|
| `regional_accuracy_table.csv` | Provincial classification accuracy statistics derived from the independent validation samples. |
| `rf_feature_importance.csv` | Relative importance of predictor variables in the trained Random Forest model. |

## 2. Regional Accuracy Assessment

**File:** `regional_accuracy_table.csv`

This table reports classification accuracy statistics for the 11 provincial-level regions and the overall study area.

The evaluation metrics include overall accuracy (OA), user's accuracy (UA), producer's accuracy (PA), F1-score, and Kappa coefficient, where applicable.

The table provides the quantitative results used to assess the spatial performance of the Random Forest classification.

## 3. Random Forest Feature Importance

**File:** `rf_feature_importance.csv`

This table contains the feature importance values calculated from the trained Random Forest model.

It provides information on the relative contribution of the predictor variables to the classification process.

The results can be used to examine the importance of spectral, vegetation-index, textural, tasseled-cap, and terrain predictors.

## 4. Data Format

Both files are provided in CSV format and can be opened using spreadsheet software or imported into R and Python for further analysis.

