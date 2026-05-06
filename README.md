# Multivariate Statistics Portfolio in R

*A collection of applied multivariate methods including regression, PCA, robust statistics, discriminant analysis, canonical correlation, and symbolic data analysis.*

---

## Overview

This repository contains a series of applied analyses in **multivariate statistics**, implemented in **R**.
Each section focuses on a specific methodological framework and demonstrates both theoretical understanding and practical implementation on real datasets.

The workflow across assignments includes:

* data preprocessing and transformation
* model estimation and comparison
* visualization and interpretation
* evaluation using cross-validation or test data

---

## Topics Covered

### 1. Regression Analysis

* multiple linear regression (OLS)
* robust regression (MM-estimation)
* model diagnostics and comparison
* cross-validation and prediction performance

### 2. Principal Component Analysis (PCA)

* dimensionality reduction and interpretation
* loadings and score analysis
* robust PCA methods
* high-dimensional PCA using SVD

### 3. Advanced PCA Applications

* PCA on handwriting (Alzheimer vs healthy)
* image compression using PCA (X-ray data)
* interpretation of variance structure and reconstruction

### 4. Rotation and Factor Analysis

* varimax rotation of principal components
* factor analysis and latent structure interpretation
* comparison between PCA and FA

### 5. Canonical Correlation Analysis

* relationships between variable sets
* comparison of Pearson vs Spearman approaches
* robust canonical correlation methods
* permutation-based inference

### 6. Discriminant Analysis

* LDA, QDA, and robust variants
* classification boundaries and visualization
* model comparison using error rates
* high-dimensional classification with PCA + LDA

### 7. Symbolic Data Analysis

* interval-valued data construction
* classical vs robust aggregation
* likelihood and robust model estimation
* discriminant analysis and clustering for interval data

---

## Repository Structure

```text
multivariate-statistics/
├── 01-regression-analysis/
├── 02-pca/
├── 03-advanced-pca/
├── 04-rotation-factor-analysis/
├── 05-canonical-correlation/
├── 06-discriminant-analysis/
├── 07-symbolic-data-analysis/
└── README.md
```

Each folder contains:

* `analysis.Rmd` → full analysis
* `README.md` → rendered results
* figures (if applicable)

---

## Methods & Tools

* **R packages**:

  * `MASS`, `robustbase`, `rrcov`
  * `cvTools`, `ccaPP`
  * `ISLR`, `mlbench`, `pls`, `heplots`
  * `MAINT.Data`, `pixmap`

* **Techniques**:

  * robust estimation
  * cross-validation
  * dimensionality reduction
  * model-based clustering
  * symbolic (interval) data modeling

---

## Key Takeaways

* Robust methods consistently improve stability in the presence of outliers.
* PCA and factor analysis provide complementary insights into data structure.
* High-dimensional problems require dimensionality reduction for reliable modeling.
* Classification performance depends strongly on model assumptions (LDA vs QDA).
* Symbolic data analysis extends classical methods by incorporating variability within observations.

---

## Author

**Georgios Papadopoulos**

---

## Notes

This repository was developed as part of a university course in multivariate statistics and reflects a progression from classical methods to more advanced and robust approaches.
