# Multivariate Statistics Portfolio in R

*A collection of applied multivariate methods including regression, PCA, robust statistics, factor analysis, discriminant analysis, canonical correlation and symbolic data analysis.*

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

1. **Descriptive multivariate analysis**  
   Graphical exploration, covariance, correlation, and eigenstructure in the iris dataset.

2. **Distance-based clustering**  
   Hierarchical clustering and k-means applied to wine cultivar data.

3. **Cluster validation**  
   Calinski-Harabasz, Hartigan, silhouette, and gap statistics for selecting cluster solutions.

4. **Multivariate linear regression**  
   Joint prediction of college admissions outcomes.

5. **Robust regression**  
   Least-squares and MM-estimation for college acceptance prediction under outliers.

6. **Principal component analysis**  
   PCA of wine characteristics and NIR spectral data using loadings, scores, and explained variance.

7. **Robust PCA and image reconstruction**  
   Robust PCA for handwriting data and PCA-based X-ray image reconstruction via SVD.

8. **Rotated PCA and factor analysis**  
   Varimax rotation and factor analysis for interpretable latent handwriting structures.

9. **Multiple correlation analysis**  
   Relationships between relative weight and glucose-insulin biomarkers in diabetes data.

10. **Canonical correlation analysis**  
    Associations between gene expression and protein measurements.

11. **Discriminant analysis**  
    LDA, QDA, robust discriminant methods, and decision boundary visualization.

12. **Multiclass discriminant analysis**  
    LDA, QDA, and PCA-assisted classification for high-dimensional data.

13. **Symbolic interval data analysis**  
    Interval-valued loan profiles using aggregation, robust estimation, discrimination, and clustering.

---

## Repository Structure

Each folder contains:

* `analysis.Rmd` → full analysis
* `README.md` → rendered results
* figures (if applicable)

---

## Notes

This repository was developed as part of a university course in multivariate statistics and reflects a progression from classical methods to more advanced and robust approaches.
