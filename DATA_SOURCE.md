# Data Source

## Dataset
**Breast Cancer Wisconsin (Diagnostic)** - 569 instances, 30 numeric features, binary target.

- **UCI page:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **DOI:** 10.24432/C5DW2B
- **Creators:** William Wolberg, Olvi Mangasarian, Nick Street, W. Street
- **Donated:** 1995 (UCI Machine Learning Repository)
- **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0). Attribution is required when reusing the data.

### Suggested citation
Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

## How the CSV was obtained
The build environment could not reach the UCI website, so `disease_dataset.csv` was exported from the copy of this dataset that ships with scikit-learn (`sklearn.datasets.load_breast_cancer`), which scikit-learn documents as a copy of the UCI Breast Cancer Wisconsin (Diagnostic) dataset. The values were not modified. The original UCI download was **not** compared byte-for-byte against this file.

### Transformations applied
1. Feature names converted from `mean radius` to `mean_radius` (spaces -> underscores).
2. Target column named `diagnosis`; scikit-learn's numeric codes (0/1) mapped back to the UCI letters: `M` = malignant, `B` = benign.
3. The UCI file's patient `ID` column is not present in the scikit-learn copy and is not included. No patient-identifying information is in this project.
4. No rows were removed, imputed or altered.

## Columns
| Column | Type | Description |
|---|---|---|
| `diagnosis` | categorical (target) | `M` malignant (212 rows), `B` benign (357 rows) |
| 30 feature columns | numeric | 10 nucleus measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension) as `mean_*`, `*_error` (standard error) and `worst_*` (largest value) |

Class counts: 357 benign, 212 malignant (62.7% / 37.3%).
Quality checks (see notebook): 0 missing values, 0 duplicate rows, no negative values.

## Re-creating the CSV
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
d = load_breast_cancer()
df = pd.DataFrame(d.data, columns=[c.replace(' ', '_') for c in d.feature_names])
df.insert(0, 'diagnosis', pd.Series(d.target).map({0: 'M', 1: 'B'}))
df.to_csv('data/disease_dataset.csv', index=False)
```
