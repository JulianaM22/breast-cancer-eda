# Breast Cancer Datasets

## Dataset 1: Wisconsin Original
**File:** `breast_cancer_wisconsin_original_cleaned.csv`

### Details
- **Rows:** 683 (16 rows with missing values dropped from original 699)
- **Features:** 9 tumor characteristics
- **Scale:** All features rated 1-10 (1=normal, 10=abnormal)
- **Target:** `Diagnosis` and `Is_Malignant`

### Target Variables
- `Diagnosis`: Original labels (2=benign, 4=malignant)
- `Is_Malignant`: Binary (0=benign, 1=malignant)

### Features (all 1-10 scale)
- Clump Thickness
- Uniformity of Cell Size
- Uniformity of Cell Shape
- Marginal Adhesion
- Single Epithelial Cell Size
- Bare Nuclei
- Bland Chromatin
- Normal Nucleoli
- Mitoses

### Diagnosis Distribution
- Benign: ~65%
- Malignant: ~35%

### Cleaning Applied
- Converted `Bare Nuclei` from object to float (replaced '?' with NaN)
- Dropped 16 rows with missing values
- Removed duplicates
- Dropped `Sample code number` column

---

## Dataset 2: Coimbra
**File:** `breast_cancer_coimbra_cleaned.csv`

### Details
- **Rows:** ~116 (small dataset)
- **Features:** 9 clinical/metabolic markers
- **Scale:** Various continuous scales
- **Target:** `Classification`

### Target Variable
- `Classification`: Binary (0=healthy/control, 1=patient/malignant)

### Features
1. **Age** (years)
2. **BMI** (kg/m²)
3. **Glucose** (mg/dL)
4. **Insulin** (µU/mL)
5. **HOMA** (Homeostasis Model Assessment)
6. **Leptin** (ng/mL)
7. **Adiponectin** (µg/mL)
8. **Resistin** (ng/mL)
9. **MCP-1** (Monocyte Chemoattractant Protein-1, pg/dL)

### Diagnosis Distribution
- Healthy: ~45%
- Patient: ~55%

### Cleaning Applied
- Converted `Classification` from 1/2 to 0/1
- Removed duplicates

---

## Dataset 3: Wisconsin Diagnostic
**File:** `breast_cancer_wisconsin_diagnostic_cleaned.csv`

### Details
- **Rows:** 569 (no missing values)
- **Features:** 30 continuous measurements
- **Scale:** Various scales (NOT standardized - features range from 0 to 4000+)
- **Target:** `diagnosis`

### Target Variable
- `diagnosis`: Binary (0=benign, 1=malignant)

### Features (30 total)
Features categorize (mean, standard error, worst) for 10 measurements:
- **Measurements:** radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension

### Diagnosis Distribution
- Benign: ~63%
- Malignant: ~37%

### Cleaning Applied
- Dropped `id` column and `Unnamed: 32` column 
- Converted `diagnosis` from M/B to 1/0
- Removed duplicates

## Dataset 4: CDC WONDER Cancer Incidence
**File:** `cancer_incidence_1999_2022_cleaned.csv`

### Details
- **Rows:** 352
- **Features:** Age groups, year, incidence counts
- **Scale:** Count data (integer values)
- **Time Period:** 1999-2022
- **Geographic Coverage:** United States

### Features
- **Age Groups**: Categorized (< 1 year, 10-14 years, 15-19 years, etc.)
- **Year**: Calendar year of diagnosis
- **Count**: Number of breast cancer cases

### Cleaning Applied
- Dropped `Notes`, `Year Code`, `Age Groups Code` columns
- Removed nulls


### Original Sources
- **Dataset 1:** [Wisconsin Original](https://www.kaggle.com/datasets/mariolisboa/breast-cancer-wisconsin-original-data-set)
- **Dataset 2:** [Coimbra](https://www.kaggle.com/datasets/atom1991/breast-cancer-coimbra)
- **Dataset 3:** [Wisconsin Diagnostic](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Dataset 4:** [CDC WONDER](https://wonder.cdc.gov/)

**Last Updated:** 12/7/2025
