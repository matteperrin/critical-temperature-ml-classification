# WHAT HAS BEEN DONE SO FAR

# Data Inspection Notes

Purpose: Check the raw datasets before cleaning and modelling.
No missing values were found in either dataset.
train.csv contains 21,263 rows and 82 columns.
unique_m.csv contains 21,263 rows and 88 columns.
Both files have the same number of rows.
critical_temp matches row by row between both datasets.
The code also checks duplicates, invalid values,
infinite values, and material formatting issues.

The code loads and inspects the two raw superconductivity datasets before any cleaning or modelling is done. It checks for issues such as missing values, duplicates, invalid values, infinite values, and formatting problems. It also confirms that both files have the same number of rows and that their critical_temp values match row by row.


# Data Cleaning Notes

Loads the two raw superconductivity datasets.
Creates separate copies so the original raw data stays unchanged.
Checks how many exact duplicate rows exist in each dataset.
Removes those duplicate rows from the cleaning copies.
Prints the dataset shapes before and after duplicate removal.


# Data Transformation Notes

Loads the raw superconductivity training dataset.
Creates copies so the original raw data stays unchanged.
Creates a new column called "above_77k" for classification.
above_77k = 1 if critical_temp is greater than 77 K.
above_77k = 0 if critical_temp is 77 K or below.
Counts how many records belong to each class.
Calculates the proportion of the dataset in each class.
Purpose: Transform the temperature target into a binary classification problem.

# Outlier Notes

Potential outliers were found in many numerical features using the IQR method.
The largest numbers were found in:
std_Density: 3505
wtd_entropy_ElectronAffinity: 3435
gmean_atomic_mass: 3324
range_Density: 2894
wtd_std_FusionHeat: 2629
critical_temp had only 1 potential outlier.
The outliers were kept because they may represent real differences
between superconducting materials rather than incorrect data.

# Transformation Notes

Created a binary target called above_77k.
0 = critical temperature of 77 K or below.
1 = critical temperature above 77 K.
17,368 records are class 0 and 3,895 are class 1.
The classes are imbalanced: about 81.7% class 0 and 18.3% class 1.