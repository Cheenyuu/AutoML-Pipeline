# AutoML Pipeline Project

---

A basic machine learning project that employs the use of statistical analysis tools and machine learning to get useful information about large data.

---

# Features

* Live progress updates within terminal
* Metadata tracking
* Removed feature tracking
* Manual override token functionality
* Upload data and define target values
* Automatic type detection for dataframe columns
* Automatic data encoding for categorical and continuous data, optimized for model training
* Missing value handling
* Simple feature selection using variance tests, chi2 tests, correlation tests, and mutual information scores
* Automatic evalutation of models
* Automatic selection of models

---

# Pipeline

```
    Upload a CSV to raw_data file
                  |
                  v
Specify the target column within the code
                  |
                  v
 Automatic Encoding with Type Detection
                  |
                  v
      Initial Feature Selection
                  |
                  v
    Model Training and Comparison
                  |
                  v
          Model Selection
```

# Automatic Feature Type Detection

Pipeline determines whether the feature is continuous or categorical through basic test that involve assessing the columns individually and filtering based on expected *look* through numerical ratios.

### Continuous

A feature is considered continuous if:

* Decimal values are involved
* Most values are unique

### Categorical

A feature is considered categorical if:

* Text is found
* Small number of repeating integers (likely labels)

---

# Encoding and Processing

### Continuous

Continuous features are processed using min-max normalization so the magnitude of the numbers do not affect the modeling.

### Categorical

Categorical data is encoded in order to do two main things:

* Convert text-represented categories to numerical values
* Ensure numerically labeled categories are not unfairly weighted based on the magnitude of their label

For low cardinality categorical types, we use either binary encoding or one-hot encoding. For high cardinality categorical types, we use frequency encoding to minimize fragmentation and size of the overall dataset.

---

# Feature Selection

We run a quick and comprehensive feature selection in order to remove columns that are unlikely to provide useful information and reduce computational overhead for model evaluation.

We start by determining mutual information scores between all columns and the outcomes column to determine which features are likely worth keeping. If any feature falls above the threshold, we store it away so it will not be evaluated and will be kept.

If a feature does not meet the threshold for mutual information, it is tested for its variance measure, its correlation value (continuous features), and its chi2 value (categorical features).

---

# Feature Removal and Manual Override

Features are automatically removed during encoding or during feature selection through a variety of statistical measures. However, users can specify features that they require the code to keep within the MANUAL_OVERRIDE.txt file provided. When starting the code, the program will automatically fetch these tokens before any kind of parsing happens and ensure that these columns remain throughout the entirety of run.

