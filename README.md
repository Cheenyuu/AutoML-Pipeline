# AutoML Pipeline

An end-to-end Automated Machine Learning (AutoML) web application built with **Flask**, **Flask-SocketIO**, and **scikit-learn**. The application accepts a CSV dataset, automatically cleans and preprocesses the data, performs feature engineering and feature selection, evaluates several machine learning models using cross-validation, and selects the best-performing model.

The project is designed to minimize manual preprocessing while still producing interpretable results through a transparent pipeline.

---

# Features

* Upload datasets through a web interface
* Automatic data cleaning
* Automatic feature type detection
* Missing value handling
* Continuous feature normalization
* Categorical encoding
* Feature selection
* Multiple model evaluation
* Live progress updates through WebSockets
* Automatic model selection

---

# Pipeline Overview

```
CSV Upload
      │
      ▼
Data Validation
      │
      ▼
Automatic Cleaning
      │
      ▼
Feature Type Detection
      │
      ▼
Encoding & Normalization
      │
      ▼
Initial Model Training
      │
      ▼
Feature Selection
      │
      ▼
Retraining
      │
      ▼
Model Comparison
      │
      ▼
Best Model Selected
```

---

# Data Cleaning

Before any model is trained, every feature is evaluated.

The pipeline automatically:

* Removes unusable columns
* Removes columns with excessive missing values (>50%)
* Removes obvious identifier columns (ID, Name, Time, Jersey, etc.)
* Removes mixed-type columns
* Converts common missing-value tokens

Supported missing value tokens include:

```
?
NA
N/A
null
None
nan
NaN
(blank)
```

---

# Automatic Feature Type Detection

The pipeline attempts to determine whether each feature is **continuous** or **categorical**.

### Continuous Detection

A feature is considered continuous if:

* It contains decimal values
* Most values are unique
* The feature appears numerical

### Categorical Detection

A feature is considered categorical if:

* It contains text
* It contains a small number of unique integer values

This allows datasets without metadata to still be processed automatically.

---

# Continuous Feature Processing

Continuous variables undergo:

* Numeric conversion
* Missing value imputation

  * Mean (low skew)
  * Median (high skew)
* Min-Max normalization

Normalization formula:

```
(x - min) / (max - min)
```

---

# Categorical Feature Processing

Encoding is selected automatically based on cardinality.

### Binary Features

Uses factorization.

Example:

```
Yes → 0
No  → 1
```

---

### Low Cardinality (<100)

Uses one-hot encoding.

Example:

```
Color

↓

Color_Red
Color_Blue
Color_Green
```

---

### High Cardinality (≥100)

Uses frequency encoding.

Example:

```
Country

↓

United States → 0.27
Canada → 0.05
Germany → 0.03
```

This prevents extremely wide datasets caused by one-hot encoding.

---

# Feature Selection

When datasets become high dimensional, additional feature selection is performed.

A dataset is considered high dimensional when:

* Number of features > 300

or

* Feature/sample ratio exceeds the defined threshold.

The pipeline then applies several automated tests.

---

## Mutual Information

A random noise feature is injected into the dataset.

Only features that outperform the random noise feature are kept.

This provides a dynamic threshold instead of relying on manually chosen values.

---

## Variance Threshold

Features with extremely low variance are removed.

These features contribute little useful information.

---

## Pearson Correlation

Continuous variables are compared using Pearson correlation.

Highly correlated features are removed to reduce redundancy.

Current threshold:

```
|r| ≥ 0.85
```

---

## Chi-Square Testing

Categorical variables are compared using a Chi-Square contingency test.

Features determined to be statistically redundant may be removed.

---

# Supported Models

## Regression

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

Evaluation metrics:

* R² Score
* Mean Squared Error

---

## Classification

* Logistic Regression
* Decision Tree
* Random Forest

Evaluation metrics:

* Accuracy
* F1 Score

Cross-validation uses 5 folds.

The model with the highest evaluation score is selected automatically.

---

# Live Progress Updates

The application uses **Flask-SocketIO** to stream progress updates directly to the browser.

Users receive updates such as:

* Dataset uploaded
* Cleaning started
* Mutual information running
* Model evaluation
* Feature selection
* Final model results

This allows long-running jobs to provide real-time feedback instead of appearing frozen.

---

# Project Structure

```
project/
│
├── app.py
├── raw_data/
├── templates/
│     └── index.html
├── static/
├── MANUAL_OVERRIDE_FILE.txt
├── README.md
```

---

# Manual Override File

The pipeline supports a manual override file.

```
MANUAL_OVERRIDE_FILE.txt
```

Feature names listed in this file bypass automatic removal rules.

This is useful when domain knowledge indicates a feature should be preserved despite automated heuristics.

---

# Running the Application

Clone the repository.

```
git clone https://github.com/yourusername/AutoML-Pipeline.git
```

Install dependencies.

```
pip install -r requirements.txt
```

Run the application.

```
python app.py
```

Open your browser.

```
http://127.0.0.1:5000
```

Upload a CSV file to begin training.

---

# Current Limitations

* Supports one CSV file at a time
* Target column is currently hardcoded
* Structured data only
* No GPU acceleration
* Limited model library
* No hyperparameter optimization
* No model persistence
* No prediction endpoint

---

# Future Improvements

* Automatic target column detection
* Hyperparameter tuning
* Feature importance visualization
* SHAP explanations
* GPU acceleration
* Additional models (XGBoost, LightGBM, CatBoost)
* Automatic train/test splitting strategies
* Export trained models
* Prediction API
* Multi-file dataset merging
* Support for image and text datasets
* Pipeline configuration through the web interface

---

# Technologies Used

* Python
* Flask
* Flask-WTF
* Flask-SocketIO
* Eventlet
* Pandas
* NumPy
* SciPy
* scikit-learn
* tqdm

---

# Design Philosophy

This project focuses on creating an interpretable AutoML pipeline rather than a black-box optimizer.

Instead of hiding preprocessing decisions, each stage of the pipeline performs deterministic, explainable transformations. The goal is to automate repetitive machine learning tasks while keeping every preprocessing and model-selection decision transparent and understandable.

The project is intended as both a practical AutoML tool and an educational framework demonstrating how an automated machine learning workflow can be built from first principles.
