import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate
from datetime import datetime

# GLOBAL VALUES

class feature_metadata:
    def __init__(self, feature_name):
        self.name = feature_name
        self.type = None
        self.categorical_type = None
        self.mi_score = None

def LOAD_MANUAL_OVERRIDE_TOKENS():
    override_file = open("MANUAL_OVERRIDE_FILE.txt")
    tokens = override_file.read().split()
    return tokens

GARBAGE_TOKENS = ['?', 'NA', 'N/A', 'null', 'None', 'nan', 'NaN', '']
REMOVED_FEATURES = {}
PREDEFINED_REMOVE_FLAGS = ['name', 'time', 'id', 'jersey']
MANUAL_OVERRIDE_TOKENS = LOAD_MANUAL_OVERRIDE_TOKENS()
#feature metadata idea
METADATA = {}

#------------------------BROADCAST-----------------------#

def get_time():
    return datetime.now().strftime("%I:%M:%S %p")

#------------------------LOADING DATA-----------------------#

class ManyFilesError(Exception):
    """Too many files within a single folder"""
    pass

def load_data():
    #data collection
    folder_path = "raw_data"
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv") and os.path.isfile(os.path.join(folder_path, f))]
    dataframe = None
    #I might need to write a unique joining algorithm
    for file_name in files:
        #we need to convert them to dataframes as we iterate
        cur_df = None
        try:
            cur_df = pd.read_csv(f"raw_data/{file_name}")
        except Exception as e:
            print(f"Could not read {file_name}")
            continue
        if dataframe is None:
            dataframe = pd.read_csv(f"raw_data/{file_name}")
        else:
            raise ManyFilesError("Only one file in the folder at a time")
    return dataframe

#------------------------STATISTICAL TOOLS-----------------------#

def data_normalization(data, feature):

    #filling empty values appropriately
    data[feature] = pd.to_numeric(data[feature], errors = 'coerce')
    skew = data[feature].skew()
    if abs(skew) > 0.5:
        data[feature] = data[feature].fillna(data[feature].median())
    else:
        data[feature] = data[feature].fillna(data[feature].mean())
    
    #normalization
    min_val = data[feature].min()
    max_val = data[feature].max()

    if min_val == max_val:
        data[feature] = 0.0
        return
    
    data[feature] = (data[feature] - min_val) / (max_val - min_val)
    data[feature] = data[feature].astype(float)

def encode_categorical(data, feature):

    cardinality = data[feature].nunique()
    
    if cardinality == 1:
        data.drop(columns = feature, inplace = True)

    #binary encoding
    elif cardinality == 2:
        METADATA[feature].categoricaltype = "binary"
        data[feature], _ = data[feature].factorize()
    
    #hot-one encoding
    elif cardinality > 2 and cardinality < 100:
        old_columns = set(data.columns)
        one_hot = pd.get_dummies(data, columns = [feature], dtype = int)
        new_columns = set(one_hot) - old_columns
        METADATA.pop(feature)
        for col in new_columns:
            current_feature_metadata = feature_metadata(col)
            current_feature_metadata.type = "categorical"
            current_feature_metadata.categorical_type = "binary_one_hot"
            METADATA[col] = current_feature_metadata
        return one_hot
    
    #frequency encoding
    else:
        METADATA[feature].categoricaltype = "frequency"
        freq_map = data[feature].value_counts(normalize = True).to_dict()
        data[feature] = data[feature].map(freq_map)
    
    return data

#------------------------INITAL ENCODING OF DATA------------------------#

#helper functions
def usability(data, feature):
    #initial tests
    if feature in MANUAL_OVERRIDE_TOKENS:
        return True
    if feature in GARBAGE_TOKENS:
        REMOVED_FEATURES[feature] = "empty feature name, please label"
    
    #testing missingness
    total_entries = data[feature].notnull().sum()
    missing_values = data[feature].isna().sum()
    if missing_values/total_entries > .5:
        REMOVED_FEATURES[feature] = "feature missing more than half of its values"
        return False
    
    #semantic removal with commonly low info features
    for remove_flag in PREDEFINED_REMOVE_FLAGS:
        if remove_flag in feature and remove_flag not in MANUAL_OVERRIDE_TOKENS:
            REMOVED_FEATURES[feature] = "feature is predetermined to be unreliable"
            return False
    
    #removal of mixed data types (strings and ints)
    numeric_count = pd.to_numeric(data[feature], errors = 'coerce').notnull().sum()
    ratio = numeric_count/total_entries
    if ratio < 1.0 and ratio > 0.0:
        REMOVED_FEATURES[feature] = "feature has mixed types"
        return False

    return True
def type_prediction(column_data):
    temp = pd.to_numeric(column_data, errors = 'coerce')
    number_ratio = temp.notna().mean()
    
    #not all numerical
    if number_ratio != 1.0:
        return 'categorical'
    
    #finding decimals to determine continuous data
    if(temp%1 != 0).any():
        return 'continuous'

    #now it is more ambiguous, we perform a cardinality check to determine if we could have numerical representations of categories
    total_entries = column_data.notnull().sum()
    if temp.nunique()/total_entries > 0.9:
        return 'continuous'
    return 'categorical'

#main function
def encode(data):
    for feature in data.columns:
        data[feature] = data[feature].astype(str).str.strip()
        data[feature] = data[feature].replace(GARBAGE_TOKENS, np.nan)

        #unusable features are removed
        if not usability(data, feature):
            data.drop(columns = feature, inplace = True)
            continue

        #create metadata for a feature if we are using it
        current_feature_metadata = feature_metadata(feature)
        METADATA[feature] = current_feature_metadata

        if type_prediction(data[feature]) == 'continuous':
            METADATA[feature].type = "continuous"
            data_normalization(data, feature)
        else:
            METADATA[feature].type = "categorical"
            data = encode_categorical(data, feature)

    return data

#------------------------INITIAL FEATURE SELECTION-----------------------#

def MI_categorization(data, outcomes, outcome_type):
    #generate random column
    data["random_noise"] = np.random.randn(len(data))
    is_discrete = data.dtypes == int
    mutual_info_scores = None

    if outcome_type == "continuous":
        mutual_info_scores = mutual_info_regression(data, outcomes, discrete_features = is_discrete)
    else:
        mutual_info_scores = mutual_info_classif(data, outcomes, discrete_features = is_discrete)
    
    mutual_info_scores  = pd.Series(mutual_info_scores, index = data.columns, name = "mutual_info")
    mutual_info_scores = mutual_info_scores.sort_values(ascending = False)
    threshold = mutual_info_scores["random_noise"]
    kept_features = mutual_info_scores[mutual_info_scores>threshold].index.tolist()
    data.drop(columns = ["random_noise"], inplace = True)

    return kept_features

def variance_test(data, feature_name):
    if data[feature_name].var() < 0.01:
        data.drop(columns = [feature_name], inplace = True)
        REMOVED_FEATURES[feature_name] = "below variance threshold"
        METADATA.pop(feature_name)
        return True
    return False

def correlation_test(data, feature_name):
    for feature in data.columns:
        if METADATA.type == "continuous" and feature != feature_name and feature not in MANUAL_OVERRIDE_TOKENS:
            current_feature_data = data[feature_name]
            comparison_feature_data = data[feature]
            corr, p_val = pearsonr(current_feature_data, comparison_feature_data)
            if corr >= 0.85:
                data.drop(columns = [feature], inplace = True)
                REMOVED_FEATURES[feature] = f"high correlation to {feature_name}, correlation value of {corr}"
                METADATA.pop(feature)

def chi_square_test(data, feature_name):
    for feature in data.columns:
        if METADATA[feature].type == "categorical" and feature != feature_name and feature not in MANUAL_OVERRIDE_TOKENS:
            current_feature_data = data[feature_name]
            comparison_feature_data = data[feature]
            contingency_table = pd.crosstab(current_feature_data, comparison_feature_data)
            chi_2, p_val, dof, expected = chi2_contingency(contingency_table)

            if p_val > 0.05:
                data.drop(columns = [feature], inplace = True)
                REMOVED_FEATURES[feature] = f"high p_val for chi2 contingency to {feature_name}, p value of: {p_val}"
                METADATA.pop(feature)

def fast_select(data, outcomes, outcome_type):

    print(f"{get_time()}: Running mutual info test...")
    high_mi_features = MI_categorization(data, outcomes, outcome_type)

    print(f"{get_time()}: Running variance, correlation test on continuous features and chi2 on categorical...")
    for feature in data.columns:
        if feature in high_mi_features or feature in MANUAL_OVERRIDE_TOKENS:
            continue
        #this may seem wrong- but the reason why it needs to be here is because
        #when this code runs initially, it will delete anything it deems fit as it
        #iterates through, it may even delete more than one in a single run
        #so we just keep track of what it deletes so it doesn't try to delete it again
        elif feature in REMOVED_FEATURES:
            continue
        elif variance_test(data, feature):
            continue
        
        if METADATA[feature].type == "continuous":
            correlation_test(data, feature)
        elif METADATA[feature].type == "categorical":
            chi_square_test(data, feature)

#------------------------MODELING FUNCTIONS------------------------#

def modeling(data, outcomes, outcomes_type):
    models = {}
    model_type = None
    if outcomes_type == "continuous":
        outcomes = pd.to_numeric(outcomes, errors = 'coerce')
        models = {
            "Linear Regression": LinearRegression(),
            "Regressor Decision Tree": DecisionTreeRegressor(),
            "Regressor Random Forest": RandomForestRegressor(max_depth = 20, n_estimators = 25)
        }
        model_type = "Regression"
    else:
        outcomes, _ = outcomes.factorize()
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Radnom Forest": RandomForestClassifier
        }
        model_type = "Classification"
    
    print(f"{get_time()}: Model Type Selected: {model_type}... Evaluating individual models")

    best_model = None
    score = -np.inf
    model_name = None

    for name, model, in models.items():
        print(f"{get_time()}: Evaluating {name}")
        if outcomes_type == "categorical":
            results = cross_validate(
                model,
                data,
                outcomes,
                cv = 5,
                scoring = ["accuracy", "f1"]
            )

            accuracy = results["text_accuracy"].mean()
            f1 = results["test_f1"].mean()
            print(f"\n{name} | Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}\n")
            if f1 > score:
                best_model = model
                model_name = name
                score = f1
        else:
            results = cross_validate(
                model,
                data,
                outcomes,
                cv = 5,
                scoring = ["r2", "neg_mean_squared_error"]
            )

            r2 = results["test_r2"].mean()
            mse = -results["test_neg_mean_squared_error"].mean()
            print(f"\n{name} | R^2: {r2:.4f} | MSE: {mse:.4f}\n")

            if r2 > score:
                best_model = model
                model_name = name
                score = r2
    return best_model, model_name, score

#------------------------BACKEND PIPELINE------------------------#

def pipeline(raw_data, target_column):
    
    raw_data[target_column] = raw_data[target_column].astype(str).str.strip()
    raw_data[target_column] = raw_data[target_column].replace(GARBAGE_TOKENS, np.nan)
    raw_data.dropna(subset = [target_column])
    outcomes = raw_data.pop(target_column)

    print(f"{get_time()}: Performing data encoding and normalization")
    data = encode(raw_data)

    p = data.shape[1]
    n = data.shape[0]
    ratio = p/n
    high_dimensionality = (p > 300) or (ratio > 0.01)
    if high_dimensionality:
        print(f"{get_time()}: High dimensionality detected ~ Expect longer processing time...")
    
    #get the type of the target for model selection
    outcomes_type = type_prediction(outcomes)

    print(f"{get_time()}: Running basic feature selection")
    fast_select(data, outcomes, outcomes_type)

    
    print(f"{get_time()}: Running model fit and selection")
    model, model_name, score = modeling(data, outcomes, outcomes_type)

    print(f"{get_time()}: Completed, best model is {model_name} with score of {score}")

data = load_data()
target_column = "performance_score"

pipeline(data, target_column)
