import pandas as pd
import os
import numpy as np
from scipy.stats import pearsonr

def categorical_preprocessing(dataframe, feature, unique_values):
    #make them numerical for our modeling
    swap_map = {
        value: i
        for i, value in enumerate(unique_values)
    }
    #binary encoding, nominal or ordinance does not really matter
    if len(unique_values) <= 2:
        try:
            dataframe[feature] = dataframe[feature].map(swap_map)
            return 1
        except Exception as e:
            print(f"Error mapping {feature}: {e}")
            return 0
    #one-hot encoding for more than one category
    else:
        for value in unique_values:
            dataframe[f"{feature}_{value}"] = (dataframe[feature] == value).astype(int)
        dataframe.drop(columns = [feature], inplace = True)


def normalization(dataframe, feature):
    #with the dataframe and the feature, I need to be able to normalize
    series = dataframe[feature]
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val:
        try:
            dataframe[feature] = pd.Series([0.0] * len(series), index = series.index)
        except Exception as e:
            print(f"Error normalizing {feature}: {e}")
            return 1
    try:
        dataframe[feature] = (series - min_val) / (max_val - min_val)
        return 1
    except Exception as e:
        print(f"Error normalizing {feature}: {e}")
        return 0

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
            dataframe = pd.concat([dataframe, cur_df], ignore_index = True)
    return dataframe

def test_continuous(dataframe, feature):
    
    return

def preprocess(dataframe):
    #preprocessing
    target_column = "classification"
    target_frame = dataframe[target_column]
    #features that need specification
    tbd_list = []
    if target_column not in dataframe.columns:
        print("no output")
    #it's more likely that we need some kind of human intervention here.
    #dataframe.drop([target_column], axis = 1, inplace = True)
    for feature in dataframe.columns:
        print(f"{feature}: {dataframe[feature].dtype}")
        #get rid of any \t or extra spaces for each entry
        if pd.api.types.is_object_dtype(dataframe[feature]) or pd.api.types.is_string_dtype(dataframe[feature]):
            #these operations need to be done to do automatic testing to see if it is categorical
            dataframe[feature] = dataframe[feature].str.strip()
            #fill null values with the most common
            dataframe[feature] = dataframe[feature].replace('?', np.nan)
            dataframe[feature] = dataframe[feature].fillna(dataframe[feature].mode()[0])
            unique_values = dataframe[feature].dropna().unique()
            
            
            #I want to get rid of this so that the program can be automatic
            if len(unique_values) > 5:
                print(f"{feature} has many categories, is this a categorical feature?")
                print("unique categories: ")
                print(unique_values)
                answer = input("YES (Y) | NO (N)\n").strip().upper()
                while answer not in ("YES", "NO", "Y", "N"):
                    print("ERROR: No valid answer given")
                    answer = input("YES (Y) | NO (N)\n").strip().upper()
                    print(answer) 
                if answer in ("NO", "N"):
                    tbd_list.append(feature)
                    continue
                #if the answer is yes, then this 'if' will still go on
            
            unique_count = dataframe[feature].nunique()
            ratio = unique_count / len(dataframe)
            if ratio > 0.5:
                #likely continuous, so we need to test it
                test_continuous(dataframe, feature)

            
            categorical_preprocessing(dataframe, feature, unique_values)
        elif pd.api.types.is_numeric_dtype(dataframe[feature]):
            dataframe[feature] = dataframe[feature].astype(str).str.strip()
            dataframe[feature] = dataframe[feature].replace('?', np.nan)
            dataframe[feature] = pd.to_numeric(dataframe[feature], errors = 'coerce')
            dataframe[feature] = dataframe[feature].fillna(dataframe[feature].mean())
            unique_values = dataframe[feature].dropna().unique()
            if len(unique_values) < 5:
                print(f"{feature} has few numerical inputs, is this a continuous feature?")
                print("unique categories: ")
                print(unique_values)
                answer = input("YES (Y) | NO (N)\n").strip().upper()
                while answer not in ("YES", "NO", "Y", "N"):
                    print("ERROR: No valid answer given")
                    answer = input("YES (Y) | NO (N)\n").strip().upper()
                    print(answer) 
                if answer in ("NO", "N"):
                    tbd_list.append(feature)
                    continue
            normalization(dataframe, feature)
    for feature in tbd_list:
        print(f"[][][][] is {feature} 1. categorical or 2. continuous? [][][][]")
        unique_values = dataframe[feature].dropna().unique()
        print(unique_values)
        answer = None
        while answer not in ("1", "2"):
            answer = input("[1] or [2]\n")
        if answer == "1":
            categorical_preprocessing(dataframe, feature, unique_values)
        elif answer == "2":
            #insurance
            dataframe[feature] = pd.to_numeric(dataframe[feature], errors = 'coerce')
            normalization(dataframe, feature)
    return dataframe

"""
def feature_selection(preprocessed_dataframe):
    #now I need to do a correlation test to determine whether or not 
    legend = {}
    for feature in preprocessed_dataframe.drop(columns = ['classification'], axis = 1).columns:
        corr_val = pearsonr(preprocessed_dataframe[feature], preprocessed_dataframe['classification'])[0]
        legend[feature] = corr_val
    
    print(legend)
    return
"""

def modeling():
    return

def main():
    return
















dataframe = load_data()
df = preprocess(dataframe)
print(df)