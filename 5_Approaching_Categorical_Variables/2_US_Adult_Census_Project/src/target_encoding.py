"""
Target Encoding: It is a way of feature engineering from categorical features.
It is a technique in which you map each category in a given feature to its 
mean target value, but this must be done in a cross-validated manner.
"""

import copy
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import config

def mean_target_encoding(data):
    # Create a copy of the data
    df = copy.deepcopy(data)

    # Mapping income to numerical
    income_map = {
        " <=50K":0,
        " >50K":1
    }
    df.loc[:, 'income'] = df['income'].map(income_map)

    # numerical features
    num_cols = ['age', 'fnlwgt', 'capital_gain', 'capital_loss','hours_per_week']

    # Features
    features = [c for c in df.columns if c not in ['kfold', 'income'] and c not in num_cols]

    # FillNA to "None"
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("None")
    
    # Label encode categorical features
    for col in features:
        lbl_enc = LabelEncoder()
        lbl_enc.fit(df[col].values)
        df.loc[:, col] = lbl_enc.transform(df[col].values)
    
    encoded_dfs = []
    for fold_ in range(5):
        # Split into 
        df_train = df[df['kfold'] != fold_].reset_index(drop=True)
        df_valid = df[df['kfold'] == fold_].reset_index(drop=True)
        # Map each category of column to its mean target value
        for col in features:
            col_map = dict(
                df_train.groupby(col)["income"].mean()
            )
            df_valid.loc[:, col+"_enc"] = df_valid[col].map(col_map)
        # Add encoded validation df to the list
        encoded_dfs.append(df_valid)
    # Concat encoded_dfs to have complete dataframe
    encoded_df = pd.concat(encoded_dfs, axis=0)
    
    # print(encoded_df.head())
    return encoded_df

def run(df, fold_):
    """
    Train and evaluate the model on the dataframe for a fold value

    :params df: Dataframe
    :params fold_: Fold value to run
    """

    # features
    features = [c for c in df.columns if c not in ['kfold', 'income']]

    # Split the data into training and validation dataframes
    df_train = df[df['kfold'] != fold_].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold_].reset_index(drop=True)

    # Prepare train and valid features
    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # Create XGBClassifier object
    xgb = XGBClassifier(
        n_jobs=-1, 
        max_depth=7)
    
    # Fit the model
    xgb.fit(x_train, df_train['income'].values)

    # Predict the probabilities of class 1
    y_pred_prob = xgb.predict_proba(x_valid)[:, 1]

    # AUC Score
    auc_score = roc_auc_score(df_valid['income'].values, y_pred_prob)

    print(f'XGBClassifier Fold:{fold_} AUC Score={auc_score:.2f}')


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FOLDS_FILE)
    df = mean_target_encoding(df)
    for fold_ in range(5):
        run(df, fold_)