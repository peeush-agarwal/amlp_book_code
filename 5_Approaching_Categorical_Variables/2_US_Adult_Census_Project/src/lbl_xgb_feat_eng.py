import itertools
import pandas as pd 
import numpy as np 

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost

import joblib
import os
import argparse

import config
import model_dispatcher

def feature_engineering(df, cat_cols):
    """
    This creates new features taking categorical cols for df_train/df_test dataframes.

    :param df: Dataframe with all data
    :param cat_cols: Categorical columns from the dataframe
    :returns new dataframe with more features from categorical columns.
    """

    # Create categorical features combining 2 cols at a time.
    # Ex: itertools.combinations creates combinations from the iterable of size 2 (in this case)
    # list(itertools.combinations([1, 2, 3], 2))
    # Output: [(1, 2), (1, 3), (2, 3)]
    comb = itertools.combinations(cat_cols, 2)
    for c1, c2 in comb:
        df.loc[:, c1+"_"+c2] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df

def run(fold, model_name):
    # Load folds file into dataframe
    df = pd.read_csv(config.TRAINING_FOLDS_FILE)

    # Map target column to numerical values
    mapping = {
        ' <=50K':0,
        ' >50K':1
    }
    df.loc[:,'income'] = df['income'].map(mapping)

    # List all numerical columns in the dataframe
    num_cols = ['age', 'fnlwgt', 'capital_gain', 'capital_loss','hours_per_week']

    # Get all categorical columns from dataframe columns
    cat_cols = [c for c in df.columns if c not in num_cols and c not in ['income', 'kfold']]

    # Feature engineering on categorical columns
    df = feature_engineering(df, cat_cols)

    # Features after feature engineering
    features = [c for c in df.columns if c not in ['income', 'kfold']]

    # Fill NA as "None" and Label Encode categorical columns
    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("None")

            lbl_enc = LabelEncoder()
            lbl_enc.fit(df[col])
            df[col] = lbl_enc.transform(df[col])

    # Train dataframe
    df_train = df[df['kfold'] != fold].reset_index(drop=True)

    # Test dataframe
    df_test = df[df['kfold'] == fold].reset_index(drop=True)

    # Create XGBClassifier
    xgb = model_dispatcher.models[model_name]

    # Train the classifier
    xgb.fit(df_train[features].values, df_train['income'].values)

    # Predict the probabilities of class:1
    y_pred_prob = xgb.predict_proba(df_test[features].values)[:, 1]

    # Evaluate the model using ROC AUC Score
    auc_score = roc_auc_score(df_test['income'].values, y_pred_prob)
    print(f'Fold={fold}, AUC score={auc_score}')

    # Save the model
    joblib.dump(xgb, os.path.join(config.MODEL_OUTPUT_DIR, f'xgb_feat_eng_model_{model_name}_fold_{fold}.bin'))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True, help='Fold value', type=int)
    ap.add_argument('--model_name', required=True, help='Model name from model_dispatcher')
    args = ap.parse_args()

    run(args.fold, args.model_name)