import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import joblib
import os
import argparse

import config

def run(fold):
    # Load data into memory
    train_data = pd.read_csv(config.TRAINING_FILE)

    # Get all features from train data
    features = [feat for feat in train_data.columns if feat not in ['id', 'target', 'kfold']]

    # Fill NaN values with "NONE"
    for col in features:
        train_data.loc[:, col] = train_data[col].astype(str).fillna("NONE")

    # Split into train and val_data
    df_train = train_data.loc[train_data['kfold'] != fold, :].reset_index(drop=True)
    df_val = train_data.loc[train_data['kfold'] == fold, :].reset_index(drop=True)

    # Concat train and val data to prepare for OHE
    data = pd.concat([df_train[features], df_val[features]], axis=0)

    # Create object for OneHotEncoder
    ohe = OneHotEncoder()

    # fit the data
    ohe.fit(data[features])
    
    # Transform train features 
    x_train = ohe.transform(df_train[features])

    # Transform val features
    x_val = ohe.transform(df_val[features])

    # Create LogisticRegression model object
    logres = LogisticRegression()

    # Fit the train data
    logres.fit(x_train, df_train['target'].values)

    # Predict Probabilities for each target value
    val_proba = logres.predict_proba(x_val)[:, 1]

    # Calculate the AUC score
    print(f"AUC score: {roc_auc_score(df_val['target'].values, val_proba)}")

    # Save the model
    joblib.dump(logres, os.path.join(config.MODEL_OUTPUT_DIR, f'ohe_logres_fold_{fold}.bin'))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('--fold', type=int, required=True, help="Fold value")
    params = ap.parse_args()

    run(params.fold)