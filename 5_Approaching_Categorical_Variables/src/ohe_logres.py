import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config

import os
import joblib
import argparse

def run(fold):
    # Load the data file into memory
    train_data = pd.read_csv(config.TRAINING_FILE)

    # Get all features from the dataframe
    features = [feat for feat in train_data.columns if feat not in ['id', 'target', 'kfold']]

    # Fill NaN with "NONE" after converting into string.
    # All columns here are categorical so we can convert values to string
    for col in features:
        train_data.loc[:, col] = train_data[col].astype(str).fillna("NONE")
    
    # Split the training data into training and validation dataframes
    df_train = train_data.loc[train_data['kfold'] != fold, :].reset_index(drop=True)
    df_val = train_data.loc[train_data['kfold'] == fold, :].reset_index(drop=True)

    # Merge these dataframes to fit OneHotEncoder for all the categories in training and validation dataframes
    data = pd.concat([df_train, df_val], axis=0)

    # Create object of OneHotEncoder
    ohe = OneHotEncoder()

    # Fit OneHotEncoder for features in the data
    ohe.fit(data[features])

    # Transform the features in train and validation dataframes
    x_train = ohe.transform(df_train[features])
    x_val = ohe.transform(df_val[features])

    # Create object of LogisticRegression
    logres = LogisticRegression()

    # Fit the logistic regression model over the training data
    logres.fit(x_train, df_train['target'].values)

    # Predict the probabilities for validation data and store predictions only for label '1'
    y_val_preds = logres.predict_proba(x_val)[:, 1]

    # Measure the AUC score for trained model
    auc_score = roc_auc_score(df_val['target'].values, y_val_preds)

    # Print the AUC score
    print(f'Fold={fold}, AUC Score={auc_score}')

    # Save the model
    joblib.dump(logres, os.path.join(config.MODEL_OUTPUT_DIR, f'ohe_logres_fold_{fold}.bin'))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, help='Fold value')
    args = ap.parse_args()

    run(fold=args.fold)