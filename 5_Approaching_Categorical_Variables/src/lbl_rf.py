"""
Random Forest model after transformation using LabelEncoder.
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import joblib
import os
import argparse

import config
import model_dispatcher

def run(fold, model_name):
    """
    Run an iteration of training model for a single fold. It displays the AUC score of the trained model.

    :params fold: Fold value
    :params model_name: Model name from model_dispatcher
    """

    # Load the data file into memory
    train_data = pd.read_csv(config.TRAINING_FILE)

    # Extract features from the dataframe
    features = [feat for feat in train_data.columns if feat not in ['id', 'target', 'kfold']]

    # Fill NaN values in features with "NONE"
    for col in features:
        train_data.loc[:, col] = train_data[col].astype(str).fillna("NONE")
    
    # Split the training data into train and validation dataframes
    df_train = train_data.loc[train_data['kfold'] != fold, :].reset_index(drop=True)
    df_val = train_data.loc[train_data['kfold'] == fold, :].reset_index(drop=True)

    # Transform each column into numerical encoding using LabelEncoder object
    for col in features:
        # Create an object of LabelEncoder
        lbl = LabelEncoder()

        # Fit the LabelEncoder on complete dataframe
        lbl.fit(train_data[col])

        # Transform the training and validation column using LabelEncoder
        df_train.loc[:, col] = lbl.transform(df_train[col])
        df_val.loc[:, col] = lbl.transform(df_val[col])
    
    # Get training and validation features 
    x_train = df_train[features]
    x_val = df_val[features]

    # Create an object of RandomForestClassifier
    modl = model_dispatcher.MODELS[model_name]

    # Fit the classifier on training data
    modl.fit(x_train, df_train['target'].values)

    # Predict the target values for Validation features
    y_val_preds = modl.predict_proba(x_val)[:, 1]

    # Measure the AUC score
    auc_score = roc_auc_score(df_val['target'].values, y_val_preds)

    print(f'Model={model_name}, Fold={fold}, AUC score={auc_score}')

    # Save the model
    joblib.dump(modl, os.path.join(config.MODEL_OUTPUT_DIR, f'lbl_{model_name}_fold_{fold}.bin'))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, help='Fold value')
    ap.add_argument('--model', type=str, required=True, help="Model name from model_dispatcher")
    args = ap.parse_args()

    run(args.fold, args.model)