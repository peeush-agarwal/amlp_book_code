import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import joblib
import os
import argparse

import config

def run(fold):
    # Load training with folds file into dataframe
    df = pd.read_csv(config.TRAINING_FOLDS_FILE)

    # Drop numerical features from dataframe
    num_cols = ['age','fnlwgt','capital_gain','capital_loss','hours_per_week']
    df = df.drop(num_cols, axis=1)

    # Fetch features from the dataframe
    features = [feat for feat in df.columns if feat not in ['income', 'kfold']]

    # Fill NaN with "NONE"
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # Transform target column to numerical column
    mapping = {
        ' <=50K':0,
        ' >50K':1
    }
    df.loc[:, 'income'] = df['income'].map(mapping)

    # Split dataframe into train and validation dataframes using fold value
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_val = df[df['kfold'] == fold].reset_index(drop=True)

    # Concat features dataframes to prepare for OneHotEncoding transformation
    df_full = pd.concat([df_train[features], df_val[features]], axis=0)

    # Create an object of OneHotEncoder
    ohe = OneHotEncoder()

    # Fit OneHotEncoder object on the full dataframe
    ohe.fit(df_full[features])

    # Transform train and validation features using OneHotEncoder
    x_train = ohe.transform(df_train[features])
    x_val = ohe.transform(df_val[features])

    # Create an object of LogisticRegression
    model = LogisticRegression()

    # Fit the LogisticRegression model on train data
    model.fit(x_train, df_train['income'].values)

    # Predict the probabilities target using validation features
    y_val_preds = model.predict_proba(x_val)[:, 1]

    # Evaluate the model using AUC Score
    auc_score = roc_auc_score(df_val['income'].values, y_val_preds)
    print(f'Fold={fold}, AUC score={auc_score}')

    # Save the trained model with fold value
    joblib.dump(model, os.path.join(config.MODEL_OUTPUT_DIR, f'ohe_logres_fold_{fold}.bin'))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, help='Fold value')
    args = ap.parse_args()

    run(args.fold)