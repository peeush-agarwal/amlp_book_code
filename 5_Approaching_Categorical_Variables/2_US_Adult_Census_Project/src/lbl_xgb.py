import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import joblib
import os
import argparse

import config

def run(fold):
    # Load dataframe from training file with folds
    df = pd.read_csv(config.TRAINING_FOLDS_FILE)

    # Drop numerical features
    num_cols = ['age','fnlwgt','capital_loss','capital_gain','hours_per_week']
    # df = df.drop(num_cols, axis=1)

    # Fetch features 
    features = [feat for feat in df.columns if feat not in ['income', 'kfold']]

    # Fill NaN with "NONE"
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # Map target column to numerical values
    mapping = {
        ' <=50K':0,
        ' >50K':1
    }
    df.loc[:,'income'] = df['income'].map(mapping)

    # Label encode each categorical feature only
    for col in features:
        if col not in num_cols:
            lbl = LabelEncoder()
            df.loc[:, col] = lbl.fit_transform(df[col].values)

    # Split the dataframe into train and validation dataframes using fold value
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_val = df[df['kfold'] == fold].reset_index(drop=True)

    # Create an object of XGBoost model
    model = xgb.XGBClassifier(
        n_jobs=-1)

    # Fit the model using training data
    model.fit(df_train[features].values, df_train['income'].values)

    # Predict the probabilities of target using validation features
    y_val_preds = model.predict_proba(df_val[features].values)[:, 1]

    # Evaluate the model using AUC score
    auc_score = roc_auc_score(df_val['income'].values, y_val_preds)
    print(f'Fold={fold}, AUC score={auc_score}')

    # Save the model
    joblib.dump(model, os.path.join(config.MODEL_OUTPUT_DIR, f'lbl_xgb_fold_{fold}.bin'))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, help='Fold value')
    args = ap.parse_args()

    run(args.fold)