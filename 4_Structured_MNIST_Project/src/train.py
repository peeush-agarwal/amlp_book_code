import argparse
import joblib
import os

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import config
import model_dispatcher

def run(fold, model):
    """
    Function that splits the data according to fold,
    then trains the model, and saves it.

    :param fold: Value for 'kfold' column to filter with
    :param model: model name to load from model_dispatcher
    """

    # Load the dataframe from train data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # Split the data into train and validation
    train_data = df[df.kfold != fold].reset_index(drop=True)
    val_data = df[df.kfold == fold].reset_index(drop=True)

    # Fetch features, labels from train and val data
    x_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values

    x_val = val_data.drop('label', axis=1).values
    y_val = val_data['label'].values

    # Fetch model from model_dispatcher
    clf = model_dispatcher.models[model]

    # Fit the model on train data
    clf.fit(x_train, y_train)

    # Predict the data on x_val
    y_val_preds = clf.predict(x_val)

    # Evaluate the model using accuracy_score
    accuracy = accuracy_score(y_val, y_val_preds)
    print(f'Fold={fold}, Model={model}, Accuracy: {accuracy}')

    # Save the model into output folder
    joblib.dump(clf, os.path.join(config.MODELS_OUTPUT, f'{model}_fold_{fold}.bin'))

if __name__ == "__main__":
    # Parse arguments from CLI
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--model', type=str, required=True)
    args = ap.parse_args()

    run(args.fold, args.model)