import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def run(fold):
    """
    Function that splits the data according to fold,
    then trains the model, and saves it.

    :param fold: Value for 'kfold' column to filter with
    """

    # Load the dataframe from train data with folds
    df = pd.read_csv('../input/train_folds.csv')

    # Split the data into train and validation
    train_data = df[df.kfold != fold].reset_index(drop=True)
    val_data = df[df.kfold == fold].reset_index(drop=True)

    # Fetch features, labels from train and val data
    x_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values

    x_val = val_data.drop('label', axis=1).values
    y_val = val_data['label'].values

    # Make an object of the DecisionTreeClassifier model
    clf = DecisionTreeClassifier()

    # Fit the model on train data
    clf.fit(x_train, y_train)

    # Predict the data on x_val
    y_val_preds = clf.predict(x_val)

    # Evaluate the model using accuracy_score
    accuracy = accuracy_score(y_val, y_val_preds)
    print(f'Fold={fold}, Accuracy: {accuracy}')

    # Save the model into output folder
    joblib.dump(clf, f'../models/dt_fold_{fold}.bin')

if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)