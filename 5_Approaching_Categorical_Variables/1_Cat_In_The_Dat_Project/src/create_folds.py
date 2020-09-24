import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config

def run():
    # Load train data into memory
    train_data = pd.read_csv(config.INPUT_TRAIN_FILE)

    # Assign 'kfold' = -1
    train_data['kfold'] = -1

    # shuffle the data
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    y = train_data['target'].values

    # create object of StratifiedKFold
    kf = StratifiedKFold(n_splits=5)

    # split the data into train and validation
    for fold, (t_, v_) in enumerate(kf.split(train_data, y)):
        train_data.loc[v_, 'kfold'] = fold

    # save data to 'input folder'
    train_data.to_csv(config.INPUT_FOLDS_FILE, index=False)

if __name__ == "__main__":
    run()