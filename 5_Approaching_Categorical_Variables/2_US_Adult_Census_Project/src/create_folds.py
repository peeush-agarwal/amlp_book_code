import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config

def run():
    # Load training data into dataframe
    train_data = pd.read_csv(config.TRAINING_FILE, header=None, names=config.COLUMNS)

    # Create new column 'kfold' with default value = -1
    train_data['kfold'] = -1

    # Fetch target column values
    y = train_data['income'].values

    # Create new object for StratifiedKFold
    kf = StratifiedKFold(n_splits=5)

    # Assign values to kfold column
    for f_, (t_, v_) in enumerate(kf.split(train_data, y)):
        train_data.loc[v_, 'kfold'] = f_
    
    # Save this dataframe with fold column
    train_data.to_csv(config.TRAINING_FOLDS_FILE, index=False)

if __name__ == "__main__":
    run()