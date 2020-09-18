import pandas as pd
from sklearn.model_selection import StratifiedKFold

def k_folds_train(k=5):
    filename = '../input/train.csv' #'input/train.csv'
    df = pd.read_csv(filename)

    # Create new column and fill it with -1
    df['kfold'] = -1
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch target
    y = df['label']

    # StratifiedKFold object
    kfold = StratifiedKFold(n_splits=k)

    # Assign fold in 'kfold' column
    for i, (t_, v_) in enumerate(kfold.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = i

    # Save the data to 'train_folds.csv'
    df.to_csv('../input/train_folds.csv', index=False)

if __name__ == "__main__":
    k_folds_train(k=5)
