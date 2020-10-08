# US Adult Census project

## Prepare dataset

+ Add `input` folder
+ Download dataset from **[Kaggle's US Adult Census dataset](https://www.kaggle.com/johnolafenwa/us-census-data)**.
+ You get a zip file from above command. Unzip the file and move `adult-training.csv` and `adult-test.csv` files inside `/input` directory.
+ *Note: There are no header names given for the dataset*

## Explore the data

+ Add `notebooks` folder
+ Create a new notebook [explore_data.ipynb](notebooks/explore_data.ipynb) inside `notebooks` directory.

## Model building

+ After exploring the data, we'll start building different models using different algorithms (Logistic regression, Random Forest, etc.)
+ Add `src` folder
+ Add a new python file `ohe_logres.py` which will train a LogisticRegression model after transformation of one-hot encoding on categorical features.
  + This gives us AUC score of `~0.87` which is really good from the simple model.
+ Next, train a Label encoded XGBoost model in `lbl_xgb.py` file.
  + This also gives us AUC score of `~0.87` with no hyperparameter tuning.
  + AUC score of `~0.86` with *max_depth=7* and *n_estimators=200*.
  + Next, include numerical columns and default hyperparameters gives AUC score of `~0.92`. Much better than LogisticRegression model.
+ Next, train a Label encoded XGBoost model with Feature engineering in `lbl_xgb_feat_eng.py` file.
  + This also gives us AUC score of `~0.91` with no hyperparameter tuning.
  + AUC score of `~0.92` with *max_depth=7*.