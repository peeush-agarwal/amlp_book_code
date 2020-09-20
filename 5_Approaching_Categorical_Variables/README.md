# Approaching categorical variables

## What are categorical variables?

+ Nominal variables
+ Ordinal variables

+ Binary
+ Cyclic

## Prepare dataset

+ Add `input` folder
+ Download dataset from **[Kaggle's Categorical Feature Encoding Challenge II](https://www.kaggle.com/c/cat-in-the-dat-ii/overview)** using [Kaggle CLI](https://www.kaggle.com/docs/api):
  ``` shell
  $ kaggle competitions download -c cat-in-the-dat-ii
  ```
+ You get a zip file from above command. Unzip the file and move `train.csv` and `test.csv` files inside `/input` directory.

## Explore the data

+ Add `notebooks` folder
+ Create a new notebook [explore_data.ipynb](notebooks/explore_data.ipynb) inside `notebooks` directory.

## Model building

+ After exploring the data, we'll start building different models using different algorithms (Logistic regression, Random Forest, etc.)
+ Add `src` folder
+ Add a new python file `ohe_logres.py` which will train a LogisticRegression model after transformation of one-hot encoding on features.
