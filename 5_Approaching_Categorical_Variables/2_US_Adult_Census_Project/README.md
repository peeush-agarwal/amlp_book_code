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
