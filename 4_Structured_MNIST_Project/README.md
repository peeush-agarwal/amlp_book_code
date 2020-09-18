# Project structure for any ML project

This demonstrates a project structure for any ML project. I'll demo it using MNIST dataset. The project structure is as follows:

+ *input*: This contains all input files like CSV files, images, videos, etc. Any dataset file that is required across the project.
+ *src*: This contains all code files we can have in a project. For an example:
  + create_folds.py
  + train.py
  + inference.py
  + models.py
  + model_dispatcher.py
  + config.py
+ *models*: This contains all trained models which are trained or used in the project.
+ *notebooks*: This contains all notebooks(*.ipynb) used for EDA and plotting.
+ README.md: This is the markdown file that briefs about the project and steps if user need to follow to reach the desired output.
+ LICENSE: This is the simple text file that consists of a license for the project.

# MNIST project

In this project, we want to classify the given pixel values (28x28 = 784) of an image containing a digit.

## Prepare the dataset

+ Download the dataset from [here](https://www.kaggle.com/oddrationale/mnist-in-csv)
+ Copy the files into `input` folder with the names `train.csv` and `test.csv`

## Data exploration

+ Explore the data in CSV files to check the distribution of label in train dataset. [Check notebook here](notebooks/check_data.ipynb).
+ This gives us good indication that `accuracy_score` can be used for evaluation.

## Model building

+ Run `src/create_folds.py` to create folds for `train.csv` and push it in `train_folds.csv`.
  ``` shell
  $ pythons create_folds.py
  ```
+ Run `src/run.sh` to train the model and save trained models into `models` folder.
  ``` shell
  $ sh run.sh {MODEL_NAME}
  ```
  MODEL_NAME is the key name from model_dispatcher.py
  + dt_gini: `DecisionTreeClassifier` with `criterion=gini`
  + dt_entropy: `DecisionTreeClassifier` with `criterion=entropy`

