# Approaching (Almost) Any ML problem book's code for practice

Practice code from above book. 

1. [Supervised and unsupervised learning](/1_Supervised_Unsupervised_Learning/)
1. [Cross validation](/2_Cross_Validation/)
1. [Evaluation metrics](/3_Evaluation_Metrics/)
1. [Project structure for any ML project](/4_Structured_MNIST_Project/)
1. [Approaching categorical variables](/5_Approaching_Categorical_Variables/)
   1. [OneHot encoding + Logistic Regression model](/5_Approaching_Categorical_Variables/src/ohe_logres.py)  
      This gives us AUC score of ~0.78 which is good. As the AUC score is in range of 0-1 and 1 being the perfect model.
   1. [LabelEncoding](/5_Approaching_Categorical_Variables/src/lbl_rf.py)
      1. Random Forest model  
         + This gives us AUC score of ~0.71 which is worse than Logistic regression model.
         + This model also takes more time and space compared to Logistic regression model.
         + This implies that we should never ignore basic model when training for the problem.
      1. XGBoost model
         + This gives us AUC score of ~0.76 which is better than RandomForest model, but still not better than Logistic regression model.
         + This model also takes more time and space compared to Logistic regression and RandomForest models.