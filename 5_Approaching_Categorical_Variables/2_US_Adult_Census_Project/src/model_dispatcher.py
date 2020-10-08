from sklearn import linear_model
import xgboost

models = {
    "log_reg": linear_model.LogisticRegression(),
    "xgb": xgboost.XGBClassifier(n_jobs=-1),
    "xgb_depth_7": xgboost.XGBClassifier(max_depth=7, n_jobs=-1)
}