from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

MODELS = {
    "rf": RandomForestClassifier(n_jobs=-1),
    "xgb": xgb.XGBClassifier(
        n_jobs=-1, 
        max_depth=7, 
        n_estimators=200)
}