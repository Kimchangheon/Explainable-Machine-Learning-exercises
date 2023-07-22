from ce3_compare import set_name, compare, write_submission_txt, grade, set_idm
from feature_model import FeatureModel
import numpy as np
bike_path = "datasets/bike/rented_bikes_day_pre.csv"
bike_data = FeatureModel(bike_path)
bike_data.df = bike_data.df.drop(labels=["casual", "registered"], axis=1)
bike_data.add_target("cnt")
bike_data.add_all_features_but_target()

if __name__ == '__main__':
    from sklearn.inspection import permutation_importance

    X_train, y_train, X_val, y_val = bike_data.return_Xy(train_test_split=True)

    from sklearn.ensemble import GradientBoostingRegressor

    regr = GradientBoostingRegressor(random_state=1)
    regr.fit(X_train, y_train)
    regr.score(X_val, y_val)

    r = permutation_importance(regr, X_val, y_val, n_repeats=20, random_state=1)
    importances_mean, importances_std, importances = r.importances_mean, r.importances_std, r.importances
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{X_val.columns[i]:<22} | "
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")