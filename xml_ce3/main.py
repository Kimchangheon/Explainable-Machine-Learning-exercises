# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

from ce3_compare import set_name, compare, write_submission_txt, grade, set_idm
from feature_model import FeatureModel
import numpy as np

import numpy as np
from scipy.integrate import cumtrapz
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

bike_path = "datasets/bike/rented_bikes_day_pre.csv"
bike_data = FeatureModel(bike_path)
bike_data.df = bike_data.df.drop(labels=["casual", "registered"], axis=1)
bike_data.add_target("cnt")
bike_data.add_all_features_but_target()

X,y = bike_data.return_Xy()
regr = RandomForestRegressor(random_state=1)
regr.fit(X,y)

def ALE(regr, feature_idx, X, bins=40):
    # Check if X is a pandas DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.values

    # 1. Divide feature into intervals
    X_sorted = np.sort(X[:, feature_idx])
    feature_min = X_sorted[0]
    feature_max = X_sorted[-1]
    intervals = np.linspace(feature_min, feature_max, bins+1)

    ale_values = []

    # 2. For each interval, and for each datapoint in a interval, calculate the prediction difference dy
    # by replacing x1 with the interval lower and upper limit
    for i in range(bins):
        lower = intervals[i]
        upper = intervals[i + 1]
        mask = (X[:, feature_idx] >= lower) & (X[:, feature_idx] < upper)

        if not np.any(mask):
            ale_values.append(0)
            continue

        X_lower = X[mask].copy()
        X_upper = X[mask].copy()
        X_lower[:, feature_idx] = lower
        X_upper[:, feature_idx] = upper

        dy = regr.predict(X_upper) - regr.predict(X_lower)

        # 3. For each interval average prediction differences dy(s)
        avg_dy = np.mean(dy)

        # 4. Integrate average prediction differences
        ale_values.append(avg_dy)

    # 5. Center integrated average prediction differences
    ale_values = cumtrapz(ale_values, intervals[:-1], initial=0)
    ale_values -= np.mean(ale_values)

    return intervals, ale_values


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    feature_idx = X.columns.get_loc("temp")  # specify the index of the feature for which you want to compute the ALE
    grid_size = 10
    intervals, ale_values = ALE(regr, feature_idx, X, bins=grid_size)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
