bike_path = "datasets/bike/rented_bikes_day_pre.csv"
bike_data = FeatureModel(bike_path)
bike_data.df = bike_data.df.drop(labels=["casual", "registered"], axis=1)
bike_data.add_target("cnt")
bike_data.add_all_features_but_target()
X_train, y_train, X_test, y_test = bike_data.return_Xy(train_test_split=True)

from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor([64, 32], random_state=1, max_iter=3_000)
mlp.fit(X_train.to_numpy(), y_train.to_numpy())

mlp.score(X_test, y_test)

import lime
from lime import lime_tabular

X_train.columns
Index(['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
       'weathersit', 'temp', 'hum', 'windspeed', 'days_since_01_01_2011'],
      dtype='object')

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.to_numpy(), # Q: Why does the algorithm need that? suppose no correlated
    feature_names=list(X_train.columns),
    mode="regression",
    categorical_features=np.arange(7), # Q: Why does the algorithm need that? ex) season cannot be 2.5
    random_state=1
)

suppress_warnings()
exp = explainer.explain_instance(
    X_test.to_numpy()[0], mlp.predict, num_features=5
)

exp.show_in_notebook()

exp.as_list()

[('temp <= 8.61', -1806.3062361108077),
 ('season=1', -681.4413278801666),
 ('mnth=2', 662.2461590034777),
 ('371.50 < days_since_01_01_2011 <= 555.25', 558.4031333484863),
 ('weathersit=1', 426.38613362789926)]
)

#### Exercise 4.5

Explain the highest-cnt/bikes-rented test-datapoint of the bike-dataset using the `explainer` as defined above. For this local explanation return the string representation of the feature with the highest negative impact on the `cnt`-variable. Set `num_features=10`

Re-instantiate the `explainer` to reset the `random_state`

def ex_4_5() -> str:
    return "14.0 <= temp"
