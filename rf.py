from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import pickle

train = pd.read_csv('./train/datapoints.csv')
test = pd.read_csv('./test/datapoints.csv')

X_train = train.iloc[:, 0: -3].values
y_train = train['label'].values

X_test = test.iloc[:, 0: -3].values
y_test = test['label'].values

rf = RandomForestClassifier(max_depth=15, random_state=0, n_estimators=500, criterion='entropy')
rf.fit(X_train, y_train)


accuracies = cross_val_score(estimator=rf, X=X_test, y=y_test)
print("Average score: {}".format(accuracies.mean()))

filename = "./models/rf.pickle"
pickle.dump(rf, open(filename, "wb"))