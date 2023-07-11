from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd

df = pd.read_csv('./data/datapoints.csv')

X = df.iloc[:, 0: -3].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dt = DecisionTreeClassifier(random_state=0, criterion='entropy')

# param_grid = {
#     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
#     'criterion': ['gini', 'entropy']
# }
#
# CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
# CV_rfc.fit(X_train, y_train)
#
# print(CV_rfc.best_params_)


dt.fit(X_train, y_train)

accuracies = cross_val_score(estimator=dt, X=X_test, y=y_test)
print("Average score: {:.3f}".format(accuracies.mean()))
# print(dt.score(X_test, y_test))
