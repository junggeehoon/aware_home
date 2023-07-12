from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle

df = pd.read_csv('./train/datapoints.csv')
df = df.dropna()

X = df.iloc[:, 0: -3].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)


# param_grid = {
#     'estimator__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# }
#
# CV_knn = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)
# CV_knn.fit(X_train, y_train)
# #
# print(CV_knn.best_params_)

# test = [[-86, -87, -73, -68, -81, -78, -91, 0, -95, -86, 0, -94]]
# print(neigh.predict(test))

# knn.fit(X_train, y_train)
# # print(knn.score(X_test, y_test))

# accuracies = cross_val_score(estimator=knn, X=X_test, y=y_test)
# print("Average score: {}".format(accuracies.mean()))


filename = "./models/knn.pickle"

pickle.dump(knn, open(filename, "wb"))
