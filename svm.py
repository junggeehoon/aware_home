from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pandas as pd
import pickle

df = pd.read_csv('./data/kalman.csv')

X = df.iloc[:, 0: -3].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

svm = SVC(kernel='rbf', gamma='scale', shrinking=False)
svm.fit(X_train, y_train)

accuracies = cross_val_score(estimator=svm, X=X_test, y=y_test)
print("Average score: {:.3f}".format(accuracies.mean()))

filename = "./models/svm.pickle"
pickle.dump(svm, open(filename, "wb"))