from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process.kernels import RBF
import pandas as pd
import pickle

df = pd.read_csv('./data/sample.csv')

X = df.iloc[:, 0: -3].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

kernel = 1.0 * RBF(1.0)
GPClf = GaussianProcessClassifier(kernel=kernel,
                                  random_state=0)

GPClf.fit(X_train, y_train)

accuracies = cross_val_score(estimator=GPClf, X=X_test, y=y_test)
print("Average score: {:.3f}".format(accuracies.mean()))

filename = "./models/gaussian.pickle"
pickle.dump(GPClf, open(filename, "wb"))

# prob = model.predict_proba([[-72,	-79,	-74,	-92,	-90,	-96,	-79,	-78,	-98,	-89,	-92,	-95]])
# print(prob)

# for i in range(3):
#     for j in range(3):
#         # print("Probability for ({}, {}) = ({:.3f}, {:.3f})".format(i, j, prob[0][0][i], prob[1][0][j]))
#         print("Probability for ({}, {}) = {:.3f}".format(i, j, prob[0][0][i] * prob[1][0][j]))

