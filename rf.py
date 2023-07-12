from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('./train/datapoints.csv')

X = df.iloc[:, 0: -3].values
y = df['label'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

rf = RandomForestClassifier(max_depth=15, random_state=0, n_estimators=500, criterion='entropy')
rf.fit(X, y)

# # Make predictions for the test set
# y_pred_test = rf.predict(X_test)
#
# matrix = confusion_matrix(y_test, y_pred_test)
#
# # Build the plot
# plt.figure(figsize=(20, 15))
# sns.set(font_scale=1.4)
# sns.heatmap(matrix/np.sum(matrix), annot=True, annot_kws={'size': 10},
#             cmap=plt.cm.Greens)
#
# # Add labels to the plot
# class_names = ['rssi0', 'rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8', 'rssi9', 'rssi10',
#                'rssi11']
# tick_marks = np.arange(len(class_names))
# tick_marks2 = tick_marks + 0.5
# plt.xticks(tick_marks, class_names, rotation=45)
# plt.yticks(tick_marks2, class_names, rotation=0)
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.title('Confusion Matrix for Random Forest Model')
# plt.tight_layout()
# plt.show()

# accuracies = cross_val_score(estimator=rf, X=X_test, y=y_test)
# print("Average score: {}".format(accuracies.mean()))

filename = "./models/rf.pickle"
pickle.dump(rf, open(filename, "wb"))