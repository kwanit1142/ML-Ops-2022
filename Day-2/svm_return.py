import numpy as np
from sklearn import datasets,svm
from sklearn.model_selection import train_test_split
from joblib import dump, load

digits = datasets.load_digits()
user_split = 0.1
n_samples = len(digits.images)
data = digits.data
clf = svm.SVC()
X_train, X_rest, Y_train, Y_rest = train_test_split(data, digits.target, test_size=user_split, shuffle=False, random_state=10)
print(type(X_train))
X_val, X_test, Y_val, Y_test = train_test_split(X_rest, Y_rest, test_size=0.5, shuffle=False, random_state=10)
clf.fit(X_train, Y_train)
print(clf.predict(X_test))
