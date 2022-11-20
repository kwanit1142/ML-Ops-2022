import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

X_v, X_t, Y_v, Y_t = 0,0,0,0
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples,-1))
user_random  = 10
X_train ,X_rest, Y_train, Y_rest = train_test_split(data, digits.target,test_size=0.5,shuffle=True, random_state=user_random)
X_val, X_test, Y_val, Y_test = train_test_split(X_rest, Y_rest, test_size=0.5, shuffle=True, random_state=user_random)
clf = svm.SVC(gamma=0.001)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)

def test_biased():
	pred1 = predictions[0]
	count=1
	for i in predictions[1:]:
		if pred1==i:
			count+=1
	assert count!=len(predictions)

def test_union():
	class_set = {pred for pred in predictions}
	assert len(class_set) == 10

def test_same_seed():
	count=0
	X_v, X_t, Y_v, Y_t = train_test_split(X_rest, Y_rest, test_size=0.5, shuffle=True, random_state=10)
	if np.array_equal(X_v, X_val) == np.array_equal(X_t, X_test) == np.array_equal(Y_v, Y_val) == np.array_equal(Y_t, Y_test) == True:
		count+=1
	assert count==1

def test_different_seed():
	count=0
	X_v, X_t, Y_v, Y_t = train_test_split(X_rest, Y_rest, test_size=0.5, shuffle=True, random_state=user_random-1)
	if np.array_equal(X_v,X_val) == np.array_equal(X_t,X_test) == np.array_equal(Y_v,Y_val) == np.array_equal(Y_t,Y_test) == False:
		count+=1
	assert count==1
