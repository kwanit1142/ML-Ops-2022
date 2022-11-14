import numpy as np
from sklearn import datasets,svm
from sklearn.model_selection import train_test_split
from skimage import transform
from joblib import dump, load

def new_data(data,size):
	new_features = np.array(list(map(lambda img: transform.resize(img.reshape(8,8),(size,size),mode='constant',preserve_range=True).ravel(),data)))
	return new_features

digits = datasets.load_digits()
user_split = 0.1
n_samples = len(digits.images)
user_size=8
data = new_data(digits.data,user_size)
GAMMA = 0.001
C = 1
hyper_params = {'gamma':GAMMA, 'C':C}
clf = svm.SVC()
clf.set_params(**hyper_params)
X_train, X, Y_train, Y = train_test_split(data, digits.target, test_size=user_split, shuffle=False)
clf.fit(X_train, Y_train)
dump(clf,'best_svm_model.joblib')
test = load('best_svm_model.joblib')
print(test.predict(X))

