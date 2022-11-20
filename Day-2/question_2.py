import argparse
import numpy as np
from sklearn import datasets, svm, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump, load

parser = argparse.ArgumentParser(description='Classifier and Parameters')
parser.add_argument('--clf_name', type=str, required=True)
parser.add_argument('--random_state',type=int, required=True)
args = parser.parse_args()
digits = datasets.load_digits()
user_split = 0.1
n_samples = len(digits.images)
data = digits.data
if args.clf_name=='svm':
	clf = svm.SVC(C=0.2,gamma=0.001)
elif args.clf_name=='tree':
	clf = tree.DecisionTreeClassifier()
else:
	print("Invalid Model Import")
X_train, X_test, Y_train, Y_test = train_test_split(data, digits.target, test_size=0.1, shuffle=True, random_state = args.random_state)
clf.fit(X_train, Y_train)
print(" ")
print("Predictions:- ", clf.predict(X_test))
print(" ")
print("Test Accuracy:- ",accuracy_score(Y_test, clf.predict(X_test)))
print("Test Macro_F1:- ",f1_score(Y_test, clf.predict(X_test),average='macro'))
path = './models/svm_gamma=0.001_C=0.2.joblib'
details = ['test accuracy:'+str(accuracy_score(Y_test, clf.predict(X_test))),'test macro-f1:'+str(f1_score(Y_test, clf.predict(X_test),average='macro')),path]
with open('./results/'+args.clf_name+'_'+str(args.random_state)+'.txt', 'w') as f:
	for detail in details:
		f.write(detail)
		f.write('\n')
dump(clf,path)
