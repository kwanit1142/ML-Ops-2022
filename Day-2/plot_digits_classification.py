"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause
from statistics import median, pstdev
# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, tree, metrics
from sklearn.model_selection import train_test_split
from skimage import transform
from tabulate import tabulate
###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

def new_data(data,size):
        new_features = np.array(list(map(lambda img: transform.resize(
                                img.reshape(8,8),(size,size),mode='constant',preserve_range=True).ravel(),data)))
        return new_features

digits = datasets.load_digits()
'''
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
'''
###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
user_split_list = [0.1,0.2,0.3,0.4,0.5]
# flatten the images
n_samples = len(digits.images)
#data = digits.images.reshape((n_samples, -1))
user_size = 8
count=1
data = new_data(digits.data,user_size)
table_final = [['Category','Test Accuracy for SVMs','Test Accuracy for Decision-Tree-Classifier']]
for user_split in user_split_list:
        entry=[]
        print(" ")
        print('For Image Size = '+str(user_size)+'x'+str(user_size)+' and Train-Val-Test Split => '+str(int(100*(1-user_split)))+
                '-'+str(int(50*user_split))+'-'+str(int(50*user_split)))

        GAMMA = [0.001,0.01,0.1]
        C = [1,0.1,10]

        best_gam = 0
        best_c = 0
        best_mean_acc=0
        best_train=0
        best_val=0
        best_test=0
        pred_test = 0
        table = [['Gamma','C','Training Acc.','Val (Dev) Acc.','Test Acc.','Min Acc.','Max Acc.','Median Acc.','Mean Acc.']]
        for GAM in GAMMA:
                for c in C:
                        hyper_params = {'gamma':GAM, 'C':c}
                        clf = svm.SVC()
                        clf.set_params(**hyper_params)
                        X_train, X, y_train, y = train_test_split(data, digits.target, test_size=user_split, shuffle=False)
                        x_val, x_test, y_val, y_test = train_test_split(X,y,test_size=0.5,shuffle=False)
                        clf.fit(X_train, y_train)
                        predicted_val = clf.predict(x_val)
                        predicted_train = clf.predict(X_train)
                        predicted_test = clf.predict(x_test)
                        accuracy_val = 100*metrics.accuracy_score(y_val,predicted_val)
                        accuracy_train = 100*metrics.accuracy_score(y_train, predicted_train)
                        accuracy_test = 100*metrics.accuracy_score(y_test, predicted_test)
                        mean_acc = (accuracy_val + accuracy_train + accuracy_test)/3
                        min_acc = min([accuracy_train,accuracy_val,accuracy_test])
                        max_acc = max([accuracy_val,accuracy_train,accuracy_test])
                        median_acc = median([accuracy_val,accuracy_train,accuracy_test])
                        table.append([GAM,c,str(accuracy_train)+'%',str(accuracy_val)+'%',str(accuracy_test)+'%',str(min_acc)+'%',
                                        str(max_acc)+'%',str(median_acc)+'%',str(mean_acc)+'%'])
                        if accuracy_test>best_test:
                                best_gam = GAM
                                best_c = c
                                best_train=accuracy_train
                                best_val=accuracy_val
                                best_test=accuracy_test
                                pred_test = predicted_test
        pred_unis, pred_counts = np.unique(pred_test, return_counts=True)
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        print(" ")
        print('Best Hyperparameters (Gamma and C) => '+str(best_gam)+' and '+str(best_c))
        print('Train, Val (Dev) and Test Accuracies => '+str(best_train)+'%, '+str(best_val)+'%, '+str(best_test)+'%')
        print(" ")
        entry.append(count)
        entry.append(best_test)
        max_dep = [40.80,120]
        max_leaf = [450,900,1350]

        best_depth = 0
        best_leaf = 0
        best_mean_accd=0
        best_traind=0
        best_vald=0
        best_testd=0
        pred_testd=0
        table_ii = [['Max_Depth','Max_Leaf_Nodes','Training Acc.','Val (Dev) Acc.','Test Acc.','Min Acc.','Max Acc.','Median Acc.','Mean Acc.']]
        for dep in max_dep:
                for leaf in max_leaf:
                        hyper_paramsd = {'max_depth':dep, 'max_leaf_nodes':leaf}
                        clfd = tree.DecisionTreeClassifier()
                        clfd.set_params(**hyper_paramsd)
                        X_train, X, y_train, y = train_test_split(data, digits.target, test_size=user_split, shuffle=False)
                        x_val, x_test, y_val, y_test = train_test_split(X,y,test_size=0.5,shuffle=False)
                        clfd.fit(X_train, y_train)
                        predicted_val = clfd.predict(x_val)
                        predicted_train = clfd.predict(X_train)
                        predicted_test = clfd.predict(x_test)
                        accuracy_val = 100*metrics.accuracy_score(y_val,predicted_val)
                        accuracy_train = 100*metrics.accuracy_score(y_train, predicted_train)
                        accuracy_test = 100*metrics.accuracy_score(y_test, predicted_test)
                        mean_acc = (accuracy_val + accuracy_train + accuracy_test)/3
                        min_acc = min([accuracy_train,accuracy_val,accuracy_test])
                        max_acc = max([accuracy_val,accuracy_train,accuracy_test])
                        median_acc = median([accuracy_val,accuracy_train,accuracy_test])
                        table_ii.append([dep,leaf,str(accuracy_train)+'%',str(accuracy_val)+'%',str(accuracy_test)+'%',str(min_acc)+'%',
                                        str(max_acc)+'%',str(median_acc)+'%',str(mean_acc)+'%'])
                        if accuracy_test>best_testd:
                                best_depth = dep
                                best_leaf = leaf
                                best_traind=accuracy_train
                                best_vald=accuracy_val
                                best_testd=accuracy_test
                                pred_testd = predicted_test
        pred_uniqued, pred_countd = np.unique(pred_testd, return_counts=True)
        print(tabulate(table_ii, headers='firstrow', tablefmt='fancy_grid'))
        entry.append(best_testd)
        print(" ")
        print('Best Hyperparameters (Max_Features and ccp) => '+str(best_feat)+' and '+str(best_ccp))
        print('Train, Val (Dev) and Test Accuracies => '+str(best_traind)+'%, '+str(best_vald)+'%, '+str(best_testd)+'%')
        print(" ")
        table_final.append(entry)
        print("Predictions of Best SVM:-", dict(zip(pred_unis, pred_counts)))
        print("Predictions of Best DTC:-", dict(zip(pred_uniqued, pred_countd)))

svm_list = [float(table_final[i][1]) for i in range(1,5)]
dtc_list = [float(table_final[i][2]) for i in range(1,5)]
mean_svm = sum(svm_list)/len(svm_list)
mean_dtc = sum(dtc_list)/len(dtc_list)
std_svm = pstdev(svm_list)
std_dtc = pstdev(dtc_list)
table_final.append(['mean',mean_svm,mean_dtc])
table_final.append(['std',std_svm,std_dtc])
print(tabulate(table_final, headers='firstrow', tablefmt='fancy_grid'))
###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.
'''
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
'''
###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.
'''
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
'''
###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.
'''
cm = metrics.confusion_matrix(y_test, predicted)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp = disp.plot()
'''
#disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
#disp.figure_.suptitle("Confusion Matrix")
#print(f"Confusion matrix:\n{disp.confusion_matrix}")
'''
plt.show()
'''
