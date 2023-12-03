import sys
sys.path.append("/Users/jaffa/AppData/Local/Programs/Python/Python310/Lib/site-packages")

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
from sklearn.dummy import DummyClassifier
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures


# Get the file. The full dataset. Place the mnist file in the same folder as
# the .py file
def load_data_set():
    data_file = os.path.join(os.path.dirname(__file__), "mnist.csv")
    
    mnist_dataset = load_dataset("csv", data_files=data_file)
    mnist_dataset = mnist_dataset.shuffle(seed=42)  # shuffle the dataset
    # no. of unique labels: 10 unique labels
    output = len(np.unique(mnist_dataset['train']['label']))
    mnist_labels = mnist_dataset['train']['label']
    mnist_features = mnist_dataset['train'].remove_columns(['label'])

    return mnist_dataset, output, mnist_labels, mnist_features

def set_up_data_sets(dataset):
    r = dataset['train'].train_test_split(test_size=0.88095)
    train_set, test_set = r['train'], r['test']
    
    return r, train_set, test_set

def split_data_set(features, labels):
    #split the mnist dataset for the logistic regression and SVM models
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels, test_size=0.88095, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test

def show_data_set(dataset):
    dataset.set_format(type="pandas")
    df = dataset['train'][:]
    return df

def reduced_show_data_set(dataset):
    dataset.set_format(type="pandas")
    df = dataset[:]
    return df

def run_digit_analysis(df):
    ax = df["label"].value_counts(ascending=True).plot.barh()
    
    for container in ax.containers:
        ax.bar_label(container)

    plt.title("Frequency of each hand-written number")

def print_results(model_description, y_pred, y_train, cv_score, score, precision, recall, f1, confusion):
    print()
    print("The results for the {}:".format(model_description))
    print('----------------')
    print("The cross validation result is:", cv_score)
    print("Estimated test labels: ", y_pred[:20])
    print("True test labels:      ", y_train[:20])
    print("The accuracy is:" ,score)
    print("The average precision is:", precision)
    print("The average recall is:", recall)
    print("The average f1-score is:", f1)
    print("The confusion matrix is: \n", confusion)

# base classifier for comparison
def base_classifier_run():
    description = 'most common baseline classifier'
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    
    # Use 3-fold cross validation to evaluate the model
    cv_score = cross_val_score(dummy_clf, X_train, y_train, cv=3, scoring="accuracy")
    y_train_pred_multinomial = cross_val_predict(dummy_clf, X_train, y_train, cv=3)
    
    
    y_pred = dummy_clf.predict(X_train)
    
    confusion = confusion_matrix(y_train, y_pred)
    score = dummy_clf.score(X_train, y_train)
    
    #print(y_test)
    precision = precision_score(y_train, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_train, y_pred, average='weighted')
    f1 = f1_score(y_train, y_pred, average='weighted')

    print_results(description, y_pred, y_train, cv_score, score, precision, recall, f1, confusion)

    return dummy_clf, y_train_pred_multinomial

# Model design.
def logistic_classifier_run(description, C, X_train, y_train, X_test, y_test):

    model = LogisticRegression(penalty='l1', C=C, fit_intercept=True,
                               solver='saga', max_iter=10000, verbose=0, multi_class='multinomial')
    clf = model.fit(X_train, y_train)

    #beta = clf.coef_
    # Use 3-fold cross validation to evaluate the model
    cv_score = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
    y_train_pred_multinomial = cross_val_predict(clf, X_train, y_train, cv=3)
    

    y_pred = clf.predict(X_train) #(X_test)
    confusion = confusion_matrix(y_train, y_pred) #(y_test, y_pred)
    score = clf.score(X_train, y_train) #(X_test, y_test)

    precision = precision_score(y_train, y_pred, average='weighted') #(y_test, y_pred, average='weighted')
    recall = recall_score(y_train, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_train, y_pred, average='weighted')

    print_results(description, y_pred, y_train, cv_score, score, precision, recall, f1, confusion)

    return clf, y_train_pred_multinomial

# SVM Classifiers [Polynomial Kernel and Gaussian Kernel]
def poly_svm_classifier_run(description, C, degree, X_train, y_train, X_test, y_test):

    model = make_pipeline(StandardScaler(), 
                          SVC(kernel="poly", degree=degree, coef0=1, C=C))
    
    polynomial_svm_clf = model.fit(X_train, y_train)

    
    # Use 3-fold cross validation to evaluate the model
    cv_score = cross_val_score(polynomial_svm_clf, X_train, y_train, cv=3, scoring="accuracy")
    y_train_pred_multinomial = cross_val_predict(polynomial_svm_clf, X_train, y_train, cv=3)
    

    y_pred = polynomial_svm_clf.predict(X_train) #(X_test)
    confusion = confusion_matrix(y_train, y_pred) #(y_test, y_pred)
    score = polynomial_svm_clf.score(X_train, y_train) #(X_test, y_test)

    precision = precision_score(y_train, y_pred, average='weighted') #(y_test, y_pred, average='weighted')
    recall = recall_score(y_train, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_train, y_pred, average='weighted')

    print_results(description, y_pred, y_train, cv_score, score, precision, recall, f1, confusion)

    return polynomial_svm_clf, y_train_pred_multinomial

def gauss_svm_classifier_run(description, C, gamma, X_train, y_train, X_test, y_test):

    model = make_pipeline(StandardScaler(), 
                          SVC(kernel="rbf", gamma=gamma, C=C))
    
    gaussian_svm_clf = model.fit(X_train, y_train)

    
    # Use 3-fold cross validation to evaluate the model
    cv_score = cross_val_score(gaussian_svm_clf, X_train, y_train, cv=3, scoring="accuracy")
    y_train_pred_multinomial = cross_val_predict(gaussian_svm_clf, X_train, y_train, cv=3)
    

    y_pred = gaussian_svm_clf.predict(X_train) #(X_test)
    confusion = confusion_matrix(y_train, y_pred) #(y_test, y_pred)
    score = gaussian_svm_clf.score(X_train, y_train) #(X_test, y_test)

    precision = precision_score(y_train, y_pred, average='weighted') #(y_test, y_pred, average='weighted')
    recall = recall_score(y_train, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_train, y_pred, average='weighted')

    print_results(description, y_pred, y_train, cv_score, score, precision, recall, f1, confusion)

    return gaussian_svm_clf, y_train_pred_multinomial



#generate the datasets to be used for training
mnist_dataset, output, mnist_labels, mnist_features = load_data_set()
r, train_set, test_set = set_up_data_sets(mnist_dataset) # this isn't necessary

df = show_data_set(mnist_dataset)
mnist_features = reduced_show_data_set(mnist_features) 
mnist_features = mnist_features.values.tolist() 
X_train, X_test, y_train, y_test = split_data_set(mnist_features, mnist_labels)


# plot the frequency of each hand-written digit (1 is the most frequent)
run_digit_analysis(df)

# A. Use a dummy classifier for comparison purposes
base_clf, base_cv = base_classifier_run()

# B. Classify Using Logistic Regression
# Set regularization parameter
# smaller values of C constrain the model more. In the L1 penalty case, this leads to sparser solutions.
#1 is the default value for C

l1_clf_0_01, l1_cv_0_01 = logistic_classifier_run('Logistic regression', 0.01, X_train, y_train, X_test, y_test)
print("/")
l1_clf_0_1, l1_cv_0_1 = logistic_classifier_run('Logistic regression', 0.1, X_train, y_train, X_test, y_test)
print("/")
l1_clf_1, l1_cv_1 = logistic_classifier_run('Logistic regression', 1.0, X_train, y_train, X_test, y_test)


# C. Classify using SVM [Polynomial and Gaussian]
# 1. Polynomial
poly_53_clf, poly_53_cv = poly_svm_classifier_run('Polynomial SVM', 5, 3, X_train, y_train, X_test, y_test)
print("/")
poly_510_clf, poly_510_cv = poly_svm_classifier_run('Polynomial SVM', 5, 10, X_train, y_train, X_test, y_test)
print("/") 
poly_13_clf, poly_13_cv = poly_svm_classifier_run('Polynomial SVM', 1, 3, X_train, y_train, X_test, y_test)
print("/")
poly_110_clf, poly_110_cv = poly_svm_classifier_run('Polynomial SVM', 1, 10, X_train, y_train, X_test, y_test)
print("/") 

# 2. Gaussain
gauss_101_clf, gauss_101_cv = gauss_svm_classifier_run('Gaussain SVM', 0.001, 0.1, X_train, y_train, X_test, y_test)
print("/")
gauss_1001_clf, gauss_1001_cv = gauss_svm_classifier_run('Gaussain SVM', 1000, 0.1, X_train, y_train, X_test, y_test)
print("/")
gauss_105_clf, gauss_105_cv = gauss_svm_classifier_run('Gaussain SVM', 0.001, 5, X_train, y_train, X_test, y_test)
print("/")                    
gauss_1005_clf, gauss_1005_cv = gauss_svm_classifier_run('Gaussain SVM', 1000, 5, X_train, y_train, X_test, y_test)