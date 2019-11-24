# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:25:17 2019

@author: Rishav
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# The digits dataset
digits = datasets.load_digits()

#splitting Data into Training and Test Sets
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    test_size=0.25, 
                                                    random_state=0)

#random forest
clf = GaussianNB()
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)

#measuring model performance
score = accuracy_score(y_test,predicted)
print(score)

#evaluating the algorithm
print(confusion_matrix(y_test,predicted))
print(classification_report(y_test,predicted))