# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 12:06:03 2019

@author: Rishav
"""
#The algorithm builds multiple models from randomly taken subsets of 
#train dataset and aggregates learners to build overall stronger learner.

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

# The digits dataset
digits = datasets.load_digits()

#splitting Data into Training and Test Sets
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    test_size=0.25, 
                                                    random_state=0)
#training bagging classifier
dtc = DecisionTreeClassifier(criterion="entropy")
bag_model=BaggingClassifier(base_estimator=dtc, n_estimators=100,
                            bootstrap=True)
bag_model=bag_model.fit(x_train,y_train)

#predict data and check the prediction accuracy
y_test_pred=bag_model.predict(x_test)
print(bag_model.score(x_test, y_test))
print(confusion_matrix(y_test, y_test_pred)) 

#checking accuracy by changing base estimator
lr = LogisticRegression(solver='lbfgs', multi_class='auto');
bnb = BernoulliNB()
gnb = GaussianNB()

base_methods=[lr, bnb, gnb, dtc]
for bm  in base_methods:
 print("Method: ", bm)
 bag_model=BaggingClassifier(base_estimator=bm, n_estimators=100,
                             bootstrap=True)
 bag_model=bag_model.fit(x_train,y_train)
 ytest_pred=bag_model.predict(x_test)
 print(bag_model.score(x_test, y_test))
 print(confusion_matrix(y_test, ytest_pred))