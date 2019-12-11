# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:55:31 2019

@author: Rishav
"""

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# The digits dataset
digits = datasets.load_digits()

#splitting Data into Training and Test Sets
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    test_size=0.25, 
                                                    random_state=0)

#create new a knn model
knn = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
params_knn = {'n_neighbors': np.arange(1,25)}
#use gridsearch to test all values for n_neighbors
knn_gs = GridSearchCV(knn, params_knn, cv=5)
#fit model to training data
knn_gs.fit(x_train, y_train)

#save best model
knn_best = knn_gs.best_estimator_
#check best n_neigbors value
print(knn_gs.best_params_)

#create a new random forest classifier
rf = RandomForestClassifier()
#create a dictionary of all values we want to test for n_estimators
params_rf = {'n_estimators': [50, 100, 200]}
#use gridsearch to test all values for n_estimators
rf_gs = GridSearchCV(rf, params_rf, cv=5)
#fit model to training data
rf_gs.fit(x_train, y_train)

#save best model
rf_best = rf_gs.best_estimator_
#check best n_estimators value
print(rf_gs.best_params_)

#create a new logistic regression model
log_reg = LogisticRegression(solver='lbfgs', multi_class='auto')
#fit the model to the training data
log_reg.fit(x_train, y_train)

#test the three models with the test data and print their accuracy scores
print('knn: {}'.format(knn_best.score(x_test, y_test)))
print('rf: {}'.format(rf_best.score(x_test, y_test)))
print('log_reg: {}'.format(log_reg.score(x_test, y_test)))

#create a dictionary of our models
estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg)]
#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='hard')

#fit model to training data
ensemble.fit(x_train, y_train)
#test our model on the test data
ensemble_score = ensemble.score(x_test, y_test)
print(ensemble_score)

#evaluating the algorithm
predicted = ensemble.predict(x_test)
print(classification_report(y_test,predicted))


