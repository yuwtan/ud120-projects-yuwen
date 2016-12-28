#!/usr/bin/python

import sys
import pickle
import math
import numpy as np

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


print len(data_dict)

""" scatterplot salary and bonus """

for person in data_dict:
    # read person's information
    data_point = data_dict[person]
    salary = float(data_point['salary'])
    bonus = float(data_point['bonus'])
    shared_receipt_with_poi = float(data_point['shared_receipt_with_poi'])
    to_messages = float(data_point['to_messages'])

    # find people with strange email data
    if shared_receipt_with_poi > to_messages:
        print "Person with strange email information: \t", person
        print "shared_receipt_with_poi is: \t", shared_receipt_with_poi
        print "to_messages is: \t", to_messages


    # scatterlpot salary and bonus
    # pyplot.scatter(salary, bonus)

# pyplot.xlabel("salary")
# pyplot.ylabel("bonus")
# pyplot.show()

""" find out the outlier """
for person in data_dict:
    data_point = data_dict[person]
    if float(data_point['salary']) > 10000000:
        print "Person with salary higher than 10 million: \t", person

""" remove the outlier """
data_dict.pop('TOTAL', 0);
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0);

""" correct email data """
data_dict['GLISAN JR BEN F']['shared_receipt_with_poi'] = data_dict['GLISAN JR BEN F']['to_messages']

for person in data_dict:

    # read dataset information
    data_point = data_dict[person]

    from_poi_to_this_person = float(data_point['from_poi_to_this_person'])
    shared_receipt_with_poi = float(data_point['shared_receipt_with_poi'])
    to_messages = float(data_point['to_messages'])

    from_this_person_to_poi = float(data_point['from_this_person_to_poi'])
    from_messages = float(data_point['from_messages'])

    total_stock_value = float(data_point['total_stock_value'])
    total_payments = float(data_point['total_payments'])

    # create new features
    if math.isnan(from_poi_to_this_person) or math.isnan(to_messages) or to_messages == 0:
        ratio_from_poi_to_this_person = 0
    else:
        ratio_from_poi_to_this_person = from_poi_to_this_person / to_messages

    if math.isnan(from_this_person_to_poi) or math.isnan(from_messages) or from_messages == 0:
        ratio_from_this_person_to_poi = 0
    else:
        ratio_from_this_person_to_poi = from_this_person_to_poi / from_messages

    if math.isnan(shared_receipt_with_poi) or math.isnan(to_messages) or to_messages == 0:
        ratio_shared_receipt_with_poi = 0
    else:
        ratio_shared_receipt_with_poi = shared_receipt_with_poi / to_messages

    if math.isnan(total_stock_value) or math.isnan(total_payments) or total_payments == 0:
        ratio_stock_to_payments = 0
    else:
        ratio_stock_to_payments = total_stock_value / total_payments

    # save new features
    data_point['ratio_from_poi_to_this_person'] = ratio_from_poi_to_this_person
    data_point['ratio_from_this_person_to_poi'] = ratio_from_this_person_to_poi
    data_point['ratio_shared_receipt_with_poi'] = ratio_shared_receipt_with_poi
    data_point['ratio_stock_to_payments'] = ratio_stock_to_payments


features_list = ['poi',
                 'salary', 'deferral_payments',
                 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options',
                 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                 'ratio_from_poi_to_this_person', 'ratio_from_this_person_to_poi',
                 'ratio_shared_receipt_with_poi', 'ratio_stock_to_payments'
                ]

data1 = featureFormat(data_dict, features_list, sort_keys = True)


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0, n_jobs=-1)

labels, features = targetFeatureSplit(data1)
forest.fit(features, labels)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(len(features[0])):
    print("%2d) %-*s %f" % (f+1, 30, features_list[indices[f]+1], importances[indices[f]]))


features_list_final = ['poi',
                       'exercised_stock_options',
                       'ratio_from_this_person_to_poi',
                       'bonus',
                       'ratio_shared_receipt_with_poi',
                       'total_stock_value',
                       'ratio_stock_to_payments',
                       'other',
                       'deferred_income',
                       'expenses',
                       'restricted_stock'
                       ]

data = featureFormat(data_dict, features_list_final, sort_keys = True)


from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler


param_svm = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100]}
svr = SVC()
clf_svm = GridSearchCV(svr, param_svm)

clf_nb = GaussianNB()

param_lr = {'penalty':('l1', 'l2'), 'C':[1, 100, 10000]}
lr = LogisticRegression()
clf_lr = GridSearchCV(lr, param_lr)

"""
param_rf = {'max_depth':[3, 4]}
rf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
clf_rf = GridSearchCV(rf, param_rf)
"""

param_dt = {'max_depth':[4, 6, 8]}
dt = DecisionTreeClassifier()
clf_dt = GridSearchCV(dt, param_dt)

sss = StratifiedShuffleSplit(n_splits=1000, random_state=42)

labels, features = targetFeatureSplit(data)

scaler = MinMaxScaler()

def test_clf(features, labels, clf, PCA_k):
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    for train_idx, test_idx in sss.split(features, labels):
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)

        pca = PCA(n_components=PCA_k, whiten=True).fit(features_train)
        features_train = pca.transform(features_train)
        features_test = pca.transform(features_test)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1

    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    precision = 1.0*true_positives/(true_positives+false_positives)
    recall = 1.0*true_positives/(true_positives+false_negatives)
    f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
    f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
    return total_predictions, accuracy, precision, recall, f1, f2


print test_clf(features, labels, clf_nb, 3)
print test_clf(features, labels, clf_lr, 3)
print test_clf(features, labels, clf_svm, 3)
print test_clf(features, labels, clf_dt, 3)

print test_clf(features, labels, clf_nb, 6)
print test_clf(features, labels, clf_lr, 6)
print test_clf(features, labels, clf_svm, 6)
print test_clf(features, labels, clf_dt, 6)

print test_clf(features, labels, clf_nb, 9)
print test_clf(features, labels, clf_lr, 9)
print test_clf(features, labels, clf_svm, 9)
print test_clf(features, labels, clf_dt, 9)

"""

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

pca = RandomizedPCA(n_components=2, whiten=True).fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)


from sklearn.naive_bayes import GaussianNB

clf_nb = GaussianNB()
labels_nb = clf_nb.fit(features_train, labels_train).predict(features_test)
print('Naive Bayes:')
print(metrics.classification_report(labels_test, labels_nb))

labels, features = targetFeatureSplit(data2)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

pca = RandomizedPCA(n_components=2, whiten=True).fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)

from sklearn.naive_bayes import GaussianNB

clf_nb = GaussianNB()
labels_nb = clf_nb.fit(features_train, labels_train).predict(features_test)
print('Naive Bayes:')
print(metrics.classification_report(labels_test, labels_nb))

clf = clf_nb
# dump_classifier_and_data(clf, my_dataset, features_list)
"""