#!/usr/bin/python

import sys
import pickle
import math
from matplotlib import pyplot
import warnings
import numpy as np

warnings.filterwarnings('ignore')
from sklearn import preprocessing

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

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

data_dict1 = {}
data_dict2 = {}
for person in data_dict:
    if math.isnan(float(data_dict[person]['to_messages'])) and \
        math.isnan(float(data_dict[person]['from_messages'])):
        data_dict2[person] = data_dict[person]
    else:
        data_dict1[person] = data_dict[person]

print len(data_dict1)
print len(data_dict2)

features_list1 = ['poi',
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
features_list2 = ['poi',
                 'salary', 'deferral_payments',
                 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options',
                 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees',
                 'ratio_stock_to_payments'
                ]

data1 = featureFormat(data_dict1, features_list1, sort_keys = True)
data2 = featureFormat(data_dict2, features_list2, sort_keys = True)
labels, features = targetFeatureSplit(data1)
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(features, labels)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(len(features[0])):
    print("%2d) %-*s %f" % (f+1, 30, features_list1[indices[f]+1], importances[indices[f]]))

labels, features = targetFeatureSplit(data2)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
forest.fit(features, labels)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(len(features[0])):
    print("%2d) %-*s %f" % (f+1, 30, features_list2[indices[f]+1], importances[indices[f]]))
