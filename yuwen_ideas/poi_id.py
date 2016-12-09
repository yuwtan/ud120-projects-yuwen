#!/usr/bin/python

import sys
import pickle
import math
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import RandomizedPCA
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus',
                 'total_payments', 'total_stock_value',
                 'ratio_from_poi', 'ratio_to_poi',
                 'ratio_shared_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

my_dataset = data_dict
if 'TOTAL' in my_dataset:
    del my_dataset['TOTAL']

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

for person in my_dataset:
    # read dataset information
    data_point = my_dataset[person]

    from_poi_to_this_person = float(data_point['from_poi_to_this_person'])
    to_messages = float(data_point['to_messages'])

    from_this_person_to_poi = float(data_point['from_this_person_to_poi'])
    from_messages = float(data_point['from_messages'])

    # create new features
    if math.isnan(from_poi_to_this_person) or math.isnan(to_messages) or to_messages == 0:
        ratio_from_poi = 0
    else:
        ratio_from_poi = from_poi_to_this_person / to_messages

    if math.isnan(from_this_person_to_poi) or math.isnan(from_messages) or from_messages == 0:
        ratio_to_poi = 0
    else:
        ratio_to_poi = from_this_person_to_poi / from_messages

    data_point['ratio_from_poi'] = ratio_from_poi
    data_point['ratio_to_poi'] = ratio_to_poi

for person in my_dataset:
    # read dataset information
    data_point = my_dataset[person]

    shared_receipt_with_poi = float(data_point['shared_receipt_with_poi'])
    to_messages = float(data_point['to_messages'])

    total_stock_value = float(data_point['total_stock_value'])
    total_payments = float(data_point['total_payments'])

    # create new features
    if math.isnan(shared_receipt_with_poi) or math.isnan(to_messages) or to_messages == 0:
        ratio_shared_with_poi = 0
    else:
        ratio_shared_with_poi = shared_receipt_with_poi / to_messages

    if math.isnan(total_stock_value) or math.isnan(total_payments) or total_payments == 0:
        ratio_stock_to_payments = 0
    else:
        ratio_stock_to_payments = total_stock_value / total_payments

    data_point['ratio_shared_with_poi'] = ratio_shared_with_poi
    data_point['ratio_stock_to_payments'] = ratio_stock_to_payments

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

from sklearn import svm
clf = svm.SVC(kernel="rbf", C=10)

pca = RandomizedPCA(n_components=3, whiten=True).fit(features)
features = pca.transform(features)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
clf.fit(features_train, labels_train)

labels_pred = clf.predict(features_test)
print(metrics.classification_report( labels_test, labels_pred ))
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)