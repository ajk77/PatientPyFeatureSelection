"""
RecurrsoveFeatureInclusion.py
package patientpy_feature_selection
version 1.0
created by AndrewJKing.com|@andrewsjourney


"""


import numpy as np
import matplotlib.pyplot as plt
import warnings

import time
import pickle
import datetime

from os import listdir, walk
from os.path import isfile, join
from Regression_Imputer import *
from joblib import Parallel, delayed
import multiprocessing

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.svm import SVC

print_model_equations = False  # Set this variable to true when models are wanted in human readable format.
export_training_data = False  # Set this variable to true when exporting training_data[]
local_path = 'D:/LEMR_archives/4_machine_learning/'


def load_list(dir):
    """
    Loads a newline separated file into a list
    """
    ll = []
    with open(dir, 'r') as f:
        for line in f:
            ll.append(line.rstrip())
    return ll


# #################################################################################################################### #
"""
Pipeline 12 code
This pipeline is an updated version with but in five-fold processing. 
It utilizes the complete feature set.
Data sets: targets (all > 10 positive); sample type (all samples); imputation (median and regression). 
"""


def determine_attribute_sets(d_names):
    """
    Determine unique attribute types and expanded sets
    """
    attribute_types = []  # unique attribute root name
    for name in d_names:
        if name.split('_')[0] not in attribute_types:
            attribute_types.append(name.split('_')[0])

    d_sets_of_attributes = []  # expanded set of an attribute
    d_names_for_attribute_sets = []  # names for each attribute in an expanded set
    for idx, attribute_type in enumerate(attribute_types):
        curr_attribute_columns = []
        curr_attribute_names = []
        for idx2, name in enumerate(d_names):
            if name.split('_')[0] == attribute_type:
                curr_attribute_columns.append(idx2)
                curr_attribute_names.append(name)

        d_sets_of_attributes.append(curr_attribute_columns)
        d_names_for_attribute_sets.append(curr_attribute_names)

    return [d_sets_of_attributes, d_names_for_attribute_sets]


def sequencial_five_fold(number_of_instances):
    """
    Returns two lists of lists that contain train and test indices.
    """
    test_folds = [[x for x in range(f, number_of_instances, 5)] for f in range(5)]
    train_folds = [[]] * 5
    for f in range(5):
        for idx, test_fold in enumerate(test_folds):
            if idx != f:
                train_folds[f] = train_folds[f] + test_fold
    return [train_folds, test_folds]


def participant_folds():
    """
    Returns two lists of lists that contain train and test indices.
    """
    train_folds = []
    test_folds = []
    participant_case_rows = load_list(local_path + 'complete_feature_files/case_rows_by_participant_folds.txt')
    for row in participant_case_rows:
        if row:
            test_folds.append([int(x) for x in row.rstrip().split(',')])
    for f in range(len(test_folds)):
        temp = []
        for x in test_folds[:f]:
            temp = temp + x
        for x in test_folds[f+1:]:
            temp = temp + x
        train_folds.append(temp)
    return [train_folds, test_folds]


def staged_feature_inclusion(x_, y, model, s_model_index, s_sets_of_attributes):
    informative_attributes = []
    rfecv = RFECV(estimator=model, step=1, scoring='roc_auc')
    # determine keep columns
    for set_index, current_set_of_attributes in enumerate(s_sets_of_attributes):
        x_current = x_[:, current_set_of_attributes]
        try:
            # ## determine staged inclusion for even rows
            scores = cross_validate(model, x_current, y, scoring='roc_auc', return_train_score=False)
            if scores['test_score'].mean() > 0.5:
                if s_model_index == 1:  # 1 = svc
                    informative_attributes += current_set_of_attributes
                else:  # 0 = lr & 2 = rf
                    rfecv.fit(x_current, y)  # recursive feature elimination
                    if rfecv.grid_scores_[-1] > 0.6:
                        reduced_set_ = [current_set_of_attributes[x] for x in np.where(rfecv.support_)[0].tolist()]
                        informative_attributes += reduced_set_
        except (ValueError, IndexError):
            pass

    return informative_attributes


def target_feature_selection(t_index, t_data, x_train_f, sets_of_attributes_f, train_f, type, pre=False):
    clf_lr = LogisticRegression(penalty='l2', random_state=42)
    clf_svc = SVC(C=1, probability=True, random_state=42)
    clf_rf = RandomForestClassifier(random_state=42)
    if not pre:
        pre = 'D:/LEMR_archives/4_machine_learning/model_storage/pipeline12'
    out_f = open(pre + type + '/features_for_' + str(t_index) + '.txt', 'w')

    # ## for each of five folds ## #
    for fold in range(len(x_train_f)):
        x_train = x_train_f[fold]
        y_train = t_data[train_f[fold]]
        sets_of_att = sets_of_attributes_f[fold]

        keep_features_1 = staged_feature_inclusion(x_train, y_train, clf_lr, 0, sets_of_att)
        keep_features_2 = staged_feature_inclusion(x_train, y_train, clf_svc, 1, sets_of_att)
        keep_features_3 = staged_feature_inclusion(x_train, y_train, clf_rf, 2, sets_of_att)

        out_f.write('model:LR,fold:' + str(fold) + '|' + str(keep_features_1) + '\n')
        out_f.write('model:SV,fold:' + str(fold) + '|' + str(keep_features_2) + '\n')
        out_f.write('model:RF,fold:' + str(fold) + '|' + str(keep_features_3) + '\n')
    out_f.close()

    print 'finished target: ', t_index

    return


def target_feature_selection_single_fold(t_index, t_data, x_train, sets_of_att, pre_out_path):
    clf_lr = LogisticRegression(penalty='l2', random_state=42)
    clf_svc = SVC(C=1, probability=True, random_state=42)
    clf_rf = RandomForestClassifier(random_state=42)

    out_f = open(pre_out_path + '/features_for_' + str(t_index) + '.txt', 'w')

    keep_features_1 = staged_feature_inclusion(x_train, t_data, clf_lr, 0, sets_of_att)
    keep_features_2 = staged_feature_inclusion(x_train, t_data, clf_svc, 1, sets_of_att)
    keep_features_3 = staged_feature_inclusion(x_train, t_data, clf_rf, 2, sets_of_att)

    out_f.write('model:LR,fold:0|' + str(keep_features_1) + '\n')
    out_f.write('model:SV,fold:0|' + str(keep_features_2) + '\n')
    out_f.write('model:RF,fold:0|' + str(keep_features_3) + '\n')
    out_f.close()

    print 'finished target: ', t_index

    return


if __name__ == '__main__':
    """
    Main for five fold
    """
    # determine folds
    target_types = ['_original', '_normalized']
    impute_types = ['-regression', '-median']
    t_type = target_types[0]
    i_type = impute_types[1]

    target_data = np.genfromtxt(local_path + 'target_features' + t_type + '/targets.txt', delimiter=',')
    target_names = load_list(local_path + 'target_features' + t_type + '/target_names.txt')

    num_cores = multiprocessing.cpu_count()
    target_indices = range(target_data.shape[1])

    # ## load fold related information ## #
    train_folds, test_folds = sequencial_five_fold(target_data.shape[0])
    if False:
        x_train_folds = []
        x_test_folds = []
        x_names_folds = []
        sets_of_attributes_folds = []
        names_for_attribute_sets_folds = []
        for f in range(5):
            x_train_folds.append(np.load(local_path + 'model_storage/complete_feature' + i_type + '-train-fold' +
                                         str(f) + '.npy'))
            x_test_folds.append(np.load(local_path + 'model_storage/complete_feature' + i_type + '-test-fold' +
                                        str(f) + '.npy'))
            x_names_folds.append(np.load(local_path + 'model_storage/complete_feature' + i_type + '-names-fold' +
                                         str(f) + '.npy'))
            sets_of_attributes, names_for_attribute_sets = determine_attribute_sets(x_names_folds[f])
            sets_of_attributes_folds.append(sets_of_attributes)
            names_for_attribute_sets_folds.append(names_for_attribute_sets)

        data_structures = [x_train_folds, x_test_folds, x_names_folds, sets_of_attributes_folds]
        np.save(local_path + 'model_storage/pipeline12_data' + i_type + '_structures', data_structures, allow_pickle=True)

    if False:  # create and store structures with user features
        user_features = np.loadtxt(local_path + 'complete_feature_files/participant_features.txt', delimiter=',')
        user_features_names = np.asarray(load_list(local_path + 'complete_feature_files/participant_folds_p_order.txt'))
        x_train_folds = []
        x_test_folds = []
        x_names_folds = []
        sets_of_attributes_folds = []
        names_for_attribute_sets_folds = []
        for f in range(len(train_folds)):
            x_train_folds.append(np.load(local_path + 'model_storage/complete_feature' + i_type + '-train-fold' +
                                         str(f) + '.npy'))
            x_train_folds[f] = np.concatenate((x_train_folds[f], user_features[train_folds[f], :]), axis=1)
            x_test_folds.append(np.load(local_path + 'model_storage/complete_feature' + i_type + '-test-fold' +
                                        str(f) + '.npy'))
            x_test_folds[f] = np.concatenate((x_test_folds[f], user_features[test_folds[f], :]), axis=1)
            x_names_folds.append(np.load(local_path + 'model_storage/complete_feature' + i_type + '-names-fold' +
                                         str(f) + '.npy'))
            x_names_folds[f] = np.concatenate((x_names_folds[f], user_features_names), axis=0)
            sets_of_attributes, names_for_attribute_sets = determine_attribute_sets(x_names_folds[f])
            sets_of_attributes_folds.append(sets_of_attributes)
            names_for_attribute_sets_folds.append(names_for_attribute_sets)

        data_structures = [x_train_folds, x_test_folds, x_names_folds, sets_of_attributes_folds]
        np.save(local_path + 'model_storage/pipeline12_data' + i_type + '-structures_w_users', data_structures,
                allow_pickle=True)

    if False:  # create and store structures with user features and diagnosis
        user_features = np.loadtxt(local_path + 'complete_feature_files/participant_features.txt', delimiter=',')
        user_features_names = np.asarray(load_list(local_path + 'complete_feature_files/participant_folds_p_order.txt'))
        user_features = np.loadtxt(local_path + 'complete_feature_files/participant_features.txt', delimiter=',')
        user_features_names = np.asarray(load_list(local_path + 'complete_feature_files/participant_folds_p_order.txt'))
        diagnosis_features = np.loadtxt(local_path + 'complete_feature_files/diagnosis_features.txt',
                                        delimiter=',')  # isAKF
        diagnosis_features = diagnosis_features.reshape((diagnosis_features.shape[0], 1))
        diagnosis_features_names = np.asarray(['is_AKF'])
        x_train_folds = []
        x_test_folds = []
        x_names_folds = []
        sets_of_attributes_folds = []
        names_for_attribute_sets_folds = []
        for f in range(len(train_folds)):
            x_train_folds.append(np.load(local_path + 'model_storage/complete_feature' + i_type + '-train-fold' +
                                         str(f) + '.npy'))
            print x_train_folds[f].shape,
            x_train_folds[f] = np.concatenate((x_train_folds[f], user_features[train_folds[f], :],
                                               diagnosis_features[train_folds[f]]), axis=1)
            x_test_folds.append(np.load(local_path + 'model_storage/complete_feature' + i_type + '-test-fold' +
                                        str(f) + '.npy'))
            print x_test_folds[f].shape,
            x_test_folds[f] = np.concatenate((x_test_folds[f], user_features[test_folds[f], :],
                                              diagnosis_features[test_folds[f]]), axis=1)
            x_names_folds.append(np.load(local_path + 'model_storage/complete_feature' + i_type + '-names-fold' +
                                         str(f) + '.npy'))
            print x_names_folds[f].shape
            x_names_folds[f] = np.concatenate((x_names_folds[f], user_features_names, diagnosis_features_names), axis=0)
            print x_train_folds[f].shape, x_test_folds[f].shape, x_names_folds[f].shape
            sets_of_attributes, names_for_attribute_sets = determine_attribute_sets(x_names_folds[f])
            sets_of_attributes_folds.append(sets_of_attributes)
            names_for_attribute_sets_folds.append(names_for_attribute_sets)

        data_structures = [x_train_folds, x_test_folds, x_names_folds, sets_of_attributes_folds]
        np.save(local_path + 'model_storage/pipeline12_data' + i_type + '-structures_w_U_and_D', data_structures,
                allow_pickle=True)

    if False:
        data_structures = np.load(local_path + 'model_storage/pipeline12_data' + i_type + '_structures.npy')
        x_train_folds = data_structures[0]
        x_test_folds = data_structures[1]
        x_names_folds = data_structures[2]
        sets_of_attributes_folds = data_structures[3]

        print 'start'

        result = Parallel(n_jobs=num_cores)(
            delayed(target_feature_selection)(idx, target_data[:, idx], x_train_folds, sets_of_attributes_folds,
                                              train_folds, i_type + t_type) for idx in target_indices)
    print 'fin'
