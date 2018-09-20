"""
RecurrsiveFeatureInclusion.py
package PatientPyFeatureSelection
version 1.1
created by AndrewJKing.com|@andrewsjourney

This program performs feature selection by first determining which classes of features are informative. 
Then recursive feature elimination is performed on those classes.
Remaining feature indices are returned. 

Pass in:
A matrix of instances and class labels for those instances.
Out put:
Indices that were selected for.
"""
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFECV


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


def staged_feature_inclusion(x_, y, sets_of_attributes, models_to_use, out_file):
    """
    This function runs staged_feature_inclusion method of feature selection. 
    It is intended for use when muliple features are constructed from a single variable.
    E.g., a time searies of heart rate measurements espanded into min, max, slopes, etc.
    The base variable must start the variable name and be underscore separated 
    (e.g., heatrate_max_value and heartrate_most_recent_value).

    sets_of_attributes is the first returned item from determine_attribute_sets()
    """
    print (out_file)
    out_f = open(out_file, 'w')
    models = {}
    models['lr'] = LogisticRegression(penalty='l2', random_state=42)
    models['sv'] = SVC(C=1, probability=True, random_state=42)
    models['rf'] = clf_rf = RandomForestClassifier(random_state=42)
    for model_name in models_to_use:
        informative_attributes = []
        rfecv = RFECV(estimator=models[model_name], step=1, scoring='roc_auc')
        # determine keep columns
        for set_index, current_set_of_attributes in enumerate(sets_of_attributes):
            x_current = x_[:, current_set_of_attributes]
            try:
                # ## determine staged inclusion for even rows
                scores = cross_validate(models[model_name], x_current, y, scoring='roc_auc', return_train_score=False) 
                # ^this is a test to see if the set of features is predictive. If not, there is no reason to run the slow rfecv.
                if scores['test_score'].mean() > 0.55:  # determine if set should be kept
                    if model_name == 'sv':  # 1 = svc  # SV is separate b/c it used to not work with rfevc. New versions of sklearn fix this 
                        informative_attributes += current_set_of_attributes
                    else:  # 0 = lr & 2 = rf
                        rfecv.fit(x_current, y)  # recursive feature elimination
                        if rfecv.grid_scores_.mean() > 0.6:
                            informative_attributes += [current_set_of_attributes[x] for x in np.where(rfecv.support_)[0].tolist()]
                            # ^keep most important features
            except (ValueError, IndexError):
                pass
        out_f.write(model_name + ':' + ','.join([str(x) for x in informative_attributes]) + '\n')
    out_f.close()
    return 

