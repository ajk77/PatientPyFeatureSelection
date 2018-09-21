"""
FeatureSelectionDriver.py
package PatientPyFeatureSelection
version 2.0
created by AndrewJKing.com|@andrewsjourney

This program demonstrates how to run RecursiveFeatureInclusion. 

To run, you will need to replace '/my_base_dir/' in __main__.

---LICENSE---
This file is part of PatientPyFeatureSelection

PatientPyFeatureSelection is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or 
any later version.

PatientPyFeatureSelection is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
"""
from RecursiveFeatureInclusion import determine_attribute_sets, staged_feature_inclusion
from joblib import Parallel, delayed
import multiprocessing
import numpy as np


def run_feature_selection(params):
    """
    This code runs feature selection for all input experiments.
    """
    def load_samples(file_name):
        samples_dict = {}
        with open(file_name, 'r') as f:
            for full_line in f:
                line = full_line.rstrip()
                if line:
                    if line[0] == '#':  # skip comment lines
                        continue
                    s_line = line.split('\t')
                    samples_dict[s_line[1] + '_' + s_line[3]] = [int(x) for x in s_line[4:]]
        return samples_dict

    mImp_data = np.load(params['mImp_filename'] + '.npy')
    mImp_names = np.load(params['mImp_filename'] + '_names.npy')
    mImp_sets_of_attributes, mImp_names_for_attribute_sets = determine_attribute_sets(mImp_names.flatten().tolist())
    rImp_data = np.load(params['rImp_filename'] + '.npy')
    rImp_names = np.load(params['rImp_filename'] + '_names.npy')
    rImp_sets_of_attributes, rImp_names_for_attribute_sets = determine_attribute_sets(rImp_names.flatten().tolist())

    target_samples = load_samples(params['target_samples_outfile'])
    feature_samples = load_samples(params['feature_samples_outfile'])
    target_feature_columns = load_list(params['target_feature_columns_file'])
    target_matrix = np.loadtxt(params['target_matrix_name'] + '.txt', delimiter=',')
    model_keys = target_samples.keys()
    target_col_indices = {}
    mImp_out_files_dict = {}
    rImp_out_files_dict = {}
    # ## populate target_col_indices, mImp_out_files_dict, and rImp_out_files_dict
    for key in model_keys:
        if key not in target_col_indices:
            target_col_indices[key] = int(key.split('_')[0])
            mImp_out_files_dict[key] = params['feature_selection_storage'] + key + '-mImp.txt'
            rImp_out_files_dict[key] = params['feature_selection_storage'] + key + '-rImp.txt'

    num_cores = multiprocessing.cpu_count()
    print ('='*10 + 'STARTING mIMP' + '='*10)
    result = Parallel(n_jobs=num_cores)(
        delayed(staged_feature_inclusion)(mImp_data[feature_samples[model_key], :], 
                                          target_matrix[~target_samples[model_key], target_col_indices[model_key]], 
                                          mImp_sets_of_attributes, 
                                          params['models_to_use'], 
                                          mImp_out_files_dict[model_key]) for model_key in model_keys)
    print ('='*10 + 'STARTING rIMP' + '='*10)
    result = Parallel(n_jobs=num_cores)(
        delayed(staged_feature_inclusion)(rImp_data[feature_samples[model_key], :], 
                                          target_matrix[~target_samples[model_key], target_col_indices[model_key]], 
                                          rImp_sets_of_attributes, 
                                          params['models_to_use'], 
                                          rImp_out_files_dict[model_key]) for model_key in model_keys)

    return


def populate_feature_selection_params(base_dir):
    """
    Populates parameter dictionary.

    # not all params are used 
    """
    params = {}
    case_dir = base_dir + 'complete_feature_files_labeling_cases/'
    out_dir = base_dir + 'feature_matrix_storage_labeling_cases/'
    params['feature_selection_storage'] = out_dir + 'feature_selection_storage/'
    params['mImp_filename'] = out_dir + 'full_labeling_mImp'
    params['rImp_filename'] = out_dir + 'full_labeling_rImp'
    params['feature_samples_outfile'] = out_dir + 'feature_samples_out.txt'
    params['target_samples_outfile'] = out_dir + 'target_samples_out.txt'
    params['target_feature_columns_file'] = case_dir + 'target_feature_columns.txt'
    params['target_matrix_name'] = case_dir + 'target_full_matrix'
    params['models_to_use'] = ['lr', 'sv', 'rf']
    params['mImp_test_filename'] = base_dir + 'feature_matrix_storage_evaluation_cases/full_evaluation_mImp'
    params['rImp_test_filename'] = base_dir + 'feature_matrix_storage_evaluation_cases/full_evaluation_rImp'
    params['result_out_dir'] = base_dir + 'evaluation_study_models/'
    params['participant_features_file'] = case_dir + 'participant_features'
    params['eval_participant_features_file'] = base_dir + 'complete_feature_files_evaluation_cases/participant_features'


if __name__=="__main__":
    # ## run feature selection
    params = populate_feature_selection_params('/my_base_dir/')
    run_feature_selection(params)
