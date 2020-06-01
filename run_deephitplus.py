'''
Main file to run DeepHitPlus models

Code by Carl Rietschel and Changhee Lee
https://github.com/carlr67/deephitplus
'''

# Import general python module requirements and functions
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time as timelib
import shutil
from copy import copy
from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Import DeepHit modules and functions
import import_data as impt
import utils_network as utils
from get_features import *
from train_deephit import *
from class_DeepHitPlus import Model_Single
from utils_eval import c_index

overall_start_time = timelib.time()

##### SETTINGS
'''
    Load data and data settings
'''
(x_dim), (full_data, time, label), (mask1, mask2), full_feat_list = impt.import_dataset_SYNTHETIC(norm_mode = 'standard')
eval_time                   = [1*12, 3*12, 5*12] # Evaluation times (for C-index) for testing

'''
    Choose DeepHitPlus version and related settings
    - features: Specifies whether to run DeepHitPlus, FilterDeepHitPlus or HybridDeepHitPlus
    - path_to_importances: Path to importances folder (only for HybridDeepHitPlus)
'''
features                    = 'all' # 'all' or 'filter_...' or 'hybrid_..._...' or 'sparse' (see Readme)

# Only relevant if running with features='all' preparing for HybridDeepHitPlus
calculate_importances       = 'ON' # ON / OFF (only works in combination with features='all')
NUM_PERMUTATIONS            = 20

# Only relevant if running HybridDeepHitPlus (features='hybrid_..._...')
path_to_immportances        = 'output/results-all_rsON/a10_b30_s01_mb050_kp6_lr1E-04_hs100_ns2_hf050-050_nf2-2' # Only required if running HybridDeepHitPlus



'''
    Choose run settings
'''
random_search_mode          = 'ON' # ON / OFF
RS_ITERATION                = 40 # How many iterations for random search
CV_ITERATION                = 5 # How many iterations for K-Fold cross-validation
cv_to_search                = [1, 0, 0, 0, 0] # 0 for "don't perform search on this iteration"
valid_mode                  = 'ON' # ON / OFF

'''
    Random number generation settings
'''
seed                        = 1234 # Random seed for dataset splits (training, validation, testing)
rs_seed                     = 1234 # Random seed for generating parameter sets for random search


##### HYPER-PARAMETERS
'''
    Values for the hyperparameters
    - DEFAULT_param_dict: What default value to use (if no random search), plus flag for whether the hyperparameter is event-specific (in which case an array is required)
    - SET_param_dict: Defines the search space for random search (if random search is enabled) - one array with all possibilities per hyperparameter
'''

DEFAULT_param_dict = { # Tuple (default_value, event-specific?)
    'alpha'               : (1.0,                   0), # for log-likelihood loss
    'beta'                : (3.0,                   0), # for ranking loss
    'gamma'               : ([0.0001, 0.0001],      1), # only for SparseDeepHitPlus - multiplier of regularisation term
    'sigma1'              : (0.1,                   0), # for ranking loss
    'mb_size'             : (50,                    0), # minibatch size
    'keep_prob'           : (0.6,                   0), # dropout keep probability
    'lr_train'            : (1e-4,                  0), # training learning rate
    'h_dim_shared'        : (50,                    0), # number of nodes per layer in shared subnetwork
    'num_layers_shared'   : (5,                     0), # number of layers in shared subnetwork
    'h_dim_FC'            : ([50, 50],              1), # number of nodes per layer in event-specific subnetworks
    'num_layers_FC'       : ([5, 5],                1), # number of layers in event-specific subnetworks
    'importancecutoff'    : ([0.001, 0.001],        1), # only for HybridDeepHitPlus with cutoff - cutoff for feature selection by importance for each event
    'top'                 : ([10, 10],              1) # only for FilterDeepHitPlus or HybridDeepHitPlus with top N - top number of features selected for each event
}

SET_param_dict = { # Search sets
    'alpha'               : [1],
    'beta'                : [3.0],
    'gamma'               : [0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001],
    'sigma1'              : [0.1],
    'mb_size'             : [50],
    'keep_prob'           : [0.6],
    'lr_train'            : [1e-4],
    'h_dim_shared'        : [50, 100],
    'num_layers_shared'   : [1, 2],
    'h_dim_FC'            : [50, 100],
    'num_layers_FC'       : [1, 2],
    'importancecutoff'    : [0.0001, 0.001, 0.01],
    'top'                 : [5, 10, 20]
}

'''
    Further parameter choices
    - itersettings
        - 'iteration': Number of iterations for training (or max number if validation mode is used (recommended))
        - 'require_improvement': In validation mode, training stops if no improvement is found after this number of steps
        - 'check_improvement': In validation mode, validation performance is checked after this number of steps
    - active_fn: Choice of activation function for neural network
    - initial_W: Choice of weights initializer for neural network
'''
itersettings = {
    'iteration': 100000,
    'require_improvement': 5000,
    'check_improvement': 1000
}
active_fn                   = tf.nn.relu
initial_W                   = tf.contrib.layers.xavier_initializer()


##### DATASET
'''
    num_Category            = max event/censoring time * 1.2 (to make enough time horizon)
    num_Event               = number of events i.e. len(np.unique(label))-1
    max_length              = maximum number of measurements
    x_dim                   = data dimension (num features) as array: [shared network, event-specific network 1, ...]
    mask1, mask2            = used for cause-specific network (FCNet structure)
'''

_, num_Event, num_Category  = np.shape(mask1)  # dim of mask1: [subj, num_Event, Num_Category]
val_eval_time               = eval_time # Evaluation times for validation
feature_mode                = features.split("_")[0]

# Arrays for storing final results
FINALCV = np.zeros([num_Event, len(eval_time), CV_ITERATION])
FINALRS = np.zeros([num_Event, len(eval_time), RS_ITERATION])


# K-fold Cross-Validation for Test Data
kf = KFold(n_splits=CV_ITERATION, shuffle=True, random_state=seed)
cv_iter = 0
best_parameter_name = ''

modes = "_".join([features, "rs" + random_search_mode])

BEST_param_dict = {key: copy(DEFAULT_param_dict[key][0]) for key in DEFAULT_param_dict}

# Loop for each cross-validation (train and test split)
for train_index, test_index in kf.split(full_data):
    print()
    print()
    print("############################ CROSS VALIDATION", cv_iter, "############################")
    print("Running feature setting:", features)

    if random_search_mode == 'OFF':
        S_ITERATION = 1
        searchmode = None
    elif random_search_mode == 'ON' and cv_to_search[cv_iter] != 1:
        S_ITERATION = 1
        searchmode = None
    elif random_search_mode == 'ON' and cv_to_search[cv_iter] == 1:
        searchmode = 'random'
        print("Conducting RANDOM search")
        S_ITERATION = RS_ITERATION

    # Loop for random search iterations (or if no random search, just one iteration)
    TRIED_param_dicts = []
    min_s_valid = [0,0,0]
    best_s_itr = 0
    for s_itr in range(S_ITERATION):
        s_itr_start_time = timelib.time() # Time how long each random search iteration takes

        cur_param_dict = get_next_param_dict(searchmode, TRIED_param_dicts, BEST_param_dict, SET_param_dict, DEFAULT_param_dict, num_Event, seed=rs_seed+s_itr)
        TRIED_param_dicts.append(deepcopy(cur_param_dict))

        tr_data     = full_data[train_index]
        tr_time     = time[train_index]
        tr_label    = label[train_index]
        tr_mask1    = mask1[train_index]
        tr_mask2    = mask2[train_index]

        te_data     = full_data[test_index]
        te_time     = time[test_index]
        te_label    = label[test_index]
        te_mask1    = mask1[test_index]
        te_mask2    = mask2[test_index]

        # Split into training and validation data
        (tr_data,va_data, tr_time,va_time, tr_label,va_label,
         tr_mask1,va_mask1, tr_mask2,va_mask2)  = train_test_split(tr_data, tr_time, tr_label, tr_mask1, tr_mask2, test_size=0.2, random_state=seed)

        # Get feature set if needed from hyperparameters
        feat_list = get_feat_list(features=features, num_Event=num_Event, eval_time=eval_time, data=tr_data, full_feat_list=full_feat_list, times=tr_time, labels=tr_label, param_dict=cur_param_dict, cv_iter=cv_iter, path_to_immportances=path_to_immportances)
        x_dim, tr_data, va_data, te_data = apply_features(full_feat_list=full_feat_list, feat_list=feat_list, tr_data=tr_data, va_data=va_data, te_data=te_data)
        print("x-dim:", str(x_dim))

        # Train the model
        model, sess, saver, parameter_name, valid_perf, valid_perf_event, file_path = train_deephit(
            searchmode,
            modes,
            valid_mode,
            itersettings,
            val_eval_time,
            tr_data,
            tr_label,
            tr_time,
            tr_mask1,
            tr_mask2,
            va_data,
            va_time,
            va_label,
            cur_param_dict,
            x_dim,
            active_fn,
            initial_W,
            features,
            s_itr=s_itr,
            best_s_itr=best_s_itr,
            best_parameter_name=best_parameter_name)

        print("    Validation performance: " + str(round(valid_perf,3)) + "  |  " + str([round(x,3) for x in valid_perf_event]))

        # Update random search - check if improvement on previous best hyperparameters
        tmp_s_valid = valid_perf_event

        if (np.mean(tmp_s_valid) > np.mean(min_s_valid)):
            min_s_valid = tmp_s_valid
            best_s_itr  = s_itr

            # Take best model from early stopping and save it as current best model for random search
            saver.restore(sess, file_path + '/models-' + modes + '/' + parameter_name + '/model_v1')
            saver.save(sess, file_path + '/models-' + modes + '/search/model_rsv1')

            BEST_param_dict = deepcopy(cur_param_dict)
            best_parameter_name      = parameter_name

            print('SEARCH updated.... ' + '(' + best_parameter_name + ') ' + str('%.4f' %(np.mean(tmp_s_valid))))

        # Remove directory of model that is now in folder search/
        shutil.rmtree(file_path + '/models-' + modes + '/' + parameter_name + '/')

        print("Time for current iteration:", timelib.time() - s_itr_start_time)
        print()

    # Before testing use the best architecture so far if have done random search
    alpha, beta, gamma, sigma1, mb_size, keep_prob, lr_train, h_dim_shared, num_layers_shared, h_dim_FC, num_layers_FC, importancecutoff, top = unzip_hyperparam_dict(BEST_param_dict, num_Event)

    # Get dataset again since x_dim could have changed
    tr_data     = full_data[train_index]
    tr_time     = time[train_index]
    tr_label    = label[train_index]
    tr_mask1    = mask1[train_index]
    tr_mask2    = mask2[train_index]

    te_data     = full_data[test_index]
    te_time     = time[test_index]
    te_label    = label[test_index]
    te_mask1    = mask1[test_index]
    te_mask2    = mask2[test_index]

    (tr_data,va_data, tr_time,va_time, tr_label,va_label,
     tr_mask1,va_mask1, tr_mask2,va_mask2)  = train_test_split(tr_data, tr_time, tr_label, tr_mask1, tr_mask2, test_size=0.2, random_state=seed)

    feat_list = get_feat_list(features=features, num_Event=num_Event, eval_time=eval_time, data=tr_data, full_feat_list=full_feat_list, times=tr_time, labels=tr_label, param_dict=BEST_param_dict, cv_iter=cv_iter, path_to_immportances=path_to_immportances)
    x_dim, tr_data, va_data, te_data = apply_features(full_feat_list=full_feat_list, feat_list=feat_list, tr_data=tr_data, va_data=va_data, te_data=te_data)
    print("x-dim:", str(x_dim))

    input_dims                  = { 'x_dim'         : x_dim,
                                    'num_Event'     : num_Event,
                                    'num_Category'  : num_Category }
    network_settings            = { 'h_dim_shared'      : int(h_dim_shared),
                                  'num_layers_shared' : num_layers_shared,
                                  'h_dim_FC'          : h_dim_FC,
                                  'num_layers_FC'     : num_layers_FC,
                                  'active_fn'         : active_fn,
                                  'initial_W'         : initial_W }

    ##### PREDICTION & EVALUATION ON TEST DATASET
    if valid_mode == 'ON':

        ##### CREATE DEEPHIT NETWORK
        tf.reset_default_graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        model = Model_Single(sess, "FHT_DeepHit", mb_size, input_dims, network_settings, feature_mode)
        saver = tf.train.Saver()

        saver.restore(sess, file_path + '/models-' + modes + '/search/model_rsv1')


    ### PREDICTION
    pred = model.predict(te_data)

    ### EVALUATION
    result1 = np.zeros([num_Event, len(eval_time)])

    for t, t_time in enumerate(eval_time):
        eval_horizon = int(t_time)

        if eval_horizon >= num_Category:
            print('ERROR: evaluation horizon is out of range')
            result1[:, t] = -1
        else:
            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until eval_time
            for k in range(num_Event):
                result1[k, t] = c_index(risk[:,k], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                # since we compare risk_scores, the true label is that event occurs before time horizon

    FINALCV[:, :, cv_iter] = result1

    ### SAVE RESULTS

    # Make directories if they don't yet exist
    file_path = 'output'
    if not os.path.exists(file_path + '/results-' + modes + '/' + best_parameter_name + '/'):
        os.makedirs(file_path + '/results-' + modes + '/' + best_parameter_name + '/')
    if not os.path.exists(file_path + '/results-' + modes + '/' + best_parameter_name + '/models/'):
        os.makedirs(file_path + '/results-' + modes + '/' + best_parameter_name + '/models/')

    row_header = []
    for t in range(num_Event):
        row_header.append(' event_' + str(t+1))

    col_header1 = []
    for t in eval_time:
        col_header1.append(str(t) + 'mo c_index')

    # c-index result
    df1 = pd.DataFrame(result1, index = row_header, columns=col_header1)
    df1.to_csv('./' + file_path + '/results-' + modes + '/' + best_parameter_name + '/result_' + features + '_CINDEX_cv' + str(cv_iter) + '.csv')

    if feature_mode == 'attentive':
        # attention results
        resultatt = model.sess.run(model.att_out, feed_dict={model.x: te_data, model.keep_prob: 1.0})
        names = ['Participant', 'Event', 'Feature']

        attindex = pd.MultiIndex.from_product([range(0, len(te_data)), range(0, num_Event), full_feat_list], names=names)
        dfatt = pd.DataFrame({'Attention': resultatt.flatten()}, index=attindex)
        dfatt.to_csv('./' + file_path + '/results-' + modes + '/' + best_parameter_name + '/attention_cv' + str(cv_iter) + '.csv')


    ### Save trained model for later use
    saver.save(sess, file_path + '/results-' + modes + '/' + best_parameter_name + '/models/dhmodel_cv' + str(cv_iter))
    shutil.rmtree(file_path + '/models-' + modes + '/')


    ### PRINT RESULTS
    print('========================================================')
    print('CV_' + str(cv_iter) + ' RESULTS >  (' + best_parameter_name + ', ' + modes + ')')
    print('--------------------------------------------------------')
    print('- CV_' + str(cv_iter) + ' C-INDEX: ')
    print(df1)
    print('--------------------------------------------------------')


    ### CALCULATE FEATURE IMPORTANCES (on validation data) - only works if all features are used (features='all')

    if calculate_importances and features == 'all':
        dfindex = ['Event', 'Feature', 'Permutation', 'Eval. horizon']
        dfcols = ['Importance']
        result = pd.DataFrame(columns = dfindex + dfcols)
        result.set_index(keys=dfindex, inplace=True)

        orig_data = np.copy(va_data)

        # Iterate through all features
        for feat_number in range(len(full_feat_list)):
            feat = full_feat_list[feat_number]

            # Average a number of random permutations to get stability in the importance figures
            for perm_iter in range(NUM_PERMUTATIONS):
                permuted_data = np.copy(orig_data)

                # One feature affects multiple parts of the data due to the structure of feat_list (shared subnetwork and one for each event-specific sub-network) - make sure all these are permuted
                for subnet in range(len(feat_list)):
                    permuted_data[:, len(full_feat_list)*subnet + feat_number] = np.random.permutation(permuted_data[:, len(full_feat_list)*subnet + feat_number])

                pred = model.predict(permuted_data)

                resultfeat = np.zeros([num_Event, len(eval_time)])
                for t, t_time in enumerate(eval_time):
                    eval_horizon = int(t_time)

                    if eval_horizon >= num_Category:
                        print('ERROR: evaluation horizon is out of range')
                        resultfeat[:, t] = -1
                    else:
                        # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
                        risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until eval_time

                        # Calculate permutation importance for each eval horizon and event
                        for k in range(num_Event):
                            resultfeat[k, t] = c_index(risk[:,k], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)

                            result.loc[k+1, feat, perm_iter, "Delta t = " + "{:03d}".format(t_time)] = result1[k,t] - resultfeat[k,t]
                            # since we compare risk_scores, the true label is that event occurs before time horizon

                print("Computed feature importance " + str(feat_number+1) + "/" + str(len(full_feat_list)) + " (" + feat + ") #" + str(perm_iter) + ": " + str(np.average(result1-resultfeat, axis=1).round(4)))

        result.to_csv('./' + file_path + '/results-' + modes + '/' + best_parameter_name + '/result_' + features + '_VAL-IMPORTANCES_cv' + str(cv_iter) + '.csv')

    cv_iter += 1

### FINAL MEAN/STD FOR CV RUN

# c-index result - write to output
df2_mean = pd.DataFrame(np.mean(FINALCV, axis=2), index = row_header, columns=col_header1)
df2_std  = pd.DataFrame(np.std(FINALCV, axis=2), index = row_header, columns=col_header1)
df2_mean.to_csv('./' + file_path + '/results-' + modes + '/result_' + features + '_CINDEX_FINAL_MEAN.csv')
df2_std.to_csv('./' + file_path + '/results-' + modes + '/result_' + features + '_CINDEX_FINAL_STD.csv')


### PRINT RESULTS
print()
print()
print('*********************************************************************************************************************')
print('FINAL RESULTS > (' + best_parameter_name + ', ' + modes + ')')
print('--------------------------------------------------------')
print('- FINAL C-INDEX: ')
print(df2_mean)
print('--------------------------------------------------------')
print("Overall time required:", timelib.time() - overall_start_time)
