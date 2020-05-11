import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import time as timelib
import shutil
from importlib import import_module
from copy import copy
from copy import deepcopy
# import sys

# from termcolor import colored
from tensorflow.contrib.layers import fully_connected as FC_Net
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA


import import_data as impt
import utils_network as utils
from get_features import *

from utils_eval import c_index

overall_start_time = timelib.time()


_EPSILON = 1e-08

##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.log(x + 1e-8)

def div(x, y):
    return tf.div(x, (y + 1e-8))

def f_get_minibatch(mb_size, x, label, time, mask1, mask2):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb = x[idx, :].astype(np.float32)
    k_mb = label[idx, :].astype(np.float32) # censoring(0)/event(1,2,..) label
    t_mb = time[idx, :].astype(np.float32)
    m1_mb = mask1[idx, :, :].astype(np.float32) #fc_mask
    m2_mb = mask2[idx, :].astype(np.float32) #fc_mask
    return x_mb, k_mb, t_mb, m1_mb, m2_mb

def unzip_hyperparam_dict(pdict, num_Event):
    alpha               = pdict['alpha']
    beta                = pdict['beta']
    gamma               = pdict['gamma']
    sigma1              = pdict['sigma1']
    mb_size             = pdict['mb_size']
    keep_prob           = pdict['keep_prob']
    lr_train            = pdict['lr_train']
    h_dim_shared        = pdict['h_dim_shared']
    num_layers_shared   = pdict['num_layers_shared']
    if type(pdict['h_dim_FC']) is list:
        h_dim_FC            = pdict['h_dim_FC']
    else:
        h_dim_FC            = [int(pdict['h_dim_FC']) for _ in range(num_Event)]
    if type(pdict['num_layers_FC']) is list:
        num_layers_FC       = pdict['num_layers_FC']
    else:
        num_layers_FC       = [int(pdict['num_layers_FC']) for _ in range(num_Event)]
    if type(pdict['importancecutoff']) is list:
        importancecutoff    = pdict['importancecutoff']
    else:
        importancecutoff    = [int(pdict['importancecutoff']) for _ in range(num_Event)]
    if type(pdict['importancecutoff']) is list:
        top    = pdict['top']
    else:
        top    = [int(pdict['top']) for _ in range(num_Event)]

    return alpha, beta, gamma, sigma1, mb_size, keep_prob, lr_train, h_dim_shared, num_layers_shared, h_dim_FC, num_layers_FC, importancecutoff, top

##### MAIN SETTING
CV_ITERATION                = 5
RS_ITERATION               = 2
eval_time = [1*12, 3*12, 5*12, 10*12] # x-month evaluation time (for C-index) for testing
val_eval_time = eval_time # Evaluation times for validation

features                    = 'all' # 'all' or 'cox' or 'topxx' or 'pcaxx' or 'feederxx' or 'cutfeederxx'
seed                        = 1234
rs_seed                     = 1234
valid_mode                  = 'ON' # ON / OFF
random_search_mode          = 'ON' # ON / OFF
cv_to_search                = [1, 0, 0, 0, 0] # 0 for "don't perform search on this iteration"
cause_specific_architecture = True


dhclass = import_module("class_DeepHitPlus")
Model_Single = getattr(dhclass, "Model_Single")


##### HYPER-PARAMETERS
itersettings = {
    'iteration': 100000,
    'require_improvement': 5000,
    'check_improvement': 1000
}
active_fn                   = tf.nn.relu
initial_W                   = tf.contrib.layers.xavier_initializer()

paramordering = [
    'alpha',
    'beta',
    'gamma',
    'sigma1',
    'mb_size',
    'importancecutoff',
    'top',
    'num_layers_shared',
    'h_dim_shared',
    'num_layers_FC',
    'h_dim_FC',
    'keep_prob',
    'lr_train'
]

DEFAULT_param_dict = { # Tuple (default_value, cause-specific?)
    'alpha'               : (1.0,                   0), #for log-likelihood loss
    'beta'                : (3.0,                   0), #for ranking loss
    'gamma'               : ([0.0001, 0.0001],      1), #for regularization
    'sigma1'              : (0.1,                   0),
    'mb_size'             : (50,                    0),
    'keep_prob'           : (0.6,                   0),
    'lr_train'            : (1e-4,                  0),
    'h_dim_shared'        : (50,                    0),
    'num_layers_shared'   : (5,                     0),
    'h_dim_FC'            : ([50, 50],              1),
    'num_layers_FC'       : ([5, 5],                1),
    'importancecutoff'    : ([0.001, 0.001],        1),
    'top'                 : ([40, 40],              1)
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
    'top'                 : [20, 40, 60]
}



##### IMPORT DATASET
'''
    num_Category            = max event/censoring time * 1.2 (to make enough time horizon)
    num_Event               = number of evetns i.e. len(np.unique(label))-1
    max_length              = maximum number of measurements
    x_dim                   = data dimension including delta (num_features)
    mask1, mask2            = used for cause-specific network (FCNet structure)
'''


time_interval = 1./1.

(x_dim), (full_data, time, label), (mask1, mask2), full_feat_list = impt.import_dataset_SYNTHETIC(norm_mode = 'standard')

_, num_Event, num_Category  = np.shape(mask1)  # dim of mask1: [subj, num_Event, Num_Category]


FINALCV = np.zeros([num_Event, len(eval_time), CV_ITERATION])
FINALRS = np.zeros([num_Event, len(eval_time), RS_ITERATION])


##### FUNCTIONS FOR NETWORK TRAINING

def get_next_param_dict(searchmode, TRIED_param_dicts, BEST_param_dict, SET_param_dict, DEFAULT_param_dict, num_Event):
    if searchmode == "random":
        # Randomly select hyperparameters
        np.random.seed(rs_seed + s_itr)

        NEXT_param_dict = dict()
        for param in paramordering:
            if DEFAULT_param_dict[param][1]:
                if cause_specific_architecture:
                    NEXT_param_dict[param] = [np.random.choice(SET_param_dict[param]) for _ in range(num_Event)]
                else:
                    NEXT_param_dict[param] = [np.random.choice(SET_param_dict[param])] * num_Event
            else:
                NEXT_param_dict[param] = np.random.choice(SET_param_dict[param])

        k = None


    else:
        NEXT_param_dict = deepcopy(BEST_param_dict)
        k = None

    if k is not None:
        event_searched = k+1
    else:
        event_searched = None

    return NEXT_param_dict, event_searched


def train_deephit(
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
    param_dict,
    x_dim,
    active_fn,
    initial_W,
    s_itr=None,
    best_s_itr=None,
    best_parameter_name=None):
    _, num_Event, num_Category  = np.shape(mask1)  # dim of mask1: [subj, num_Event, Num_Category]

    input_dims                  = { 'x_dim'         : x_dim,
                                    'num_Event'     : num_Event,
                                    'num_Category'  : num_Category }

    alpha, beta, gamma, sigma1, mb_size, keep_prob, lr_train, h_dim_shared, num_layers_shared, h_dim_FC, num_layers_FC, importancecutoff, top = unzip_hyperparam_dict(param_dict, num_Event)

    # NETWORK HYPER-PARMETERS
    network_settings            = { 'h_dim_shared'      : int(h_dim_shared),
                                  'num_layers_shared' : num_layers_shared,
                                  'h_dim_FC'          : h_dim_FC,
                                  'num_layers_FC'     : num_layers_FC,
                                  'active_fn'         : active_fn,
                                  'initial_W'         : initial_W }

    # Put cutoffs into parameter name only if feeder is used
    if features[0:9] =='cutfeeder' or features[0:10] == 'cutpfeeder':
        cutoffsstring = '_cut' + "-".join(str('%04.0f' %(10000*c)) for c in importancecutoff)
    elif features[0:6] == "filter" or features in ["toppfeeder", "topfeeder"]:
        cutoffsstring = '_top' + "-".join(str('%02.0f' %(c)) for c in top)
    else:
        cutoffsstring = ''

    layersstring =  '_hs' + str('%03.0f' %(h_dim_shared)) + '_ns'+ str('%01.0f' %(num_layers_shared)) + \
                    '_hf'+ "-".join(str('%03.0f' %(x)) for x in h_dim_FC) + \
                    '_nf'+ "-".join(str(x) for x in num_layers_FC)

    parameter_name              = 'a' + str('%02.0f' %(10*alpha)) + '_b' + str('%02.0f' %(10*beta)) + \
                                '_s' + str('%02.0f' %(10*sigma1)) + \
                                '_mb'+ str('%03.0f' %(mb_size)) + '_kp' + str('%01.0f' %(10*keep_prob)) + \
                                '_lr'+ str('%.0E' %(lr_train)) + layersstring + \
                                cutoffsstring

    print("######## Parameters:  ", "{:2.0f}".format(s_itr), parameter_name)
    if searchmode == 'random':
        print("########  best so far:", "{:2.0f}".format(best_s_itr), best_parameter_name)

    # MAKE SAVE ROOTS
    file_path = 'output'
    if not os.path.exists(file_path + '/models-' + modes + '/' + parameter_name + '/'):
        os.makedirs(file_path + '/models-' + modes + '/' + parameter_name + '/')
    if not os.path.exists(file_path + '/models-' + modes + '/search/'):
        os.makedirs(file_path + '/models-' + modes + '/search/')

    ##### CREATE DEEPHIT NETWORK
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model_Single(sess, "FHT_DeepHit", mb_size, input_dims, network_settings)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())


    min_valid = 0
    last_improvement = 0
    ### TRAINING
    for itr in range(itersettings['iteration']):
        x_mb, k_mb, t_mb, m1_mb, m2_mb = f_get_minibatch(mb_size, tr_data, tr_label, tr_time, tr_mask1, tr_mask2)
        DATA = (x_mb, k_mb, t_mb)
        MASK = (m1_mb, m2_mb)
        PARAMETERS = (alpha, beta, gamma, sigma1)
        _, loss_curr = model.train(DATA, MASK, PARAMETERS, keep_prob, lr_train)

        if (itr+1) % itersettings['check_improvement'] == 0:
            print('|| Itr: ' + str('%04d' % (itr + 1)) + ' | Loss: ' + str('%.4f' %(loss_curr)))

        if (itr+1) % itersettings['check_improvement'] == 0:
            ### VALIDATION  (based on 1yr eval_time)
            if valid_mode == 'ON':
                ### PREDICTION
                pred = model.predict(va_data)

                ### EVALUATION
                va_result1 = np.zeros([num_Event, len(val_eval_time)])

                for t, t_time in enumerate(val_eval_time):
                    eval_horizon = int(t_time/time_interval)

                    if eval_horizon >= num_Category:
                        print('ERROR: evaluation horizon is out of range')
                        va_result1[:, t] = va_result2[:, t] = -1
                    else:
                        risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until eval_time
                        for k in range(num_Event):
                            va_result1[k, t] = c_index(risk[:,k], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)

                tmp_valid = np.mean(va_result1)
                tmp_valid_event = np.mean(va_result1, axis=1)

                if tmp_valid >  min_valid:
                    min_valid = tmp_valid
                    min_valid_event = tmp_valid_event
                    saver.save(sess, file_path + '/models-' + modes + '/' + parameter_name + '/model_v1')
                    print('updated.... ' + str('%.4f' %(tmp_valid)))
                    last_improvement = itr

        if valid_mode == 'ON':
            if itr - last_improvement >= itersettings['require_improvement']:
                print("No improvement found in a while, stopping optimization.")
                break

    return model, sess, saver, parameter_name, min_valid, min_valid_event, file_path



# K-fold Cross-Validation for Test Data
kf = KFold(n_splits=CV_ITERATION, shuffle=True, random_state=seed)
cv_iter = 0
continue_random_search = True
best_parameter_name = ''

modes = "_".join([features, "rs" + random_search_mode])

BEST_param_dict = {key: copy(DEFAULT_param_dict[key][0]) for key in DEFAULT_param_dict}

for train_index, test_index in kf.split(full_data):
    print()
    print()
    print("############################ CROSS VALIDATION", cv_iter, "############################")

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

    TRIED_param_dicts = []

    min_s_valid = [0,0,0]
    best_s_itr = 0
    for s_itr in range(S_ITERATION):
        s_itr_start_time = timelib.time() # Time how long each random search iteration takes

        cur_param_dict, event_searched = get_next_param_dict(searchmode, TRIED_param_dicts, BEST_param_dict, SET_param_dict, DEFAULT_param_dict, num_Event)
        TRIED_param_dicts.append(deepcopy(cur_param_dict))

        # Get featureset if needed from hyperparameters
        feat_list = get_feat_list(features=features, num_Event=num_Event, data=full_data, full_feat_list=full_feat_list)
        x_dim, data = apply_features(full_data=full_data, full_feat_list=full_feat_list, feat_list=feat_list)

        print("x-dim:", str(x_dim))
        tr_data     = data[train_index]
        tr_time     = time[train_index]
        tr_label    = label[train_index]
        tr_mask1    = mask1[train_index]
        tr_mask2    = mask2[train_index]

        te_data     = data[test_index]
        te_time     = time[test_index]
        te_label    = label[test_index]
        te_mask1    = mask1[test_index]
        te_mask2    = mask2[test_index]

        # Split into training and validation data
        (tr_data,va_data, tr_time,va_time, tr_label,va_label,
         tr_mask1,va_mask1, tr_mask2,va_mask2)  = train_test_split(tr_data, tr_time, tr_label, tr_mask1, tr_mask2, test_size=0.2, random_state=seed)

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
            s_itr=s_itr,
            best_s_itr=best_s_itr,
            best_parameter_name=best_parameter_name)

        print("    Validation performance: " + str(round(valid_perf,3)) + "  |  " + str([round(x,3) for x in valid_perf_event]))

        # Update search
        tmp_s_valid = valid_perf_event

        if (event_searched is None and np.mean(tmp_s_valid) > np.mean(min_s_valid)) or (event_searched is not None and tmp_s_valid[event_searched - 1] > min_s_valid[event_searched - 1]):
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
    feat_list = get_feat_list(features=features, num_Event=num_Event, data=full_data, full_feat_list=full_feat_list)
    x_dim, data = apply_features(full_data=full_data, full_feat_list=full_feat_list, feat_list=feat_list)

    print("x-dim:", str(x_dim))
    tr_data     = data[train_index]
    tr_time     = time[train_index]
    tr_label    = label[train_index]
    tr_mask1    = mask1[train_index]
    tr_mask2    = mask2[train_index]

    te_data     = data[test_index]
    te_time     = time[test_index]
    te_label    = label[test_index]
    te_mask1    = mask1[test_index]
    te_mask2    = mask2[test_index]

    (tr_data,va_data, tr_time,va_time, tr_label,va_label,
     tr_mask1,va_mask1, tr_mask2,va_mask2)  = train_test_split(tr_data, tr_time, tr_label, tr_mask1, tr_mask2, test_size=0.2, random_state=seed)

    input_dims                  = { 'x_dim'         : x_dim,
                                    'num_Event'     : num_Event,
                                    'num_Category'  : num_Category }
    network_settings            = { 'h_dim_shared'      : int(h_dim_shared),
                                  'num_layers_shared' : num_layers_shared,
                                  'h_dim_FC'          : h_dim_FC,
                                  'num_layers_FC'     : num_layers_FC,
                                  'active_fn'         : active_fn,
                                  'initial_W'         : initial_W }

    ##### PREDICTION & EVALUATION
    if valid_mode == 'ON':

        ##### CREATE DEEPHIT NETWORK
        tf.reset_default_graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        model = Model_Single(sess, "FHT_DeepHit", mb_size, input_dims, network_settings)
        saver = tf.train.Saver()

        saver.restore(sess, file_path + '/models-' + modes + '/search/model_rsv1')


    ### PREDICTION
    pred = model.predict(te_data)

    ### EVALUATION
    result1 = np.zeros([num_Event, len(eval_time)])

    for t, t_time in enumerate(eval_time):
        eval_horizon = int(t_time/time_interval)

        if eval_horizon >= num_Category:
            print('ERROR: evaluation horizon is out of range')
            result1[:, t] = result2[:, t] = -1
        else:
            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until eval_time
            for k in range(num_Event):
                result1[k, t] = c_index(risk[:,k], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                # since we compare risk_scores, the true label is that event occurs before time horizon

    FINALCV[:, :, cv_iter] = result1

    ### SAVE RESULTS

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

    ### Save model for later use (feature importance)
    saver.save(sess, file_path + '/results-' + modes + '/' + best_parameter_name + '/models/dhmodel_cv' + str(cv_iter))
    shutil.rmtree(file_path + '/models-' + modes + '/')


    ### PRINT RESULTS
    print('========================================================')
    print('CV_' + str(cv_iter) + ' RESULTS >  (' + best_parameter_name + ', ' + modes + ')')
    print('--------------------------------------------------------')
    print('- CV_' + str(cv_iter) + ' C-INDEX: ')
    print(df1)
    print('--------------------------------------------------------')

    cv_iter += 1
    continue_random_search = False

### FINAL MEAN/STD FOR CV RUN

if random_search_mode == 'ON' or greedy_search_mode == 'ON':
    paramfolderstring = '/' + best_parameter_name
else:
    paramfolderstring = ''

# c-index result
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
