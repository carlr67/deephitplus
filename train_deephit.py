##### FUNCTIONS FOR NETWORK TRAINING

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import random
from copy import deepcopy

from class_DeepHitPlus import Model_Single
from utils_eval import c_index


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


def get_next_param_dict(searchmode, TRIED_param_dicts, BEST_param_dict, SET_param_dict, DEFAULT_param_dict, num_Event, seed):
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

    if searchmode == "random":
        # Randomly select hyperparameters
        np.random.seed(seed)

        NEXT_param_dict = dict()
        for param in paramordering:
            if DEFAULT_param_dict[param][1]:
                NEXT_param_dict[param] = [np.random.choice(SET_param_dict[param]) for _ in range(num_Event)]
            else:
                NEXT_param_dict[param] = np.random.choice(SET_param_dict[param])



    else:
        NEXT_param_dict = deepcopy(BEST_param_dict)

    return NEXT_param_dict



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
    features,
    s_itr=None,
    best_s_itr=None,
    best_parameter_name=None):
    _, num_Event, num_Category  = np.shape(tr_mask1)  # dim of mask1: [subj, num_Event, Num_Category]

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
    feature_mode = features.split("_")[0]

    if feature_mode == "hybrid":
        metric = features.split("_")[1]
        cutofftype = features.split("_")[2]
    else:
        metric = None
        cutofftype = None

    if (feature_mode == "hybrid" and cutofftype == "cut"):
        cutoffsstring = '_cut' + "-".join(str('%04.0f' %(10000*c)) for c in importancecutoff)
    elif feature_mode == "filter" or (feature_mode == "hybrid" and cutofftype == "top"):
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
                    eval_horizon = int(t_time)

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
