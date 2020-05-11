'''
Modifcation List:
    - Corrected error (no more layer combining inputs before cause-specific subnetworks)
    - Changed layer numbers and sizes to original DeepHit paper
    - Added sigma1 as a settable hyperparameter
    - Added hyperparameter c, adapted to feeder architecture
'''

import numpy as np
import tensorflow as tf
import random

from tensorflow.contrib.layers import fully_connected as FC_Net

import utils_network as utils

_EPSILON = 1e-08


##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.log(x + _EPSILON)

def div(x, y):
    return tf.div(x, (y + _EPSILON))


class Model_Single:
    def __init__(self, sess, name, mb_size, input_dims, network_settings):
        self.sess               = sess
        self.name               = name
        self.mb_size            = mb_size

        # INPUT DIMENSIONS
        self.x_dim_lst          = input_dims['x_dim']
        self.x_dim              = sum(self.x_dim_lst)

        self.num_Event          = input_dims['num_Event']
        self.num_Category       = input_dims['num_Category']

        # NETWORK HYPER-PARMETERS
        self.h_dim_shared       = network_settings['h_dim_shared']
        self.num_layers_shared  = network_settings['num_layers_shared']
        self.h_dim_FC           = network_settings['h_dim_FC']
        self.num_layers_FC      = network_settings['num_layers_FC']

        self.active_fn          = network_settings['active_fn']
        self.initial_W          = network_settings['initial_W']

        self._build_net()


    def _build_net(self):
        with tf.variable_scope(self.name):
            #### PLACEHOLDER DECLARATION
            self.lr_rate     = tf.placeholder(tf.float32)
            self.keep_prob   = tf.placeholder(tf.float32)                                                      #keeping rate
            self.a           = tf.placeholder(tf.float32)
            self.b           = tf.placeholder(tf.float32)
            self.c           = tf.placeholder(tf.float32, shape=[self.num_Event])
            self.sigma1      = tf.placeholder(tf.float32)                                                      # sigma hyperparameter

            self.x           = tf.placeholder(tf.float32, shape=[None, self.x_dim])
            self.k           = tf.placeholder(tf.float32, shape=[None, 1])                                     #event/censoring label (censoring:0)
            self.t           = tf.placeholder(tf.float32, shape=[None, 1])

            self.fc_mask1    = tf.placeholder(tf.float32, shape=[None, self.num_Event, self.num_Category])     #for Loss 1
            self.fc_mask2    = tf.placeholder(tf.float32, shape=[None, self.num_Category])                     #for Loss 2


            ##### SHARED SUBNETWORK w/ FCNETS
            n_inputs = self.x_dim_lst[0]
            shared_out = utils.create_FCNet(self.x, int(self.num_layers_shared), int(self.h_dim_shared), self.active_fn, int(self.h_dim_shared), self.active_fn, self.initial_W, self.keep_prob)
            #last_x = self.x  #for residual connection

            #inputs = tf.concat([last_x, shared_out], axis=1)

            #1 layer for combining inputs
            # h = FC_Net(inputs, self.h_dim_shared, activation_fn=self.active_fn, weights_initializer=self.initial_W, scope="Layer1")
            # h = tf.nn.dropout(h, keep_prob=self.keep_prob)

            #(num_layers_FC-1) layers for cause-specific (num_Event subNets)
            # !!! Changed to num_layers_FC layers for cause-specific subnetwork
            out = []
            for _event in range(self.num_Event):
                start = sum(self.x_dim_lst[0:(_event + 1)])
                end = sum(self.x_dim_lst[0:(_event + 2)])
                important_x = self.x[:, start:end]  #for residual connection

                inputs = tf.concat([important_x, shared_out], axis=1)
                n_inputs = self.x_dim_lst[0] + end - start

                cs_out = utils.create_FCNet(inputs, int(self.num_layers_FC[_event]), int(self.h_dim_FC[_event]), self.active_fn, int(self.h_dim_FC[_event]), self.active_fn, self.initial_W, self.keep_prob)

                out.append(cs_out)

            # out = tf.stack(out, axis=1) # stack referenced on subject
            # out = tf.reshape(out, [-1, sum(self.h_dim_FC)])
            out = tf.concat(out, axis=1)

            out = tf.nn.dropout(out, keep_prob=self.keep_prob)

            out = FC_Net(out, self.num_Event * self.num_Category, activation_fn=tf.nn.softmax, weights_initializer=self.initial_W, scope="Output")
            self.out = tf.reshape(out, [-1, self.num_Event, self.num_Category])


            ##### GET LOSS FUNCTIONS
            self.loss_Log_Likelihood()      #get loss1: Log-Likelihood loss
            self.loss_Ranking()             #get loss2: Ranking loss

            self.LOSS_TOTAL = self.a*self.LOSS_1 + self.b*self.LOSS_2
            self.solver = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.LOSS_TOTAL)


    ### LOSS-FUNCTION 1 -- Log-likelihood loss
    def loss_Log_Likelihood(self):
        I_1 = tf.sign(self.k)

        #for uncenosred: log P(T=t,K=k|x)
        tmp1 = tf.reduce_sum(tf.reduce_sum(self.fc_mask1 * self.out, reduction_indices=2), reduction_indices=1, keep_dims=True)
        tmp1 = I_1 * log(tmp1)

        #for censored: log \sum P(T>t|x)
        tmp2 = tf.reduce_sum(tf.reduce_sum(self.fc_mask1 * self.out, reduction_indices=2), reduction_indices=1, keep_dims=True)
        tmp2 = (1. - I_1) * log(tmp2)

        self.LOSS_1 = - tf.reduce_mean(tmp1 + tmp2)


    ### LOSS-FUNCTION 2 -- Ranking loss
    def loss_Ranking(self):
        # sigma1 = tf.constant(0.1, dtype=tf.float32) # replaced by self.sigma1

        eta = []
        for e in range(self.num_Event):
            one_vector = tf.ones_like(self.t, dtype=tf.float32)
            I_2 = tf.cast(tf.equal(self.k, e+1), dtype = tf.float32) #indicator for event
            I_2 = tf.diag(tf.squeeze(I_2))
            tmp_e = tf.reshape(tf.slice(self.out, [0, e, 0], [-1, 1, -1]), [-1, self.num_Category]) #event specific joint prob.

            R = tf.matmul(tmp_e, tf.transpose(self.fc_mask2)) #no need to divide by each individual dominator
            # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

            diag_R = tf.reshape(tf.diag_part(R), [-1, 1])
            R = tf.matmul(one_vector, tf.transpose(diag_R)) - R # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
            R = tf.transpose(R)                                 # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

            T = tf.nn.relu(tf.sign(tf.matmul(one_vector, tf.transpose(self.t)) - tf.matmul(self.t, tf.transpose(one_vector))))
            # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

            T = tf.matmul(I_2, T) # only remains T_{ij}=1 when event occured for subject i

            tmp_eta = tf.reduce_mean(T * tf.exp(-R/self.sigma1), reduction_indices=1, keep_dims=True)

            eta.append(tmp_eta)
        eta = tf.stack(eta, axis=1) #stack referenced on subjects
        eta = tf.reduce_mean(tf.reshape(eta, [-1, self.num_Event]), reduction_indices=1, keep_dims=True)

        self.LOSS_2 = tf.reduce_sum(eta) #sum over num_Events


    def get_cost(self, DATA, MASK, PARAMETERS, keep_prob, lr_train):
        (x_mb, k_mb, t_mb) = DATA
        (m1_mb, m2_mb) = MASK
        (alpha, beta) = PARAMETERS
        return self.sess.run(self.LOSS_TOTAL,
                             feed_dict={self.x:x_mb, self.k:k_mb, self.t:t_mb, self.fc_mask1: m1_mb, self.fc_mask2:m2_mb,
                                        self.a:alpha, self.b:beta, self.keep_prob:keep_prob, self.lr_rate:lr_train})

    def train(self, DATA, MASK, PARAMETERS, keep_prob, lr_train):
        (x_mb, k_mb, t_mb) = DATA
        (m1_mb, m2_mb) = MASK
        (alpha, beta, gamma, sigma1) = PARAMETERS
        return self.sess.run([self.solver, self.LOSS_TOTAL],
                             feed_dict={self.x:x_mb, self.k:k_mb, self.t:t_mb, self.fc_mask1: m1_mb, self.fc_mask2:m2_mb,
                                        self.a:alpha, self.b:beta, self.c:gamma, self.sigma1:sigma1, self.keep_prob:keep_prob, self.lr_rate:lr_train})

    def predict(self, x_test, keep_prob=1.0):
        return self.sess.run(self.out, feed_dict={self.x: x_test, self.keep_prob: keep_prob})