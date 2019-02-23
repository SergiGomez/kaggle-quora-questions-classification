import os
import pandas as pd
import numpy as np
import keras
import datetime as dt
import sys
import time
import gc

import keras.backend as K
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten, LSTM, Embedding, Dropout, Activation, concatenate
from keras.layers import CuDNNGRU, Conv1D, RNN, Bidirectional, GlobalMaxPooling1D, CuDNNLSTM, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.engine.topology import Layer
from keras.callbacks import *

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics

def train_pred(X, y, test_X, params_embedding, params_model, tokenizer, train_sentences):

    t0 = time.time()

    if params_model['cv-folds'] != 0:

        y_meta = np.zeros(y.shape[0])
        test_y = np.zeros(test_X.shape[0])

        kfold = StratifiedKFold(n_splits = params_model['cv-folds'], shuffle = True, 
                            random_state = np.random.seed(2018))
        cv_scores = []
        i = 1
        for train, val in kfold.split(X, y):

            print("\n-- Training --", flush=True)
            model = create_model(params_embedding, params_model, tokenizer)
            if i == 1:
                print(model.summary())
            print("\t-- Cross Validation --")

            if params_model['callbacks_keras'] == 0:
                callbacks_keras = []
            elif params_model['callbacks_keras'] == 1:
                print("\t-- Setting up the CyclicLR object for the Callback--", flush=True)
                clr = CyclicLR(base_lr=params_model['min_lr'], max_lr=params_model['max_lr'],
                               step_size=300., mode='exp_range',
                               gamma=0.99994)
                callbacks_keras = [clr, ]
            elif params_model['callbacks_keras'] == 2:
                print("\t-- Setting up the ReduceLROnPlateau object for the Callback --", flush=True)
                lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=params_model['min_lr'],
                                       verbose=2)
                callbacks_keras = [lr, ]

            print('Fold number: ', i, flush = True)
            train_X = X[train]
            train_y = y[train]
            # val = val[:56000]
            val_X = X[val]
            val_y = y[val]
            val_sentences = train_sentences[val]
            hist = model.fit(train_X, train_y, batch_size=params_model['batch_size'],
                epochs = params_model['epochs'],
                validation_data = (val_X, val_y), 
                callbacks = callbacks_keras,
                verbose = params_model['verbose'])
            print("\t F1 Training set (per epoch) :", *hist.history['f1'], sep = ', ')
            print("\t F1 Validation set (per epoch) :", *hist.history['val_f1'], sep = ', ')
            y_pred = make_prediction(X = val_X, model_object = model)
            test_y_fold = make_prediction(X = test_X, model_object = model)
            F1 = eval_performance(y_true = val_y, y_pred = y_pred, sentences = val_sentences)
            cv_scores.append(F1)
            y_meta[val] = y_pred.reshape(-1)
            test_y += test_y_fold.reshape(-1)/params_model['cv-folds']
            i += 1
            # Clean up some memory
            del model
            gc.collect()

        best_thresh = threshold_search(y_true = y, y_pred = y_meta)
        test_y_int = (test_y > best_thresh).astype(int) #.ravel()
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)), flush = True)

    else:

        print("\n-- Training --", flush=True)
        model = create_model(params_embedding, params_model, tokenizer)
        print(model.summary())
        print("\t-- Cross Validation --")

        if params_model['callbacks_keras'] == 0:
            callbacks_keras = []
        elif params_model['callbacks_keras'] == 1:
            print("\t-- Setting up the CyclicLR object for the Callback--", flush=True)
            clr = CyclicLR(base_lr=params_model['min_lr'], max_lr=params_model['max_lr'],
                           step_size=300., mode='exp_range',
                           gamma=0.99994)
            callbacks_keras = [clr, ]
        elif params_model['callbacks_keras'] == 2:
            print("\t-- Setting up the ReduceLROnPlateau object for the Callback --", flush=True)
            lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=params_model['min_lr'], verbose=2)
            callbacks_keras = [lr, ]

        indices = range(X.shape[0])
        train_X, val_X, train_y, val_y, ind_train, ind_val = train_test_split(X, y, indices, stratify = y,
                                         test_size = 0.08, random_state=2018)

        model.fit(X, y, batch_size=params_model['batch_size'],
                epochs = params_model['epochs'],
                validation_data = (val_X, val_y),  
                verbose = params_model['verbose'])
        y_pred = make_prediction(X = val_X, model_object = model)
        test_y = make_prediction(X = test_X, model_object = model)
        F1 = eval_performance(y_true = val_y, y_pred = y_pred, sentences = train_sentences[ind_val])
        best_thresh = threshold_search(y_true = val_y, y_pred = y_pred)
        test_y_int = (test_y > best_thresh).astype(int) #.ravel()

    print("-- Training done: %s sec --" % np.round(time.time() - t0,1))
    return test_y_int

def create_model(params_embedding, params_model, tokenizer):

    # Input -- Sequence of integers: (batch, maxlen, 1)
    inp = Input(shape=(params_embedding['maxlen'],))

    # Embedding -- Conversion from integer to feature vector (batch, maxlen, num_features)
    if params_embedding['embed_type'] is not None:
        embedding_matrix = init_embedding_matrix(params_embedding, tokenizer)
        # The embedding_size needs to be updated on dict
        params_embedding['embed_size'] = embedding_matrix.shape[1]
        x = Embedding(input_dim = params_embedding['vocab_size'],
                 output_dim = params_embedding['embed_size'],
                 weights = [embedding_matrix],
                 trainable = params_model['train_embeddings'])(inp)
        if params_model['add_new_embedding']:
            emb_new = Embedding(input_dim = params_embedding['vocab_size'],
                 output_dim = params_embedding['embed_size'],
                 trainable = True)(inp)    
            print("\t Joining embeddings ", flush = True)
            x = (x + 2*emb_new)/3  
        x = SpatialDropout1D(0.1)(x)  
    else:
        x = Embedding(input_dim = params_embedding['vocab_size'],
                 output_dim = params_embedding['embed_size'],
                 trainable = True)(inp)
        x = SpatialDropout1D(0.1)(x)
    
    # Flattening -- Reduction to a flatten tensor (batch, 1, num_hidden_nodes) 
    if params_model['model_name'] == 'baseline_model':
        x = Flatten()(x)
    elif params_model['model_name'] == 'bi_lstm':
        x = Bidirectional(CuDNNLSTM(params_model['hidden_nodes'], return_sequences = True))(x)
        max_pool = GlobalMaxPooling1D()(x)
        avg_pool = GlobalAveragePooling1D()(x)
        atten = Attention(params_embedding['maxlen'])(x)
    elif params_model['model_name'] == 'bi_gru':
        x = Bidirectional(CuDNNGRU(params_model['hidden_nodes'], return_sequences = True))(x)
        max_pool = GlobalMaxPooling1D()(x)
        avg_pool = GlobalAveragePooling1D()(x)
        atten = Attention(params_embedding['maxlen'])(x)
    elif params_model['model_name'] == 'bi_lstm_gru':
        x_lstm = Bidirectional(CuDNNLSTM(params_model['hidden_nodes'], return_sequences = True))(x)
        x_gru = Bidirectional(CuDNNGRU(params_model['hidden_nodes'], return_sequences = True))(x)
        max_pool_lstm = GlobalMaxPooling1D()(x_lstm)
        avg_pool_lstm = GlobalAveragePooling1D()(x_lstm)
        atten_lstm = Attention(params_embedding['maxlen'])(x_lstm)
        max_pool_gru = GlobalMaxPooling1D()(x_gru)
        avg_pool_gru = GlobalAveragePooling1D()(x_gru)
        atten_gru = Attention(params_embedding['maxlen'])(x_gru)
    elif params_model['model_name'] == 'bi_lstm_gru_2':
        x_lstm = Bidirectional(CuDNNLSTM(params_model['hidden_nodes'], return_sequences = True))(x)
        x_gru = Bidirectional(CuDNNGRU(params_model['hidden_nodes'], return_sequences = True))(x_lstm)
        atten_lstm = Attention(params_embedding['maxlen'])(x_lstm)
        atten_gru = Attention(params_embedding['maxlen'])(x_gru)
        max_pool_gru = GlobalMaxPooling1D()(x_gru)
        avg_pool_gru = GlobalAveragePooling1D()(x_gru)

    # Concatening -- concatening tensors
    if params_model['reduction_dim_type'] == 1:  
        if 'bi_lstm_gru' in params_model['model_name']:
            x = concatenate([atten_lstm, atten_gru, max_pool_gru, avg_pool_gru])
        else:
            x = concatenate([atten, max_pool, avg_pool])
    elif params_model['reduction_dim_type'] == 2:  
        if params_model['model_name'] == 'bi_lstm_gru':
            x = concatenate([atten_lstm, atten_gru])
        else:
            x = atten
    elif params_model['reduction_dim_type'] == 3:  
        if params_model['model_name'] == 'bi_lstm_gru':
            x = concatenate([max_pool_gru, avg_pool_gru])
        else:
            x = concatenate([max_pool, avg_pool])
            #Sergi 2019-01-12: Trying to see if it has any impact
            #x = concatenate([avg_pool, max_pool])

    # Densification -- reduction to a fully connected layer (batch, 1, dense_nodes)
    x = Dense(params_model['units_dense'], activation="relu")(x) 
    x = Dropout(params_model['dropout_final_layer'])(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs = inp, outputs = x)
    model.compile(loss='binary_crossentropy', optimizer=params_model['optimizer'], metrics=[f1])

    return model

def init_embedding_matrix(params_embedding, tokenizer):

    embedding_matrix = []
    embed_num = 0
    word_index = tokenizer.word_index
    for embed in params_embedding['embed_type']:
        embedding_file = params_embedding[embed]
        print('\t Processing the pre-trained embedding: ', embed, flush = True)
        if embed == "paragram":
            embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file,
                encoding = "utf8", errors = "ignore") if len(o) > 100)
        else:
            embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file) if len(o) > 100)
        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        emb_size = all_embs.shape[1]
        nb_words = min(params_embedding['vocab_size'], len(word_index))
        embedding_matrix_k = np.random.normal(emb_mean, emb_std, (nb_words, emb_size))
        for word, i in word_index.items():
            if i >= params_embedding['vocab_size']: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix_k[i] = embedding_vector
        if embed_num > 0:
            embedding_matrix += embedding_matrix_k
        else:
            embedding_matrix = embedding_matrix_k
        embed_num+=1

    print("\t Number of embeddings pre-traineds used: ", embed_num, flush = True)
    embedding_matrix /= embed_num
    return embedding_matrix

def make_prediction(X, model_object):

    return model_object.predict([X], batch_size = 1024, verbose=0) 

def eval_performance(y_true, y_pred, sentences, limit = None):

    best_thresh = threshold_search(y_true = y_true, y_pred = y_pred)

    y_pred_int = (y_pred > best_thresh).astype(int).ravel()
     
    res = pd.DataFrame({'real' : y_true, 'pred' : y_pred_int,
                        'y_pred_prob': y_pred.ravel(), 'sentence': sentences})

    res['type'] = 'null'

    res.loc[(res.real == 1) & (res.pred == 1), 'type'] = 'TP'
    res.loc[(res.real == 0) & (res.pred == 0), 'type'] = 'TN'
    res.loc[(res.real == 0) & (res.pred == 1), 'type'] = 'FP'
    res.loc[(res.real == 1) & (res.pred == 0), 'type'] = 'FN'

    if res.loc[res.type == 'null'].shape[0] > 0:
        print("Something wrong on the evaluation, stopping execution")
        print(res.loc[res.type == 'null'].shape[0])
        print(res.loc[res.type == 'null'])
        exit()

    TP = res.loc[res.type == 'TP'].shape[0]
    TN = res.loc[res.type == 'TN'].shape[0]
    FP = res.loc[res.type == 'FP'].shape[0]
    FN = res.loc[res.type == 'FN'].shape[0]

    recall = np.round(TP/(TP + FN),2)
    precision = np.round(TP/(TP + FP),2)

    print("\t\t Num of positives: {0}, Percentage of positives {1} %".format((TP + FN), np.round((TP+FN)*100/res.shape[0],2)))
    print("\t\t Recall: {0} , Precision: {1}".format(recall, precision))

    # False Positives analysis: Those negatives that have lower probability
    df_FP = res.loc[res.type == 'FP'].sort_values(['y_pred_prob'], ascending = [1])
    df_FP = df_FP[['y_pred_prob', 'sentence']].head(n=3)
    print("\t False Positive sentences: \n", df_FP.sentence.values)

    # False Negatives analysis: Those positives that have higher probability
    df_FN = res.loc[res.type == 'FN'].sort_values(['y_pred_prob'], ascending = [0])
    df_FN = df_FN[['y_pred_prob', 'sentence']].head(n=3)
    print("\t False Negative sentences: \n", df_FN.sentence.values)    

    if not limit:
        F1 = np.round(2*recall*precision/(recall + precision),4)
        print("\t\t F1:", np.round(F1, 4))
    else:
        rounds = np.floor(len(y_true)/limit).astype('int')
        F1_list = []
        for round in range(rounds):
            y_true_ = y_true[round*limit:(round+1)*limit]
            y_pred_ = y_pred_int[round*limit:(round+1)*limit]
            F1 = metrics.f1_score(y_true_, y_pred_)
            F1_list.append(F1)
        F1 = np.mean(F1_list)
        print("\t\t F1:", np.round(F1, 4), "over ", rounds, "rounds")

    return F1

def f1(y_true, y_pred):

    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype = 'float32')

def threshold_search(y_true, y_pred):

    thresh_array = np.array([])
    score_array = np.array([])

    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh_i = np.round(thresh, 2)
        thresh_array = np.append(thresh_array, thresh_i)
        score_i = metrics.f1_score(y_true, (y_pred>thresh_i).astype(int))
        #print("F1 score at threshold {0} is {1}".format(thresh_i,score_i))
        score_array = np.append(score_array, score_i)

    pos_max_score = np.argmax(score_array)
    print("\t MAX F1 score at threshold {0} is {1}".format(thresh_array[pos_max_score],score_array[pos_max_score]))

    return thresh_array[pos_max_score]

class Attention(Layer):

    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):

        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

	
def EDA(df):

    print('\n-- EDA --')
    t0 = time.time()
    nwords = df['question_text'].str.split().str.len()
    print('\tMaximum number of words per question: ', nwords.max())
    print("-- EDA done: %s sec --" % np.round(time.time() - t0,1))

def preprocess_data(df, params_embedding):

    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

    def clean_text(x):
        x = str(x)
        for punct in puncts:
            x = x.replace(punct, " %s " % (punct))
        return x

    print('\n-- Data Pre-processing --', flush = True)
    t0 = time.time()

    # cleaning the text
    df['question_text'] = df['question_text'].apply(lambda x: clean_text(x))

    # Fill up the missing values
    all_sentences = df['question_text'].fillna("_na_").values

    # Index number for the training set
    ind_train = df.loc[df['set'] == 'train'].shape[0]

    # 1) Word to integer: tokenize the data into a format that
    # can be used by the word embeddings. For that, we use the Tokenizer utility class
    # which can vectorize a text corpus into a list of integers
    tokenizer = Tokenizer(num_words = params_embedding['vocab_size'])
    tokenizer.fit_on_texts(list(all_sentences))
    X = tokenizer.texts_to_sequences(all_sentences)
    # We pad the input so that all vectors will have the same length
    X = pad_sequences(X, maxlen = params_embedding['maxlen'])

    # Training set pre-processed
    train_X = X[0:ind_train]
    train_sentences = all_sentences[0:ind_train]

    print('\t Length of Training vector: ', len(train_X))
    # Test set pre-processed
    test_X = X[ind_train:]
    print('\t Length of Test vector: ', len(test_X))

    # Target of the training set
    #train_y = df.loc[df['set'] == 'train']['target'].values
    train_y = df['target'].values[0:ind_train]

    print("-- Pre-processing done: %s sec --" % np.round(time.time() - t0,1))

    return train_X, train_y, test_X, tokenizer, train_sentences

# Main code
exec_type =  'azure'#'ec2' 

import init_params
args = init_params.parser.parse_args()
if bool(args.use_wiki) == True:
    embed_list = ['glove', 'wiki']
else:
    embed_list = ['glove']
params_embedding = {'embed_size': 100, 
        'vocab_size': args.vocab_size, # how many unique words to use (i.e num rows in embedding vector) 
        'maxlen': args.maxlen, # max number of words in a question to use
        'embed_type': embed_list#,'wiki', 'paragram']  # If None, it won't take a pre-trained embed
        }
params_model = {'model_name': args.model_name,
        'batch_size' : args.batch_size,
        'epochs' : args.num_epochs,
        'cv-folds': args.cv_folds, # If None, it will train over all the data w/o val.
        'dropout_final_layer':  args.dropout_final_layer,
        'train_embeddings': bool(args.train_embeddings),
        'hidden_nodes': args.hidden_nodes, #number of neurons of the LSTM, dimensionality of the output space
        'verbose': 2,
        'max_lr': args.max_lr,
        'min_lr': args.min_lr,
        'optimizer': args.optimizer,
        'units_dense': args.units_dense,
        'add_new_embedding':bool(args.add_new_embedding),
        'callbacks_keras': args.callbacks_keras,
        'reduction_dim_type': args.reduction_dim_type
        }

print_log = bool(args.print_log)
iter_name = args.iter_name

if print_log:
    todayDate = dt.datetime.today().strftime('%Y%m%d')
    orig_stdout = sys.stdout.flush()
    f = open('training_task_'+ todayDate +'_' + iter_name +'.log', 'w')
    sys.stdout = f

print("\n Model parameters ", flush = True)
for k, v in params_model.items():
    print (k, v)
print("\n Embedding parameters ", flush = True)
for k, v in params_embedding.items():
    print (k, v)


if exec_type == 'ec2':
	path_data = '/home/ec2-user/data/'
elif exec_type == 'kernel_kaggle':
	path_data = '/kaggle/input/'
elif exec_type == 'azure':
    path_data = '/home/sergi.gomez/data/'

path_embeddings = {'wiki': os.path.join(path_data + 'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'),
                   'glove': os.path.join(path_data + 'embeddings/glove.840B.300d/glove.840B.300d.txt'),
                   'paragram': os.path.join(path_data + 'embeddings/paragram_300_sl999/paragram_300_sl999.txt'),
                   'google_news': None}
params_embedding.update(path_embeddings)

target_var = 'target'

t0_exec = time.time()
print("\n-- Data loading  --", flush = True)
train = pd.read_csv(path_data + 'train.csv')
test = pd.read_csv(path_data + 'test.csv')

print("-- Data loading done: %s sec --" % np.round(time.time() - t0_exec,1), flush = True)

train['set'] = 'train'
test['set'] = 'test'
print('\tTraining set shape: {} Rows, {} Columns'.format(*train.shape), flush = True)
print('\tTest set shape: {} Rows, {} Columns'.format(*test.shape), flush = True)

# We add target variable to test just to concatenate both sets for pre-processing
# It doesn't have any meaning, as it will be removed afterwards
test[target_var] = 0
all = pd.concat([train,test],axis=0)

# I can delete fields that Test doesn't need
del test[target_var]
del test['set']

#EDA(df = all)

train_X, train_y, test_X, tokenizer, train_sentences = preprocess_data(df = all,
                                            params_embedding = params_embedding)

print('\tShape train_X: ', train_X.shape, flush = True)
print('\tShape train_y: ', train_y.shape, flush = True)
print('\tShape test_X: ', test_X.shape, flush = True)

test_y_int = train_pred(X = train_X, y = train_y, test_X = test_X,
                    params_embedding = params_embedding, 
                    params_model = params_model, 
                    tokenizer = tokenizer,
                    train_sentences = train_sentences)

if exec_type == 'kernel_kaggle':
    sub_df = pd.DataFrame({"qid":test["qid"].values})
    sub_df['prediction'] = test_y_int
    sub_df.to_csv("submission.csv", index=False)

print("-- Execution done: %s sec --" % np.round(time.time() - t0_exec,1), flush = True)
