# 单GPU

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from tensorflow.contrib import layers, cudnn_rnn
import tensorflow as tf

import numpy as np 

# 结构
# 1. embeddding layer, 
# 2.Bi-LSTM layer, 
# 3.max pooling, 
# 4.FC layer 
# 5.softmax 

class Model():
    def __init__(self,
        x,
        y,
        embedding,
        classes,
        embed_size,
        keep_prob_1,
        keep_prob_2,
        hidden_size,
        num_layers,
        cell_type='lstm'):

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.drop = 1- keep_prob_1
        self.input_x = x
        self.input_y = y


        inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        # left
        with tf.variable_scope('left_encode') as scope:
              left_outputs = self.rnn_encode(inputs,scope)
        # right
        inputs_reverse = tf.reverse(inputs,[1])
        with tf.variable_scope('right_encode') as scope:
              right_outputs = self.rnn_encode(inputs_reverse,scope)  
              right_outputs = tf.reverse(right_outputs,[1])     
        
        # ensemble left, embedding, right to output
        with tf.name_scope("context"):
            shape = [tf.shape(left_outputs)[0], 1, tf.shape(left_outputs)[2]]
            c_left = tf.concat([tf.zeros(shape), left_outputs[:, :-1,:]], 
                    axis=1, name="context_left")
            c_right = tf.concat([right_outputs[:, 1:,:], tf.zeros(shape)], 
                    axis=1, name="context_right")            
        outputs = tf.concat([c_left,inputs,c_right],axis=-1)

        with tf.variable_scope('output_max_pool'):
            w1 = tf.get_variable("ensemble_1", [2*hidden_size+embed_size, hidden_size])
            b1 = tf.get_variable("ensemble_2", [hidden_size])
            yy = tf.einsum('aij,jk->aik',outputs,w1) + b1
            yy = tf.reduce_max(yy, axis=1)


        with tf.variable_scope("dropout"):
            yy = tf.nn.dropout(yy, keep_prob_2)

        with tf.variable_scope('outputs') :
            logits = layers.fully_connected(yy, 256,
                    activation_fn = tf.tanh)
            logits = layers.fully_connected(logits, classes,
                    activation_fn = None)
            self.prediction = tf.argmax(logits, axis=1)

        with tf.name_scope('losses'):            
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,labels=self.input_y)
            self.cost = tf.reduce_mean(loss)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.prediction,self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), 
                    name="accuracy")


    def rnn_encode(self, inputs,scope=None):
        inputs = tf.transpose(inputs,perm=[1,0,2])
        with tf.variable_scope(scope or 'rnn_encoder'):

            if self.cell_type.lower() == 'gru':
                cell = cudnn_rnn.CudnnGRU(self.num_layers,self.hidden_size,dropout= 1-self.drop)
            else:
                cell = cudnn_rnn.CudnnLSTM(self.num_layers,self.hidden_size,dropout= 1-self.drop)

            outputs,_ =  cell(inputs) 

        outputs = tf.transpose(outputs,perm=[1,0,2])
        return outputs


